import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from models.nvp_cadex import NVP_v2_5_frame
from models.perceptual_loss import PerceptualLoss
from pytorch3d.loss import (
    chamfer_distance,
)
from datasets.dataset_meshes import DatasetMeshWithImages, collate_fn
from render import render_meshes
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vgg19, VGG19_Weights
import sys
from pympler import asizeof
from torch.profiler import profile, record_function, ProfilerActivity
import wandb


def print_memory_usage(local_vars):
    for name, value in local_vars.items():
        print(
            f'{name} uses {asizeof.asizeof(value)} bytes (deep size) and {sys.getsizeof(value)} bytes (shallow size)')


class Training:
    def __init__(self, device):
        self.device = device
        self.decoder = self.get_homeomorphism_model().to(self.device)
        self.image_encoder = vgg19(weights=VGG19_Weights.DEFAULT).eval().to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device).eval()
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=5e-6, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-7)
        self.chamfer_weight = 50
        self.render_weight = 1.0
        self.epsilon = 1e-6
        self.log = False
        if self.log:
            wandb.init(
                # set the wandb project where this run will be logged
                project="DeformingTemplate",

                # track hyperparameters and run metadata
                config={
                    "learning_rate": 0.02,
                    "architecture": "homemoorphism, vgg19, 4 images",
                    "dataset": "Pillows",
                    "epochs": 10,
                }
            )



    def print_memory_usage_class(self):
        for attr, value in self.__dict__.items():
            print(f'{attr} uses {asizeof.asizeof(value)} bytes (deep size)')

    def get_homeomorphism_model(self):
        n_layers = 6
        # dimension of the code
        feature_dims = 128
        hidden_size = [128, 64, 32, 32, 32]
        # the dimension of the coordinates to be projected onto
        proj_dims = 128
        code_proj_hidden_size = [128, 128, 128]
        proj_type = 'simple'
        block_normalize = True
        normalization = False

        c_dim = 128
        hidden_dim = 128
        homeomorphism_decoder = NVP_v2_5_frame(n_layers=n_layers, feature_dims=4000,
                                               hidden_size=hidden_size,
                                               proj_dims=proj_dims, code_proj_hidden_size=code_proj_hidden_size,
                                               proj_type=proj_type,
                                               block_normalize=block_normalize, normalization=normalization)
        return homeomorphism_decoder

    def load_data(self, data_path, batch_size):
        datasets = []
        for directory in os.listdir(data_path):
            current_set = DatasetMeshWithImages(os.path.join(data_path, directory), self.device)
            datasets.append(current_set)
        combined_dataset = ConcatDataset(datasets)
        # split the dataset into training and validation
        total_size = len(combined_dataset)
        validation_fraction = 0.2
        valid_size = int(total_size * validation_fraction)
        train_size = total_size - valid_size  # Rest will be for training
        train_dataset, valid_dataset = random_split(combined_dataset, [train_size, valid_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda b, device=self.device: collate_fn(b, device), drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda b, device=self.device: collate_fn(b, device), drop_last=True)

        return train_dataloader, valid_dataloader

    def transform_meshes(self, meshes, scales, centers):
        offsets = torch.zeros((meshes.verts_packed().shape[0], 3)).to(self.device)
        scales_tensor = torch.zeros((meshes.verts_packed().shape[0], 1)).to(self.device)
        total_verts = 0
        for i in range(len(meshes.verts_list())):
            num_vertices = meshes.verts_list()[i].shape[0]
            offsets[total_verts:total_verts + num_vertices, :] -= centers[i][0]
            scales_tensor[i, :] = 1/scales[i][0]/1000

            total_verts += num_vertices

        # apply the transformation
        meshes.offset_verts_(offsets)
        meshes.scale_verts_(scales_tensor)

        return meshes

    def train_epoch(self, dataloader):
        self.get_homeomorphism_model().train()
        total_chamfer_loss, total_render_loss, total_loss_epoch = 0, 0, 0
        for i, item in enumerate(dataloader):
            self.optimizer.zero_grad()
            # get data
            target_meshes, template_vertices, template_faces, template_textures, images, camera_parameters, centers, scales, frames, num_points, num_vertices = item
            print(i)
            batch_size = len(frames)

            # get features from images
            image_features = self.image_encoder.forward(images)
            reshaped_features = image_features.view(batch_size, 4000)

            # predict deformation
            coordinates = self.decoder.forward(reshaped_features, template_vertices)
            coordinates = coordinates.reshape(batch_size, 9000, 3)

            # create new source mesh
            predicted_mesh_vertices = [coordinates[s][:num_points[s]] for s in range(batch_size)]
            predicted_meshes = Meshes(verts=predicted_mesh_vertices, faces=template_faces, textures=template_textures)
            #print(template_vertices.shape, template_faces.shape)
            temp_mesh = Meshes(verts=[template_vertices[s][:num_points[s]] for s in range(batch_size)], faces=template_faces)

            print("target", target_meshes.get_bounding_boxes())
            print("temp", predicted_meshes.get_bounding_boxes())

            # sample points from meshes
            numberOfSampledPoints = 5000
            areas = predicted_meshes.faces_areas_packed()

            if not torch.any(areas <= 0):
                # sample meshes
                predicted_sampled = sample_points_from_meshes(predicted_meshes, numberOfSampledPoints).to(self.device)
                target_sampled = sample_points_from_meshes(target_meshes, numberOfSampledPoints).to(self.device)
                chamfer_loss, _ = chamfer_distance(target_sampled, predicted_sampled)
            else:
                chamfer_loss = 0
                print("Zero areas")


            # transform mesh back for rendering
            transformed_mesh = self.transform_meshes(predicted_meshes, scales, centers)
            # render images from predicted mesh
            rendered_images = render_meshes(transformed_mesh, camera_parameters, self.device)

            # get render loss
            render_loss = self.perceptual_loss.forward(images, rendered_images)

            # area loss
            area_loss = 1e-8*torch.sum(1 / (areas + self.epsilon))

            print(chamfer_loss, render_loss, area_loss)

            total_loss = chamfer_loss * self.chamfer_weight + render_loss * self.render_weight + area_loss

            total_loss.backward()
            total_chamfer_loss += chamfer_loss/batch_size
            total_render_loss += render_loss/batch_size
            total_loss_epoch += total_loss/batch_size
            self.optimizer.step()
            self.scheduler.step()
            if self.log:
                wandb.log({"chamfer_loss": chamfer_loss, "render_loss": render_loss, "total_loss": total_loss})

        return total_chamfer_loss, total_render_loss, total_loss_epoch

    def validate(self, data_loader):
        self.get_homeomorphism_model().eval()
        total_chamfer_loss, total_render_loss, total_loss_epoch = 0, 0, 0
        for i, item in enumerate(data_loader):
            self.optimizer.zero_grad()
            # get data
            target_meshes, template_vertices, template_faces, template_textures, images, camera_parameters, centers, scales, frames, num_points, num_vertices = item
            batch_size = len(frames)

            # get features from images
            image_features = self.image_encoder.forward(images)

            # predict deformation
            coordinates = self.decoder.forward(image_features, template_vertices)
            coordinates = coordinates.reshape(batch_size, 9000, 3)

            # create new source mesh
            predicted_mesh_vertices = [coordinates[s][:num_points[s]] for s in range(batch_size)]
            predicted_meshes = Meshes(verts=predicted_mesh_vertices, faces=template_faces, textures=template_textures)

            # sample points from meshes
            numberOfSampledPoints = 5000
            predicted_sampled = sample_points_from_meshes(predicted_meshes, numberOfSampledPoints).to(self.device)
            target_sampled = sample_points_from_meshes(target_meshes, numberOfSampledPoints).to(self.device)

            chamfer_loss, _ = chamfer_distance(target_sampled, predicted_sampled)

            # transform mesh back for rendering
            transformed_mesh = self.transform_meshes(predicted_meshes, scales, centers)


            # render images from predicted mesh
            rendered_images = render_meshes(transformed_mesh, camera_parameters, self.device)

            # get render loss
            render_loss = 0
            render_loss += self.perceptual_loss.forward(images, rendered_images)

            total_loss = chamfer_loss * self.chamfer_weight + render_loss * self.render_weight

            total_chamfer_loss += chamfer_loss / batch_size
            total_render_loss += render_loss / batch_size
            total_loss_epoch += total_loss / batch_size


        return total_chamfer_loss, total_render_loss, total_loss_epoch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Nepochs = 500000
    epoch_start = 0
    batch_size = 1
    data_path = '../data'

    # training session
    session = Training(device)

    # get data
    train_loader, valid_loader = session.load_data(data_path, batch_size)
    print(len(train_loader), len(valid_loader))

    for epoch in range(epoch_start, 2):
        chamfer_loss, render_loss, total_loss = session.train_epoch(train_loader)
        #print(f"Epoch: {epoch}, Chamfer Loss: {chamfer_loss}, Render Loss: {render_loss}, Total Loss: {total_loss}")
        if epoch % 100 == 0:
            chamfer_loss, render_loss, total_loss = session.validate(valid_loader)
            print(f"Epoch: {epoch}, Chamfer Loss: {chamfer_loss}, Render Loss: {render_loss}, Total Loss: {total_loss}")


if __name__ == "__main__":
    main()