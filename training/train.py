import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from models.nvp_cadex import NVP_v2_5_frame
from utils.images import show_image, inverse_normalize, mesh_plotly
from models.perceptual_loss import PerceptualLoss, MaskedPerceptualLoss
from pytorch3d.loss import (
    chamfer_distance,
)
from pytorch3d.io import save_obj
from datasets.dataset_meshes import DatasetMeshWithImages, collate_fn
from render import render_meshes
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vgg19, VGG19_Weights
import sys
from pympler import asizeof
import wandb
import time
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image



def print_memory_usage(local_vars):
    for name, value in local_vars.items():
        print(
            f'{name} uses {asizeof.asizeof(value)} bytes (deep size) and {sys.getsizeof(value)} bytes (shallow size)')


class Training:
    def __init__(self, device, params):
        self.device = device
        self.decoder = self.get_homeomorphism_model().to(self.device)
        self.image_encoder = vgg19(weights=VGG19_Weights.DEFAULT).eval().to(self.device)
        self.perceptual_loss = MaskedPerceptualLoss().to(self.device)
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-7)
        self.chamfer_weight = params['chamfer_weight']
        self.render_weight = params['render_weight']
        self.log = params['log']


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
            if directory == "out":
                continue
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

    def train_epoch(self, dataloader):
        self.decoder.train()
        total_chamfer_loss, total_render_loss, total_loss_epoch = 0, 0, 0
        time1 = time.time()
        for i, item in enumerate(dataloader):
            self.optimizer.zero_grad()
            # get data
            target_meshes, template_vertices, template_faces, template_textures, images, camera_parameters, centers, scales, frames, num_points, names = item
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

            # sample points from meshes
            numberOfSampledPoints = 5000

            # sample meshes
            predicted_sampled = sample_points_from_meshes(predicted_meshes, numberOfSampledPoints).to(self.device)
            target_sampled = sample_points_from_meshes(target_meshes, numberOfSampledPoints).to(self.device)
            chamfer_loss, _ = chamfer_distance(target_sampled, predicted_sampled)

            # transform mesh back for rendering
            transformed_mesh = self.transform_meshes(predicted_meshes, centers, scales)

            # render images from predicted mesh
            rendered_images, masks = render_meshes(transformed_mesh, camera_parameters, self.device)
            rendered_images = rendered_images.detach()
            masks = masks.detach()

            # get render loss
            render_loss = self.perceptual_loss(rendered_images, images, masks)

            print(chamfer_loss, render_loss)

            total_loss = chamfer_loss * self.chamfer_weight + render_loss * self.render_weight

            total_loss.backward()

            total_chamfer_loss += chamfer_loss
            total_render_loss += render_loss
            total_loss_epoch += total_loss
            self.optimizer.step()
            self.scheduler.step()
            if i%10 == 0 and i > 0:
                if self.log:
                    wandb.log({"chamfer_loss_training": total_chamfer_loss/10, "render_loss_training": total_render_loss/10, "total_loss_training": total_loss/10})
                    total_chamfer_loss, total_render_loss, total_loss_epoch = 0, 0, 0

        print("Time training epoch: ", time.time() - time1)

    def validate(self, dataloader):
        self.decoder.eval()
        total_chamfer_loss, total_render_loss, total_loss_epoch = 0, 0, 0
        time1 = time.time()
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                # get data
                print(i)
                target_meshes, template_vertices, template_faces, template_textures, images, camera_parameters, centers, scales, frames, num_points, names = item
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


                # sample points from meshes
                numberOfSampledPoints = 5000

                # sample meshes
                predicted_sampled = sample_points_from_meshes(predicted_meshes, numberOfSampledPoints).to(self.device)
                target_sampled = sample_points_from_meshes(target_meshes, numberOfSampledPoints).to(self.device)
                chamfer_loss, _ = chamfer_distance(target_sampled, predicted_sampled)

                # transform mesh back for rendering
                transformed_mesh = self.transform_meshes(predicted_meshes, centers, scales)

                # render images from predicted mesh
                rendered_images, masks = render_meshes(transformed_mesh, camera_parameters, self.device)
                rendered_images = rendered_images.detach()
                masks = masks.detach()

                # get render loss
                render_loss = self.perceptual_loss(rendered_images, images, masks)

                total_chamfer_loss += chamfer_loss
                total_render_loss += render_loss
                if i%10 == 0 and i > 0:
                    if self.log:
                        wandb.log({"chamfer_loss_validation": total_chamfer_loss/10, "render_loss_validation": total_render_loss/10})
                        total_chamfer_loss, total_render_loss, total_loss_epoch = 0, 0, 0

                # log meshes to wandb
                if i == 0 and batch_size >= 2:
                    for k in range(2):
                        pred = mesh_plotly(predicted_mesh_vertices[k], template_vertices[k])
                        gt = mesh_plotly(target_meshes[k].verts_packed(), target_meshes[k].faces_packed())
                        wandb.log({f"predicted_mesh{k}": wandb.Plotly(pred)})
                        wandb.log({f"target_mesh{k}": wandb.Plotly(gt)})

        print("Time validation epoch: ", time.time() - time1)

    def transform_meshes(self, meshes, centers, scales):
        transformed_verts = []
        for i, mesh in enumerate(meshes):
            verts = mesh.verts_packed()
            verts = verts * scales[i].to(self.device)
            verts = verts + centers[i].to(self.device)
            verts = verts / 1000
            transformed_verts.append(verts.float())

        transformed_meshes = Meshes(verts=transformed_verts, faces=meshes.faces_list(), textures=meshes.textures)

        return transformed_meshes

    def save_meshes(self, train_loader, valid_loader, data_path):
        self.decoder.eval()
        with torch.no_grad():
            for i, item in enumerate(train_loader):
                print(i)

                # get data
                target_meshes, template_vertices, template_faces, template_textures, images, camera_parameters, centers, scales, frames, num_points, names = item
                batch_size = len(frames)

                # get features from images
                image_features = self.image_encoder.forward(images)
                reshaped_features = image_features.view(batch_size, 4000)

                # predict deformation
                coordinates = self.decoder.forward(reshaped_features, template_vertices)
                coordinates = coordinates.reshape(batch_size, 9000, 3)

                # create new source mesh
                predicted_mesh_vertices = [coordinates[s][:num_points[s]] for s in range(batch_size)]

                for j in range(len(frames)):
                    folder_path = os.path.join(data_path, "out/train", names[j])
                    os.makedirs(folder_path, exist_ok=True)
                    filename = os.path.join(folder_path, f"mesh_f{frames[j]}.obj")
                    save_obj(filename, predicted_mesh_vertices[j], faces=template_faces[j])

            for i, item in enumerate(valid_loader):
                print(i)

                # get data
                target_meshes, template_vertices, template_faces, template_textures, images, camera_parameters, centers, scales, frames, num_points, names = item
                batch_size = len(frames)

                # get features from images
                image_features = self.image_encoder.forward(images)
                reshaped_features = image_features.view(batch_size, 4000)

                # predict deformation
                coordinates = self.decoder.forward(reshaped_features, template_vertices)
                coordinates = coordinates.reshape(batch_size, 9000, 3)

                # create new source mesh
                predicted_mesh_vertices = [coordinates[s][:num_points[s]] for s in range(batch_size)]

                for j in range(len(frames)):
                    folder_path = os.path.join(data_path, "out/validation", names[j])
                    os.makedirs(folder_path, exist_ok=True)
                    filename = os.path.join(folder_path, f"mesh_f{frames[j]}.obj")
                    save_obj(filename, predicted_mesh_vertices[j], faces=template_faces[j])


def main(log):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = '../data'
    epochs = 10
    epoch_start = 0
    batch_size = 4
    lr = 5e-5
    weight_decay = 5e-7
    chamfer_weigth = 10000
    render_weight = 1
    paramters = {"lr": lr, "weight_decay": weight_decay, "chamfer_weight": chamfer_weigth, "render_weight": render_weight, "log": log}

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="DeformingTemplate",

            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "model": "homemoorphism, vgg19, 4 images",
                "weight_decay": weight_decay,
                "chamfer_weight": chamfer_weigth,
                "render_weight": render_weight,
                "batch_size": batch_size,
                "dataset": "Couch_T1",
                "epochs": epochs,
            }
        )

    session = Training(device, paramters)
    # get data
    train_loader, valid_loader = session.load_data(data_path, batch_size)
    print(len(train_loader), len(valid_loader))

    for epoch in range(epoch_start, epochs):
        session.train_epoch(train_loader)
        session.validate(valid_loader)

    session.save_meshes(train_loader, valid_loader, data_path)


if __name__ == "__main__":
    log = True
    main(log)