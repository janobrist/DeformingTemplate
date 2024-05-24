import os

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from models.nvp_cadex import NVP_v2_5_frame
from utils.visualization import show_image, inverse_normalize, mesh_plotly
from models.perceptual_loss import PerceptualLoss, MaskedPerceptualLoss
from models.feature_extraction import ForceFeatures
from pytorch3d.loss import (
    chamfer_distance,
)
from pytorch3d.io import save_obj
from datasets.dataset_meshes import DatasetMeshWithImages, collate_fn
from training.render import render_meshes
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights #, VGG19_Weights, vit_l_32, ViT_L_32_Weights
import sys
from pympler import asizeof
import wandb
import torch.nn.functional as F
import time


def print_memory_usage(local_vars):
    for name, value in local_vars.items():
        print(
            f'{name} uses {asizeof.asizeof(value)} bytes (deep size) and {sys.getsizeof(value)} bytes (shallow size)')


class Training:
    def __init__(self, device, args):
        self.device = device

        # models
        self.force_features = args.force_features
        self.num_cameras = len(args.cameras)
        self.decoder = self.get_homeomorphism_model().to(self.device)
        #self.image_encoder = vgg19(weights=VGG19_Weights.DEFAULT).eval().to(self.device)
        #self.image_encoder = vit_l_32(weights=ViT_L_32_Weights.DEFAULT).eval().to(self.device)
        self.image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(self.device)
        #self.perceptual_loss = MaskedPerceptualLoss().to(self.device)
        self.force_encoder = ForceFeatures().to(self.device)


        # optimizer and weights
        params = list(self.force_encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=5e-6)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-7)
        self.chamfer_weight = args.chamfer_weight
        self.normals_weight = args.normals_weight
        self.roi_chamfer_weight = args.roi_chamfer_weight
        self.roi_normals_weight = args.roi_normals_weight
        self.log_meshes = None

    def print_memory_usage_class(self):
        for attr, value in self.__dict__.items():
            print(f'{attr} uses {asizeof.asizeof(value)} bytes (deep size)')

    def get_homeomorphism_model(self):
        n_layers = 6
        # dimension of the code
        if self.force_features:
            feature_dims = 1000*self.num_cameras + 64
        else:
            feature_dims = 1000*self.num_cameras

        hidden_size = [128, 64, 32, 32, 32]
        # the dimension of the coordinates to be projected onto
        proj_dims = 128
        code_proj_hidden_size = [128, 128, 128]
        proj_type = 'simple'
        block_normalize = True
        normalization = False

        homeomorphism_decoder = NVP_v2_5_frame(n_layers=n_layers, feature_dims=feature_dims,
                                               hidden_size=hidden_size,
                                               proj_dims=proj_dims, code_proj_hidden_size=code_proj_hidden_size,
                                               proj_type=proj_type,
                                               block_normalize=block_normalize, normalization=normalization)
        return homeomorphism_decoder

    def load_data(self, data_paths, batch_size, cameras):
        datasets = []
        for data_path in data_paths:
            for directory in os.listdir(data_path):
                if directory == "out":
                    continue
                current_set = DatasetMeshWithImages(os.path.join(data_path, directory), cameras, self.device)
                datasets.append(current_set)
        combined_dataset = ConcatDataset(datasets)
        # split the dataset into training and validation
        total_size = len(combined_dataset)
        validation_fraction = 0.15
        valid_size = int(total_size * validation_fraction)
        train_size = total_size - valid_size  # Rest will be for training
        train_dataset, valid_dataset = random_split(combined_dataset, [train_size, valid_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda b, device=self.device: collate_fn(b, device), drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda b, device=self.device: collate_fn(b, device), drop_last=True)


        return train_dataloader, valid_dataloader

    def get_roi_meshes(self, meshes, target_position, rotation_matrices, distance_threshold, cosine_threshold):
        face_indices_array = []
        cosine_threshold = torch.tensor(cosine_threshold)
        for i, mesh in enumerate(meshes):
            pos = target_position[i].unsqueeze(0).to(self.device)
            distances_squared = torch.sum((mesh.verts_packed() - pos) ** 2, dim=1)

            # get vertices
            vertices_mask = distances_squared < distance_threshold ** 2
            verts = mesh.verts_packed()[vertices_mask]
            vertices_indices0 = vertices_mask.nonzero(as_tuple=True)[0]

            # get directions
            boundaries = mesh.get_bounding_boxes()
            if boundaries[0, 1, 1] < pos[0, 1]:
                directions = (verts - pos)
            else:
                directions = - (verts - pos)
            directions /= directions.norm(p=2, dim=1, keepdim=True)

            # filter for directions
            target_direction = rotation_matrices[i][:, 0].to(self.device)  # Adjust the index based on actual target
            dot_products = torch.matmul(directions, target_direction)
            angle_threshold = torch.cos(torch.deg2rad(torch.tensor(cosine_threshold)))
            similar_vectors = dot_products > angle_threshold

            directions_mask = similar_vectors.nonzero(as_tuple=True)[0]

            vertices_indices = vertices_indices0[directions_mask]

            if vertices_indices.shape[0] == 0:
                if boundaries[0, 1, 1] - boundaries[0, 1, 0] > 0.1:
                    directions *= -1
                    dot_products = torch.matmul(directions, target_direction)
                    angle_threshold = torch.cos(torch.deg2rad(torch.tensor(cosine_threshold)))
                    similar_vectors = dot_products > angle_threshold

                    directions_mask = similar_vectors.nonzero(as_tuple=True)[0]

                    vertices_indices = vertices_indices0[directions_mask]

            # get faces
            faces = mesh.faces_packed()
            faces_mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=self.device)
            for index in vertices_indices:
                faces_mask |= (faces == index).any(dim=1)

            face_indices = faces_mask.nonzero(as_tuple=True)[0]
            face_indices_array.append([face_indices])

        selected_meshes = meshes.submeshes(face_indices_array)

        return selected_meshes

    def get_closest_vertices(self, target, predicted):
        # Expand dimensions to calculate pairwise distances
        predicted = predicted.unsqueeze(2)  # Shape: (batch, num_verts, 1, 3)
        target = target.unsqueeze(1)  # Shape: (batch, 1, num_verts, 3)

        # Compute squared Euclidean distances
        distances_squared = torch.sum((predicted - target) ** 2,
                                      dim=-1)  # Shape: (batch, num_verts, num_verts)

        # Find the indices of the minimum distances along the target_vertices dimension
        closest_indices = torch.argmin(distances_squared, dim=2)

        return closest_indices

    def cosine_similarity_loss(self, pred_normals, gt_normals):
        # normalize vectors
        pred_normals = F.normalize(pred_normals, p=2, dim=-1)  # Normalize along the last dimension
        gt_normals = F.normalize(gt_normals, p=2, dim=-1)

        # Compute cosine similarity between corresponding normals
        cosine_loss = 1 - (pred_normals * gt_normals).sum(dim=-1).mean()

        return cosine_loss

    def train_epoch(self, dataloader, epoch):
        wandb_dict = {}
        self.decoder.train()
        total_chamfer_loss, total_roi_loss, total_loss_epoch, total_normals_loss, total_roi_normals_loss = 0, 0, 0, 0, 0
        for i, item in enumerate(dataloader):
            #print(i)
            self.optimizer.zero_grad()
            # get data
            target_meshes, template_vertices, template_faces, images, camera_parameters, centers, scales, frames, num_points, names, robot_data = item
            batch_size = len(frames)

            # get features from images
            image_features = self.image_encoder.forward(images)
            reshaped_features = image_features.view(batch_size, 1000*self.num_cameras)

            # get force features
            if self.force_features:
                forces = torch.stack([torch.tensor(data_item['forces']) for data_item in robot_data])
                ee_position = torch.stack([torch.tensor(data_item['ee_pos']) for data_item in robot_data])
                forces_input = torch.cat((forces, ee_position), dim=1).to(self.device).float()
                force_features = self.force_encoder(forces_input)
                reshaped_features = torch.cat((reshaped_features, force_features), dim=1)


            # predict deformation
            coordinates = self.decoder.forward(reshaped_features, template_vertices)
            coordinates = coordinates.reshape(batch_size, 9000, 3)

            # create new source mesh
            predicted_mesh_vertices = [coordinates[s][:num_points[s]] for s in range(batch_size)]
            predicted_meshes = Meshes(verts=predicted_mesh_vertices, faces=template_faces)

            # chamfer loss
            numberOfSampledPoints = 2500
            predicted_sampled, normals_predicted = sample_points_from_meshes(predicted_meshes, numberOfSampledPoints,
                                                                             return_normals=True)
            target_sampled, normals_target = sample_points_from_meshes(target_meshes, numberOfSampledPoints,
                                                                       return_normals=True)
            chamfer_loss_mesh, normals_loss = chamfer_distance(target_sampled, predicted_sampled, x_normals=normals_target, y_normals=normals_predicted)

            # roi loss
            numberOfSampledPoints = 250
            ee_pos = [data_item['ee_pos'] for data_item in robot_data]
            ee_ori = [torch.tensor(np.array(data_item['T_ME'])[:3, :3]) for data_item in robot_data]
            rotation_matrices = torch.stack(ee_ori)
            target_roi_meshes = self.get_roi_meshes(target_meshes, ee_pos, rotation_matrices, 0.4, 80)
            predicted_roi_meshes = self.get_roi_meshes(predicted_meshes, ee_pos, rotation_matrices, 0.25, 30)
            try:
                target_roi_sampled, target_roi_normals = sample_points_from_meshes(target_roi_meshes, numberOfSampledPoints, return_normals=True)
                predicted_roi_sampled, predicted_roi_normals = sample_points_from_meshes(predicted_roi_meshes, numberOfSampledPoints, return_normals=True)
                chamfer_loss_roi, normals_loss_roi = chamfer_distance(target_roi_sampled, predicted_roi_sampled, x_normals=target_roi_normals, y_normals=predicted_roi_normals)

            except ValueError:
                chamfer_loss_roi = torch.tensor(0.05, device=self.device)
                normals_loss_roi = torch.tensor(0.4, device=self.device)



            # # render images from predicted mesh
            # transformed_mesh = self.transform_meshes(predicted_meshes, centers, scales)
            # rendered_images, masks = render_meshes(transformed_mesh, camera_parameters, self.device)
            # rendered_images = rendered_images.detach()
            # masks = masks.detach()
            #
            # # get render loss
            # render_loss = self.perceptual_loss(rendered_images, images, masks)

            # backward pass
            total_loss = chamfer_loss_mesh * self.chamfer_weight + normals_loss*self.normals_weight + chamfer_loss_roi * self.roi_chamfer_weight + normals_loss_roi * self.roi_normals_weight
            total_loss.backward()

            # losses for logging
            total_chamfer_loss += chamfer_loss_mesh
            total_roi_loss += chamfer_loss_roi
            total_roi_normals_loss += normals_loss_roi
            total_normals_loss += normals_loss
            total_loss_epoch += total_loss


            self.optimizer.step()
            self.scheduler.step()

            if epoch % 100 == 0:
                transformed_meshes = self.transform_meshes(predicted_meshes, centers, scales)
                self.save_meshes(transformed_meshes, names, frames, validation=False)

        wandb_dict["chamfer_loss_training"] = total_chamfer_loss / len(dataloader)
        wandb_dict["normals_loss_training"] = total_normals_loss / len(dataloader)
        wandb_dict["roi_loss_training"] = total_roi_loss / len(dataloader)
        wandb_dict["roi_normals_loss_training"] = total_roi_normals_loss / len(dataloader)
        wandb_dict["total_loss_training"] = total_loss_epoch / len(dataloader)

        return wandb_dict

    def validate(self, dataloader, epoch):
        self.decoder.eval()
        wandb_dict = {}
        total_chamfer_loss, total_normals_loss = 0, 0
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                # get data
                target_meshes, template_vertices, template_faces, images, camera_parameters, centers, scales, frames, num_points, names, robot_data = item
                batch_size = len(frames)

                # get features from images
                image_features = self.image_encoder.forward(images)
                reshaped_features = image_features.view(batch_size, 1000 * self.num_cameras)

                # get force features
                if self.force_features:
                    forces = torch.stack([torch.tensor(data_item['forces']) for data_item in robot_data])
                    ee_position = torch.stack([torch.tensor(data_item['ee_pos']) for data_item in robot_data])
                    forces_input = torch.cat((forces, ee_position), dim=1).to(self.device).float()
                    force_features = self.force_encoder(forces_input)
                    reshaped_features = torch.cat((reshaped_features, force_features), dim=1)


                # predict deformation
                coordinates = self.decoder.forward(reshaped_features, template_vertices)
                coordinates = coordinates.reshape(batch_size, 9000, 3)

                # create new source mesh
                predicted_mesh_vertices = [coordinates[s][:num_points[s]] for s in range(batch_size)]
                predicted_meshes = Meshes(verts=predicted_mesh_vertices, faces=template_faces)

                # chamfer loss
                numberOfSampledPoints = 2500
                predicted_sampled, normals_predicted = sample_points_from_meshes(predicted_meshes,
                                                                                 numberOfSampledPoints,
                                                                                 return_normals=True)
                target_sampled, normals_target = sample_points_from_meshes(target_meshes, numberOfSampledPoints,
                                                                           return_normals=True)
                chamfer_loss_mesh, normals_loss = chamfer_distance(target_sampled, predicted_sampled,
                                                                   x_normals=normals_target,
                                                                   y_normals=normals_predicted)

                # roi loss
                numberOfSampledPoints = 250
                ee_pos = [data_item['ee_pos'] for data_item in robot_data]
                ee_ori = [torch.tensor(np.array(data_item['T_ME'])[:3, :3]) for data_item in robot_data]
                rotation_matrices = torch.stack(ee_ori)
                target_roi_meshes = self.get_roi_meshes(target_meshes, ee_pos, rotation_matrices, 0.4, 70)
                predicted_roi_meshes = self.get_roi_meshes(predicted_meshes, ee_pos, rotation_matrices, 0.25, 50)
                try:
                    target_roi_sampled, target_roi_normals = sample_points_from_meshes(target_roi_meshes,
                                                                                       numberOfSampledPoints,
                                                                                       return_normals=True)
                    predicted_roi_sampled, predicted_roi_normals = sample_points_from_meshes(predicted_roi_meshes,
                                                                                             numberOfSampledPoints,
                                                                                             return_normals=True)
                    chamfer_loss_roi, normals_loss_roi = chamfer_distance(target_roi_sampled, predicted_roi_sampled,
                                                                          x_normals=target_roi_normals,
                                                                          y_normals=predicted_roi_normals)

                except ValueError:
                    chamfer_loss_roi = torch.tensor(0.05, device=self.device)
                    normals_loss_roi = torch.tensor(0.4, device=self.device)

                # # render images from predicted mesh
                # transformed_mesh = self.transform_meshes(predicted_meshes, centers, scales)
                # rendered_images, masks = render_meshes(transformed_mesh, camera_parameters, self.device)
                # rendered_images = rendered_images.detach()
                # masks = masks.detach()
                #
                # # get render loss
                # render_loss = self.perceptual_loss(rendered_images, images, masks)

                total_chamfer_loss += chamfer_loss_mesh
                total_normals_loss += normals_loss

                if epoch % 100 == 0:
                    transformed_meshes = self.transform_meshes(predicted_meshes, centers, scales)
                    self.save_meshes(transformed_meshes, names, frames, validation=True)

                if not self.log_meshes:
                    if batch_size > 1:
                        self.log_meshes = [{"name": names[0], "frame": frames[0]}, {"name": names[1], "frame": frames[1]}]
                    else:
                        self.log_meshes = [{"name": names[0], "frame": frames[0]}]



                for k in range(batch_size):
                    for item in self.log_meshes:
                        if names[k] == item["name"] and int(frames[k]) == int(item["frame"]):
                            pred = mesh_plotly(predicted_meshes[k], predicted_roi_meshes[k], ["Predicted", "ROI"],
                                               ee_pos[k])
                            gt = mesh_plotly(target_meshes[k], target_roi_meshes[k], ["Target", "ROI"], ee_pos[k])
                            comparison = mesh_plotly(target_meshes[k], predicted_meshes[k], ["Target", "Predicted"],
                                                     ee_pos[k])
                            wandb_dict[f"predicted_mesh_{item['name']}"] = wandb.Plotly(pred)
                            wandb_dict[f"target_mesh_{item['name']}"] = wandb.Plotly(gt)
                            wandb_dict[f"comparison_{item['name']}"] = wandb.Plotly(comparison)

        wandb_dict["chamfer_loss_validation"] = total_chamfer_loss / len(dataloader)
        wandb_dict["normals_loss_validation"] = total_normals_loss / len(dataloader)

        return wandb_dict

    def transform_meshes(self, meshes, centers, scales):
        transformed_verts = []
        for i, mesh in enumerate(meshes):
            verts = mesh.verts_packed()
            verts = verts * scales[i].to(self.device)
            verts = verts + centers[i].to(self.device)
            verts = verts / 1000
            transformed_verts.append(verts.float())

        transformed_meshes = Meshes(verts=transformed_verts, faces=meshes.faces_list())

        return transformed_meshes

    def save_meshes(self, meshes, names, frames, validation):
        with torch.no_grad():
            for i, mesh in enumerate(meshes):
                if validation:
                    data_path = f"data/out/validation/{names[i]}"
                else:
                    data_path = f"data/out/train/{names[i]}"
                os.makedirs(data_path, exist_ok=True)
                file = os.path.join(data_path, f"mesh-f{frames[i]}.obj")
                save_obj(file, mesh.verts_packed(), mesh.faces_packed())

def training_main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_paths = []
    for set in args.datasets:
        data_paths.append(f"data/{set}")
    epochs = args.epochs
    epoch_start = 1
    batch_size = args.batch_size
    log = args.log

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="DeformingTemplate",

            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "model": "resnet50",
                "chamfer_weight": args.chamfer_weight,
                "normals_weight": args.normals_weight,
                "roi_chamfer_weight": args.roi_chamfer_weight,
                "roi_normals_weight": args.roi_normals_weight,
                "batch_size": batch_size,
                "dataset": args.datasets,
                "cameras": len(args.datasets),
                "epochs": epochs,
                "force_features": args.force_features,
            }
        )

    session = Training(device, args)
    # get data
    train_loader, valid_loader = session.load_data(data_paths, batch_size, args.cameras)
    print("Training dataset size:", len(train_loader), "Validation set size:", len(valid_loader))

    for epoch in range(epoch_start, epochs):
        print("Epoch: ", epoch)
        training_dict = session.train_epoch(train_loader, epoch)
        valid_dict = session.validate(valid_loader, epoch)
        if log:
            total_dict = {**training_dict, **valid_dict}
            wandb.log(total_dict)

