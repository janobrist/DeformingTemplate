import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from nvp_cadex import NVP_v2_5_frame
from torch.utils.data import Dataset
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import trimesh
from dataset_meshes import Dataset_mesh, Dataset_mesh_objects, collate_fn
from torch.utils.data import DataLoader
import random
from foldingNet_model import AutoEncoder
import torch.nn.init as init

import open3d as o3d
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


defomed_model = './nvp_foldingnet_ycb_cosinusAneal/'
B = 4
path_autoencoder='/home/elham/Desktop/FoldingNet/first/logs/model_lowest_cd_loss.pth'

valid_dataset = Dataset_mesh_objects(trg_root='/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_new_5_off/val/', src_root='/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_new_5_off/in')
valid_dataloader = DataLoader(valid_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn)
#print('after')


os.makedirs(defomed_model+ 'check', exist_ok=True)
os.makedirs(defomed_model + 'meshes', exist_ok=True)
os.makedirs(defomed_model + 'meshes_compare_deform_decode', exist_ok=True)

path_load_check_decoder = defomed_model+'check/'+ 'check_min'+'.pt'

#print('path:' ,path)

#print('here')
parser = argparse.ArgumentParser(description="Just an example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--load", action="store_true", help="checksum blocksize")
args = vars(parser.parse_args())
print('args: ', args['load'])

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

print('device: ', device)

writer = SummaryWriter(log_dir=defomed_model+'events/')
###########################################################################

#homeomorphism_decoder:
n_layers = 6
#dimension of the code
feature_dims = 128
hidden_size = [128, 64, 32, 32, 32]
#the dimension of the coordinates to be projected onto
proj_dims = 128
code_proj_hidden_size = [128, 128, 128]
proj_type = 'simple'
block_normalize = True
normalization = False

#print('here1')
c_dim = 128
hidden_dim = 128

homeomorphism_decoder = NVP_v2_5_frame(n_layers=n_layers, feature_dims=feature_dims*8, hidden_size=hidden_size, proj_dims=proj_dims,\
code_proj_hidden_size=code_proj_hidden_size, proj_type=proj_type, block_normalize=block_normalize, normalization=normalization).to('cuda')


network = AutoEncoder()
device = torch.device('cuda')
network.to(device)


check_auto = torch.load(path_autoencoder)
network.load_state_dict(check_auto["model_state_dict"])

print('here2')
#optimizer = optim.Adam(homeomorphism_decoder.parameters(), lr=5e-4, weight_decay=1e-5)
#scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-7)
#optimizer = torch.optim.SGD([homeomorphism_decoder.parameters()], lr=lr, momentum=0.0)

if(args['load']):
    checkpoint = torch.load(path_load_check_decoder)
    #homeomorphism_encoder.load_state_dict(checkpoint['encoder'])
    homeomorphism_decoder.load_state_dict(checkpoint['decoder'])
    #optimizer.load_state_dict(checkpoint['optimizer'])


Nepochs = 500000

w_chamfer = 1.0
#print('here3')
w_edge = 0 #1.0

w_normal = 1.0 #0.01

w_laplacian = 0 #1.0 #0.1

plot_period = 250

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


iteration = 0


losses = []
#print('epoch:', epoch)
homeomorphism_decoder.eval()
for i, item in enumerate(valid_dataloader):


    num_points = item['num_points']
    num_faces = item['num_faces']
    B,_,_ = item['vertices_src'].shape
    trg_mesh_verts_rightSize = [item['vertices_trg'][s][:num_points[s]]for s in range(B)]
    trg_mesh_faces_rightSize = [item['faces_trg'][s][:num_faces[s]]for s in range(B)]

    src_mesh_verts_rightSize = [item['vertices_src'][s][:num_points[s]]for s in range(B)]
    src_mesh_faces_rightSize = [item['faces_src'][s][:num_faces[s]]for s in range(B)]

    trg_mesh = Meshes(verts=trg_mesh_verts_rightSize, faces=trg_mesh_faces_rightSize)
    src_mesh = Meshes(verts=src_mesh_verts_rightSize, faces=src_mesh_faces_rightSize)
    
    seq_pc_trg = sample_points_from_meshes(trg_mesh, 4000).to('cuda')
    seq_pc_src = sample_points_from_meshes(src_mesh, 4000).to('cuda')

    with torch.no_grad():
        #print('seq_pc_trg shape: ', seq_pc_trg.shape)
        
        code_trg = network.encoder(seq_pc_trg.permute(0, 2, 1))

    
    #print('code_trg.shape: ', code_trg.shape)
    b, k = code_trg.shape

    query = item['vertices_src'].to('cuda')

    #print('code trg shape: ', code_trg.shape)
    #print('query shape: ', query.shape)
    with torch.no_grad():
        coordinates = homeomorphism_decoder.forward(code_trg, query)
    #print('coordinates shape: ', coordinates.shape)
    coordinates = coordinates.reshape(B, 9000, 3)

    new_src_mesh_verts_rightSize = [coordinates[s][:num_points[s]]for s in range(B)]
    new_src_mesh_faces_rightSize = [item['faces_src'][s][:num_faces[s]].to('cuda') for s in range(B)]

    new_src_mesh = Meshes(verts=new_src_mesh_verts_rightSize, faces=new_src_mesh_faces_rightSize)

    sample_trg = sample_points_from_meshes(trg_mesh, 5000).to('cuda')
    new_sample_src = sample_points_from_meshes(new_src_mesh, 5000).to('cuda')


    loss_chamfer, _ = chamfer_distance(sample_trg, new_sample_src)
    loss = loss_chamfer * w_chamfer
    print('loss: ',loss)
    losses.append(loss)


    #loss.backward()

    #scheduler.step()


    #final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    
    for i_item in range(B):
        name = item['name'][i_item]
        scale_trg = item['scale_obj'][i_item]
        center_trg = item['center_obj'][i_item]
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(i_item)
        final_verts = final_verts * scale_trg + center_trg
        final_obj = os.path.join(defomed_model+'meshes/', 'mesh_'+name.split('.')[0]+'_'+'.obj')
        save_obj(final_obj, final_verts, final_faces)


loss_mean = sum(losses) / len(losses)
