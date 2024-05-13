import os
import torch
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.nvp_cadex import NVP_v2_5_frame
from pytorch3d.loss import (
    chamfer_distance,
)
from datasets.dataset_meshes import Dataset_mesh_objects, collate_fn_nofor
from torch.utils.data import DataLoader
from foldingNet_model import AutoEncoder

from modelAya import Autoencoder
#from model import Autoencoder

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser(description="Just an example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--load", action="store_true", help="checksum blocksize")
parser.add_argument("-e", "--euler", action="store_true", help="checksum blocksize")
parser.add_argument('--encoder_type', type=str, default='2018')
args = vars(parser.parse_args())
config='4'

if(config=="4"):
    auto="auto2018_1024dim_3000points_NoAug_1000seq_5ycb"
    trainName='/ycb_mult_5_thousand_seq/train/'
    valName='/ycb_mult_5_thousand_seq/val/'
    inName='/ycb_mult_5_thousand_seq/in'
    deformName='nvp_2018_1024dim_ycb_1000seq_5ycb_cosinusAneal_20/'

if(args['euler']):
    defomed_model = '/hdd/eli/'+deformName
    rootData="/hdd/eli/data/ycb"
    train_deformed=rootData+trainName
    train_src=rootData+inName
    valid_deformed=rootData+valName
    valid_src=rootData+inName
    #path_autoencoder='./first_50_each_2018_1024dim/logs/model_lowest_cd_loss.pth'
    path_autoencoder='/hdd/eli/'+auto+'/models/check_min.pt'
    if torch.cuda.is_available():
        device = torch.device("cuda:3")
else:
    defomed_model = '/home/elham/srl-nas/elham/research_project/logs/'+deformName
    rootData="/home/elham/hdd/data/ycb"#"/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path"
    train_deformed=rootData+trainName
    train_src=rootData+inName
    valid_deformed=rootData+valName
    valid_src=rootData+inName
    #path_autoencoder='/home/elham/Desktop/deformTemplate/first_50_each_2018_1024dim/logs/model_lowest_cd_loss.pth'
    path_autoencoder='/home/elham/srl-nas/elham/research_project/logs/'+auto+'/models/check_min.pt'
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

B = 8
#print('before')
training_dataset = Dataset_mesh_objects(trg_root=train_deformed, src_root=train_src, lastConfig=True)
train_dataloader = DataLoader(training_dataset, batch_size=B, shuffle=True, collate_fn=lambda b, device=device: collate_fn_nofor(b, device), drop_last=True)

valid_dataset = Dataset_mesh_objects(trg_root=valid_deformed, src_root=valid_src, lastConfig=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=B, shuffle=True, collate_fn=lambda b, device=device: collate_fn_nofor(b, device), drop_last=True)
#print('after')


os.makedirs(defomed_model+ 'check', exist_ok=True)
os.makedirs(defomed_model + 'meshes', exist_ok=True)
os.makedirs(defomed_model + 'meshes_compare_deform_decode', exist_ok=True)

path_load_check_decoder = defomed_model+'check/'+ 'check'+str(3)+'.pt'

#print('path:' ,path)

#print('here')

print('args: ', args['load'])

# Set the device


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
code_proj_hidden_size=code_proj_hidden_size, proj_type=proj_type, block_normalize=block_normalize, normalization=normalization).to(device)
numOfPoints=3000

if(args['encoder_type'] == '2018'):
    network = Autoencoder(k=1024, num_points=numOfPoints).to(device)
elif(args['encoder_type'] == 'folding'):
    network = AutoEncoder()

network.to(device)


check_auto = torch.load(path_autoencoder, map_location='cuda:0')
if(args['encoder_type'] == '2018'):
    #print('here')
    network.load_state_dict(check_auto["model"])
else:
    network.load_state_dict(check_auto["model_state_dict"])

#print('here2')
#optimizer = optim.Adam(homeomorphism_decoder.parameters(), lr=5e-4, weight_decay=1e-5)
optimizer = optim.Adam(homeomorphism_decoder.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-7)

if(args['load']):
    checkpoint = torch.load(path_load_check_decoder, map_location='cuda:0')
    #homeomorphism_encoder.load_state_dict(checkpoint['encoder'])
    homeomorphism_decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    minLoss = checkpoint['val_min_loss']
    epoch_start = checkpoint['epoch']
else:
    minLoss = 1000000
    epoch_start = 0

Nepochs = 500000

w_chamfer = 1.0
print('here3')
w_edge = 0 #1.0

w_normal = 1.0 #0.01

w_laplacian = 0 #1.0 #0.1

plot_period = 250

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


iteration = 0
import time
for epoch in range(epoch_start, Nepochs):  # loop over the dataset multiple times
    losses = []
    #print('epoch:', epoch)
    homeomorphism_decoder.train()
    network.train()
    for i, item in enumerate(train_dataloader):

        #print('i: ',i)
        start = time.time()
        orig_verts_trg, orig_faces_trg, orig_verts_src, orig_faces_src, changedItem=item

        print('item reading: ', time.time()- start)
        optimizer.zero_grad()


        num_points = changedItem['num_points']
        #num_faces = changedItem['num_faces']
        B,_,_ = changedItem['vertices_src'].shape

        trg_mesh_verts_rightSize = orig_verts_trg #[item['vertices_trg'][s][:num_points[s]]for s in range(B)]

        trg_mesh_faces_rightSize = orig_faces_trg #[item['faces_trg'][s][:num_faces[s]]for s in range(B)]


        src_mesh_verts_rightSize = orig_verts_src #[item['vertices_src'][s][:num_points[s]]for s in range(B)]

        src_mesh_faces_rightSize = orig_faces_src #[item['faces_src'][s][:num_faces[s]]for s in range(B)]


        trg_mesh = Meshes(verts=trg_mesh_verts_rightSize, faces=trg_mesh_faces_rightSize)

        src_mesh = Meshes(verts=src_mesh_verts_rightSize, faces=src_mesh_faces_rightSize)

        #print('creating meshes: ', time.time()- start)
        
        seq_pc_trg = sample_points_from_meshes(trg_mesh, numOfPoints).to(device)

        #print('sample points: ', time.time()- start)
        #seq_pc_src = sample_points_from_meshes(src_mesh, 4000).to(device)

        #with torch.no_grad():
            #print('seq_pc_trg shape: ', seq_pc_trg.shape)
        if(args['encoder_type'] == '2018'):
            code_trg , _ = network(seq_pc_trg.permute(0, 2, 1))
        else:
            code_trg = network.encoder(seq_pc_trg.permute(0, 2, 1))

        #print('encode: ', time.time()- start)
        #print('code_trg.shape: ', code_trg.shape)
        b, k = code_trg.shape

        query = changedItem['vertices_src'].to(device)

        #print('code trg shape: ', code_trg.shape)
        #print('query shape: ', query.shape)
        beforeDecoder = time.time()
        coordinates = homeomorphism_decoder.forward(code_trg, query)
        #print('decode: ', time.time()- start )
        #print('coordinates shape: ', coordinates.shape)

        coordinates = coordinates.reshape(B, 9000, 3)

        loopstart= time.time()
        new_src_mesh_verts_rightSize = [coordinates[s][:num_points[s]]for s in range(B)]
        loopend = time.time()
        #print('loop time: ', loopend - loopstart)
        new_src_mesh_faces_rightSize = src_mesh_faces_rightSize#.to(device) #[changedItem['faces_src'][s][:num_faces[s]].to(device) for s in range(B)]

        new_src_mesh = Meshes(verts=new_src_mesh_verts_rightSize, faces=new_src_mesh_faces_rightSize)
        #print('new_src_mesh device: ', new_src_mesh.device)

        numberOfSampledPoints = 5000
        sample_trg = sample_points_from_meshes(trg_mesh, numberOfSampledPoints).to(device)
        new_sample_src = sample_points_from_meshes(new_src_mesh, numberOfSampledPoints).to(device)


        loss_chamfer, _ = chamfer_distance(sample_trg, new_sample_src)
        loss = loss_chamfer * w_chamfer
        print('loss: ',loss)
        losses.append(loss)

        
        backstart = time.time()
        loss.backward()
        optimizer.step()
        scheduler.step()
        backend = time.time()
        #print('back time: ', backend - backstart)

        end = time.time()
        #print('how long did it take: ', end-start)


        #final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        
        #q = 0
        for param_group in optimizer.param_groups:
            #print(param_group['lr'])
            lr = param_group['lr']
            #q+=1
            #print('q: ', q)
        #break
        writer.add_scalar("train/loss_iteration", loss, iteration)
        iteration+=1

        #break
        
    for i_item in range(B):
        name = changedItem['name'][i_item]
        scale_trg = changedItem['scale_obj'][i_item].to(device)
        center_trg = changedItem['center_obj'][i_item].to(device)
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(i_item)
        final_verts = final_verts * scale_trg + center_trg
        final_obj = os.path.join(defomed_model+'meshes/', 'mesh_'+name.split('.')[0]+'_'+'.obj')
        save_obj(final_obj, final_verts, final_faces)


    loss_mean = sum(losses) / len(losses)
    #print('loss_mean: ', loss_mean)
    
    

    writer.add_scalar("train/lr", lr, epoch)
    writer.add_scalar("train/loss", loss_mean, epoch)

    losses=[]
    homeomorphism_decoder.eval()
    network.eval()
    for i, item in enumerate(valid_dataloader):

        orig_verts_trg, orig_faces_trg, orig_verts_src, orig_faces_src, changedItem=item
        num_points = changedItem['num_points']
        #num_faces = changedItem['num_faces']
        B,_,_ = changedItem['vertices_src'].shape
        trg_mesh_verts_rightSize = orig_verts_trg #[item['vertices_trg'][s][:num_points[s]]for s in range(B)]
        trg_mesh_faces_rightSize = orig_faces_trg #[item['faces_trg'][s][:num_faces[s]]for s in range(B)]

        src_mesh_verts_rightSize = orig_verts_src #[item['vertices_src'][s][:num_points[s]]for s in range(B)]
        src_mesh_faces_rightSize = orig_faces_src #[item['faces_src'][s][:num_faces[s]]for s in range(B)]

        trg_mesh = Meshes(verts=trg_mesh_verts_rightSize, faces=trg_mesh_faces_rightSize)
        src_mesh = Meshes(verts=src_mesh_verts_rightSize, faces=src_mesh_faces_rightSize)
        
        seq_pc_trg = sample_points_from_meshes(trg_mesh, numOfPoints).to(device)#3000


        with torch.no_grad():
            #code_trg = network.encoder(seq_pc_trg.permute(0, 2, 1))
            if(args['encoder_type'] == '2018'):
                code_trg , _ = network(seq_pc_trg.permute(0, 2, 1))
            else:
                code_trg = network.encoder(seq_pc_trg.permute(0, 2, 1))

            #pcd = o3d.geometry.PointCloud()

        
        #print('code_trg.shape: ', code_trg.shape)
        b, k = code_trg.shape

        #print('N.shape: ', N)
        query = changedItem['vertices_src'].to(device)#torch.cat(B*[src_mesh.verts_packed().unsqueeze(0)], axis=0)

        with torch.no_grad():
            coordinates = homeomorphism_decoder.forward(code_trg, query)
        coordinates = coordinates.reshape(B, 9000, 3)

        new_src_mesh_verts_rightSize = [coordinates[s][:num_points[s]]for s in range(B)]
        #new_src_mesh_faces_rightSize = [item['faces_src'][s][:num_faces[s]].to(device) for s in range(B)]

        new_src_mesh = Meshes(verts=new_src_mesh_verts_rightSize, faces=src_mesh_faces_rightSize)

        numberOfSampledPoints = 5000
        sample_trg = sample_points_from_meshes(trg_mesh, numberOfSampledPoints).to(device)#5000
        new_sample_src = sample_points_from_meshes(new_src_mesh, numberOfSampledPoints).to(device)#5000

        loss_chamfer, _ = chamfer_distance(sample_trg, new_sample_src)
        #loss_chamfer, _ = chamfer_distance(decode_trg, sample_trg_3000)
        loss = loss_chamfer * w_chamfer
        print('val loss: ', loss)
        losses.append(loss)




        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    loss_mean = sum(losses) / len(losses)
    print('val loss mean: ', loss_mean)
    path_check = defomed_model+'check/'+ 'check'+str(epoch)+'.pt'
    path_check_min = defomed_model+'check/'+ 'check'+'_min'+'.pt'

    #print('loss_mean: ', loss_mean)

    

    writer.add_scalar("valid/lr", lr, epoch)
    writer.add_scalar("valid/loss", loss_mean, epoch)

        
    if(loss_mean < minLoss):
        minLoss=loss_mean
        torch.save({
            #'encoder': homeomorphism_encoder.state_dict(),
            'network': network.state_dict(),
            'decoder': homeomorphism_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch':epoch,
            }, path_check_min)
    #print('--------------------------------------------')

    path_check = defomed_model+'check/'+ 'check'+str(epoch)+'.pt'
    path_check_min = defomed_model+'check/'+ 'check'+'_min'+'.pt'
    if(epoch % 20 == 0):
        torch.save({
            #'encoder': homeomorphism_encoder.state_dict(),
            'network': network.state_dict(),
            'decoder': homeomorphism_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch':epoch,
            'minLoss':minLoss,
            'lr':lr
            }, path_check)

writer.flush()
writer.close()