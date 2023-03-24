import argparse
import os
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch
import torch.optim as optim
import numpy as np
from foldingNet_model import AutoEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Autoencoder
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from torch.utils.tensorboard import SummaryWriter
from dataset import PointClouds

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--k', type=int, default=256)

parser.add_argument('--encoder_type', type=str, default='folding')
parser.add_argument('--load', action='store_true')
parser.add_argument("-e", "--euler", action="store_true", help="checksum blocksize")
args = parser.parse_args()



# prepare training and testing dataset
numOfPoints = 3000


# device
if(args.euler):
    rootData="/hdd/eli"
    train_deformed=rootData+'/ycb_mult_5_one_seq/train/'
    train_src=rootData+'/ycb_mult_5_one_seq/in'
    valid_deformed=rootData+'/ycb_mult_5_one_seq/val/'
    path_autoencoder='/hdd/eli/first_50_each_folding_'+str(numOfPoints)+'_'+str(args.k)+'dim/'
    if torch.cuda.is_available():
        device = torch.device("cuda:2")
else:
    rootData="/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path"
    train_deformed=rootData+'/ycb_mult_5_one_seq/train/'
    train_src=rootData+'/ycb_mult_5_one_seq/in'
    valid_deformed=rootData+'/ycb_mult_5_one_seq/val/'
    path_autoencoder='./first_50_each_folding_'+str(numOfPoints)+'_'+str(args.k)+'dim/'
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

train_dataset = PointClouds(train_deformed, is_training=True, num_points=numOfPoints)
test_dataset = PointClouds(valid_deformed, is_training=True, num_points=numOfPoints)
#train_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='train', classification=False, data_augmentation=True)
#test_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='test', classification=False, data_augmentation=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


#str(args.encoder_type)
os.makedirs(path_autoencoder+'events', exist_ok=True)
os.makedirs(path_autoencoder+'logs', exist_ok=True)
args.log_dir =path_autoencoder+'logs'
args.events = path_autoencoder+'events'


# model
if(args.encoder_type == '2018'):
    def weights_init(m):
        if isinstance(m, nn.Conv1d):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    autoencoder = Autoencoder(k=args.k, num_points=numOfPoints).to(device)
    autoencoder = autoencoder.apply(weights_init).to(device)
elif(args.encoder_type == 'folding'):
    autoencoder = AutoEncoder(k=args.k)

autoencoder.to(device)

# loss function
# optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)
#num_steps = args.epochs * (len(train_dataset) // args.batch_size)
#optimizer = optim.Adam(autoencoder.parameters(), lr=5e-4, weight_decay=1e-5)
#scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-7)
#batches = int(len(train_dataset) / args.batch_size + 0.5)

best_epoch = -1


writer = SummaryWriter(log_dir=args.events)

print('\033[31mBegin Training...\033[0m')
if(not args.load):
    epoch_start = 0
    min_cd_loss = 1000000000
    iter = 0
else:
    dict = torch.load(args.log_dir+'/model_epoch_9000.pth')
    epoch_start = dict['epoch']
    min_cd_loss = dict['loss']
    iter = dict['epoch'] * len(test_dataset)
    autoencoder.load_state_dict(dict['model_state_dict'])
    optimizer.load_state_dict(dict['optimizer_state_dict'])

for epoch in range(epoch_start, args.epochs + 1):
    # training
    start = time.time()
    autoencoder.train()
    losses = []

    for data, path, mean, scale in train_dataloader:
        point_clouds = data
        point_clouds = point_clouds.to(device)
        _, recons = autoencoder(point_clouds)
        print('recons shape: ', recons.shape)
        ls = chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))[0]
        print('train loss: ', ls)
        losses.append(ls.detach().cpu())
        #ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        #scheduler.step()
        
        #q+=1
        #print('q: ', q)
        writer.add_scalar("loss_iteration/train",  ls.detach().cpu(), iter)
        iter+=1


    for param_group in optimizer.param_groups:
            #print(param_group['lr'])
            lr = param_group['lr']

    writer.add_scalar("loss_mean/train",  np.mean(losses), epoch)
    writer.add_scalar("lr/train", lr, epoch)


    # evaluation
    autoencoder.eval()
    total_cd_loss = 0
    losses = []
    with torch.no_grad():
        for data, path, mean, scale in test_dataloader:
            point_clouds = data
            point_clouds = point_clouds.to(device)
            _, recons = autoencoder(point_clouds)
            ls = chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            print('valid loss: ', ls)
            losses.append(ls[0].detach().cpu())
    
    # calculate the mean cd loss
    mean_cd_loss = np.mean(losses)
    writer.add_scalar("loss_mean/valid",  mean_cd_loss,  epoch)

    # records the best model and epoch
    if mean_cd_loss < min_cd_loss:
        min_cd_loss = mean_cd_loss
        best_epoch = epoch
        #torch.save(autoencoder.state_dict(), os.path.join(args.log_dir, 'model_lowest_cd_loss.pth'))
        torch.save({'lr':lr,'epoch': epoch,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': min_cd_loss}, os.path.join(args.log_dir, 'model_lowest_cd_loss.pth'))
    
    # save the model every 100 epochs
    if (epoch) % 100 == 0:
        torch.save({'lr':lr,'epoch': epoch, 
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': min_cd_loss}, os.path.join(args.log_dir, 'model_epoch_{}.pth'.format(epoch)))
        #torch.save(autoencoder.state_dict(), os.path.join(args.log_dir, 'model_epoch_{}.pth'.format(epoch)))
    
    end = time.time()
    cost = end - start

    #print('\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m'.format(
    #    epoch, args.epochs, mean_cd_loss, min_cd_loss, best_epoch))
    #print('\033[31mCost {} minutes and {} seconds\033[0m'.format(int(cost // 60), int(cost % 60)))
