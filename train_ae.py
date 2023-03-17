import argparse
import os
import time

import torch
import torch.optim as optim
import numpy as np
from foldingNet_model import AutoEncoder
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from torch.utils.tensorboard import SummaryWriter
from dataset import PointClouds

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/home/rico/Workspace/Dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0')
parser.add_argument('--npoints', type=int, default=3048)
parser.add_argument('--mpoints', type=int, default=3048)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--name_conf', type=str, default='./first_50_each/')
parser.add_argument('--load', action='store_true')
args = parser.parse_args()


os.makedirs(args.name_conf+'events', exist_ok=True)
os.makedirs(args.name_conf+'logs', exist_ok=True)
args.log_dir =args.name_conf+'logs'
args.events = args.name_conf+'events'
# prepare training and testing dataset
train_dataset = PointClouds('/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_mult_5_one_seq/train', is_training=True)
test_dataset = PointClouds('/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_mult_5_one_seq/val', is_training=True)
#train_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='train', classification=False, data_augmentation=True)
#test_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='test', classification=False, data_augmentation=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
autoencoder = AutoEncoder()
autoencoder.to(device)

# loss function
# optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)

batches = int(len(train_dataset) / args.batch_size + 0.5)

min_cd_loss = 1e3
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
        recons = autoencoder(point_clouds)
        ls = chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))[0]
        print('train loss: ', ls)
        losses.append(ls.detach().cpu())
        #ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        
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
            recons = autoencoder(point_clouds)
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
