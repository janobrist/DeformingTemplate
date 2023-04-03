import torch
import json
from torch.utils.data import DataLoader
from input_pipeline import PointClouds
from trainer import Trainer
import trimesh as trimesh
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser()

NUM_EPOCHS = 1000000
BATCH_SIZE = 16


TRAIN_PATH = ''
VAL_PATH = ''
labels = ['02747177']

parser.add_argument("-e", "--euler", action="store_true", help="checksum blocksize")
args = parser.parse_args()

if(args.euler):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:3")
    dataset_path_train='/hdd/eli/ycb_mult_5_one_seq/train_sc'
    #dataset_path_val='/hdd/eli/ycb_mult_1_thousand_seq/val' 
    folder='/hdd/eli/auto2018_1024dim_3000points_NoAug_1seq_scissor'
else:
    dataset_path_train='/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_mult_5_one_seq/train_sc'
    #dataset_path_val='/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_mult_1_thousand_seq/val'
    folder='../auto2018_1024dim_3000points_NoAug_1seq_scissor' 
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")


#dataset_path = '/home/elham/Desktop/point-cloud-autoencoder/donut/' 
#dataset_path = '/home/elham/Desktop/point-cloud-autoencoder/car/'
#dataset_path = '/home/elham/Desktop/point-cloud-autoencoder/car-donut/'
dataset_path_train='/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_mult_5_one_seq/train_sc'
#dataset_path_val='/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_mult_5_one_seq/val_sc'   
#dataset_path = '/home/elham/Desktop/point-cloud-autoencoder/latent_3d_points/data/shape_net_core_uniform_samples_2048/'
os.makedirs(folder, exist_ok=True)
os.makedirs(folder+'/models', exist_ok=True)
os.makedirs(folder+'/plies', exist_ok=True)
os.makedirs(folder+'/events', exist_ok=True)
PATHCheck = folder+'/models/check'
PATHCheckMin = folder+'/models/check_min'
writer = SummaryWriter(log_dir=folder+'/events/')
def train_and_evaluate():
    import trimesh
    train = PointClouds(dataset_path_train, labels, is_training=True)
    #val = PointClouds(dataset_path, labels, is_training=False)

    train_loader = DataLoader(
        dataset=train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4)
    # for x in train_loader:
    #     print(x.shape)
    #     pt = x[0,...].permute(1, 0)
    #     print('pt.shape: ', pt.shape)
    #     color = np.ones_like(pt)
    #     cloud = trimesh.PointCloud(vertices=pt, colors=color)
    #     cloud.show(background=[0,0,0,0])
    # val_loader = DataLoader(
    #     dataset=val, batch_size=1, shuffle=False,
    #     num_workers=1, pin_memory=True
    # )

    num_steps = NUM_EPOCHS * (len(train) // BATCH_SIZE)
    print('num_steps: ', num_steps)
    model = Trainer(num_steps, DEVICE, folder)
    # model.network.to(DEVICE)

    i = 0
    logs = []
    text = 'e: {0}, i: {1}, loss: {2:.3f}'

    #path_encoder='./models_pointnetEncoder2018paper_3000points_256Dim_noAug/check18500.pt'
    #check_auto = torch.load(path_encoder)
    #print('check_auto keys: ', check_auto['model'].keys())
    #model.network.homeomorphism_encoder.load_state_dict(check_auto['encoder'])
    #model.network.load_state_dict(check_auto['model'])
    #model.network.decoder.load_state_dict(check_auto['decoder'])
    #model.optimizer.load_state_dict(check_auto['optimizer'])
    #print('model network: ', model.network)
    e_start=0
    #e_start=check_auto['epoch']
    

    minLoss = 10000000
    for e in range(e_start, NUM_EPOCHS):
        eval_losses = []
        model.network.eval()
        for batch, p, mean, scale in train_loader:

            x = batch.to(DEVICE)
            loss = model.evaluate(x, e, p, mean, scale)
            eval_losses.append(loss)
            print('eval_losses: ', eval_losses)
        losses=[]
        print('e: ', e)
        model.network.train()
        #model.network.train()
        for x, p, mean, scale in train_loader:
            import trimesh
            #print('p: ', p)
            #print('x: ', x)
            #print('x shape: ', x.shape)
            #pt = x[5,...].cpu().permute(1, 0)
            #print('pt.shape: ', pt.shape)
            #color = np.ones_like(pt)
            # cloud = trimesh.PointCloud(vertices=pt, colors=color)
            # cloud.show(background=[0,0,0,0])

            x = x.to(DEVICE)
            loss = model.train_step(x)

            i += 1
            log = text.format(e, i, loss)
            
            #writer.add_scalar("Loss/train", loss_mean, epoch)
            print(log)
            logs.append(loss)
            losses.append(loss)

        for param_group in model.optimizer.param_groups:
            #print(param_group['lr'])
            lr = param_group['lr']
            #q+=1
            #print('q: ', q)
        writer.add_scalar("loss/train",  np.mean(losses), e)
        writer.add_scalar("lr/train", lr, e)
        print('losss: ', np.mean(losses))
        
        
        if(e% 100 == 0):
            torch.save({
            #'encoder': homeomorphism_encoder.state_dict(),
            'model': model.network.state_dict(),
            #'encoder': model.network.homeomorphism_encoder.state_dict(),
            #'decoder': model.network.decoder.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'epoch':e,
            }, PATHCheck+str(e)+'.pt')
        #print(model.network.state_dict())
        if(minLoss > np.mean(losses)):
            minLoss = np.mean(losses)
            torch.save({
            #'encoder': homeomorphism_encoder.state_dict(),
            'model': model.network.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'epoch':e,
            }, PATHCheckMin+'.pt')
        #with open(TRAIN_LOGS, 'w') as f:
        #    json.dump(logs, f)


        # eval_losses = {k: sum(d[k] for d in eval_losses)/len(eval_losses) for k in losses.keys()}
        # eval_losses.update({'type': 'eval'})
        # print(eval_losses)
        # logs.append(eval_losses)

        #model.save(PATH)
  

train_and_evaluate()
