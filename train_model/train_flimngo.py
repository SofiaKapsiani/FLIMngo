import os
import numpy as np
import math
import logging
import argparse
import random
from tqdm import tqdm
from datetime import datetime
import wandb

import torch
import torch.nn as nn
from torch import optim

# laod dataloader for handling the .h5 files
from utils_yolo.datahandler import HDF5Dataset
# load yolo backbone
from yolo_backbone.flimngo_net import GetModel

# load cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# set random seed
seed_value = 0
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)


dataset_path = '../../data/simulated/train/3d_cnn_2828/tg_256_100_2500' # path were train data are stored

def get_args():
    parser = argparse.ArgumentParser(description='Train a YOLO-based model for FLIM lifetime prediction')
    parser.add_argument('--wandb_load', '-wl', metavar='WL', type=bool, default=True, help='Number of epochs')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=10, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=0.01, help='Learning rate',)
    parser.add_argument('--width_multiple', '-wm', metavar='wm', type=float, default=0.5, help='width multiple for yolo',)
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay for optimizer')
    parser.add_argument('--early_stopping', type=int, default=20, help='stop model training if val loss doesnt decrease for a number of epochs')
    parser.add_argument('-f')

    return parser.parse_args()

# get args
args = get_args()

# get train data
train_dataset = HDF5Dataset(imgs_dir=os.path.join(dataset_path))
# split into 80% train and 20% validation data
gen = torch.Generator()
gen.manual_seed(0) # set seed
train_set, val_set = torch.utils.data.random_split(train_dataset, [round(len(train_dataset)*0.8), (len(train_dataset)-round(len(train_dataset)*0.8))],
                                                  generator=gen)

# load into dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

# set model name
now = datetime.now()
model_id = f'{now.strftime("%b-%d-%Y_%H-%M-%S")}_irf_lr{args.lr}.pth'

# initiate wandb if args.wandb_load has been set to True
if args.wandb_load == True:
    wandb.init(
    project='FLIMngo',
    config={
        'epochs': args.epochs, 
        'batch_size': args.batch_size,
        'wokers': args.workers,
        'lr': args.lr,
        'width_multiple': args.width_multiple,
        'weight_decay': args.weight_decay,
        'early_stopping': args.early_stopping,
        })

def train_net(net, args, train_loader, val_loader, model_id, device):
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_valid_loss = math.inf
    epochs_without_improvement = 0  # Counter for epochs without improvement

    for epoch in range(args.epochs):
        net.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            for batch in train_loader:
                image, ground_truth = batch
                image, ground_truth = image.to(device), ground_truth.to(device)

                optimizer.zero_grad()
                target = net(image)
                loss = criterion(target, ground_truth)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)
            
        train_loss /= len(train_loader)
        valid_loss = 0.0
        net.eval()
        with torch.no_grad():
            for batch in val_loader:
                image, ground_truth = batch
                image, ground_truth = image.to(device), ground_truth.to(device)

                target = net(image)
                loss = criterion(target, ground_truth)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)  # Average validation loss
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss:.6f} \t\t Validation Loss: {valid_loss:.6f}')
        
        # save train and val loss on wandb if wandb_load has been set to true
        if args.wandb_load: 
            wandb.log({'train_loss': train_loss,  'valid_loss': valid_loss})

        if min_valid_loss > valid_loss:
            epochs_without_improvement = 0
            print(f'Validation Loss Decreased ({min_valid_loss:.6f} --> {valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss

            save_model_path ='checkpoints/'
            if not os.path.exists(save_model_path):
                # Create the new directory
                os.mkdir(save_model_path)
            # Saving State Dict
            torch.save(net.state_dict(), os.path.join(save_model_path, model_id))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping:
                print("Early stopping triggered. Training terminated.")
                break

if __name__ == '__main__':
    # load yolo model
    class Options:
        def __init__(self):
            self.model = 'flimngo'  # specify the model you want to load
            self.cpu = [False if torch.cuda.is_available() else True][0]  # set to True if you want to use CPU instead of GPU
            self.imageSize = train_set[0][0].shape[-1]  # set the desired (x, y) image dimentions 
            self.n_in_channels = 256  # number of time bins 
            self.width_multiple = 0.5 

    # Create an instance of the options
    opt = Options()
    model = GetModel(opt).to(device=device)
    

    
    train_net(model, args, train_loader, val_loader, model_id, device)
