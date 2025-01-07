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

from utils.datahandler import HDF5Dataset  # Data loader
from backbone.flimngo_net import GetModel  # YOLO model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

# Set random seed
seed_value = 0
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a YOLO-based model for FLIM lifetime prediction')
    parser.add_argument('--wandb_load', '-wl', type=bool, default=True, help='Enable/disable WandB logging')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='Batch size for training')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--lr', '-l', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--width_multiple', '-wm', type=float, default=0.5, help='YOLO width multiple')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--dataset_path', type=str, default='../train_data/', help='Path to the training dataset')
    parser.add_argument('-f')  # Ignore unnecessary Jupyter arguments
    return parser.parse_args()

def prepare_data(dataset_path, batch_size):
    """Prepare datasets and data loaders."""
    logging.info('Loading dataset...')
    try:
        dataset = HDF5Dataset(imgs_dir=os.path.join(dataset_path))
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

    # Split dataset
    train_size = round(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0)
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    logging.info(f'Dataset loaded: {len(train_set)} training samples, {len(val_set)} validation samples')
    return train_loader, val_loader

def init_model(args, train_set):
    """Initialize the YOLO model."""
    class Options:
        def __init__(self):
            self.model = 'flimngo'
            self.cpu = not torch.cuda.is_available()
            self.imageSize = train_set[0][0].shape[-1] # spatial (x,y) dimentions
            self.n_in_channels = 256 # number of time channels, only works for 256
            self.time_resolution = 0.09765625 # time resolution (in ns)
            self.width_multiple = args.width_multiple

    opt = Options()
    model = GetModel(opt).to(device)
    logging.info('Model initialized.')
    return model

def train_model(net, args, train_loader, val_loader, model_id):
    """Train the model."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_valid_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        net.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            for image, ground_truth in train_loader:
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
            for image, ground_truth in val_loader:
                image, ground_truth = image.to(device), ground_truth.to(device)
                target = net(image)
                loss = criterion(target, ground_truth)
                valid_loss += loss.item()
        valid_loss /= len(val_loader)

        logging.info(f'Epoch {epoch + 1} - Training Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')

        if args.wandb_load:
            wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss})

        if valid_loss < min_valid_loss:
            epochs_without_improvement = 0
            min_valid_loss = valid_loss
            save_path = os.path.join('checkpoints', model_id)
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(net.state_dict(), save_path)
            logging.info(f'Saving model at {save_path}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping:
                logging.info('Early stopping triggered.')
                break

def main():
    args = parse_args()
    train_loader, val_loader = prepare_data(args.dataset_path, args.batch_size)
    model_id = f"{datetime.now():%b-%d-%Y_%H-%M-%S}_irf_lr{args.lr}.pth"
    model = init_model(args, train_loader.dataset)
    
    if args.wandb_load:
        wandb.init(
            project='FLIMngo',
            config=vars(args)
        )

    train_model(model, args, train_loader, val_loader, model_id)

if __name__ == '__main__':
    main()
