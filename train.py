from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import make_dataset, make_dataloader
from config import cfg
from models import NeuroPose, TransformerModel, make_model
import argparse
import os
from tqdm import tqdm
#setup logger
import logging

from util import create_logger

def weights_init(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0.1)
    

class AverageMeter(object):

    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, value: Any, n: int = 1) -> None:
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
    
    def reset(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def __str__(self) -> str:
        return f"{self.avg}"

def train_epoch(cfg, epoch, model, train_loader, criterion, optimizer, logger=None, device='cpu'):

    model.train()

    avg_loss = AverageMeter()
    smoothness_loss_meter = AverageMeter()

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        
        data, target  = data.to(device), target.to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(data)
        if cfg.MODEL.NAME.lower() == 'transformer':
            target = target.squeeze(1)[:,-1,:]
        loss = criterion(output, target)
        avg_loss.update(loss.item(), data.size(0))

        # shift the prediction by one time step and calculate the loss
        if cfg.MODEL.NAME.lower() == 'transformer':
            smoothness_loss = 0
        else:
            smoothness_loss = nn.functional.smooth_l1_loss(output.squeeze()[:,:-1,:], output.squeeze()[:, 1:, :])
            smoothness_loss_meter.update(smoothness_loss.item(), data.size(0))

        total_loss = loss + smoothness_loss


        total_loss.backward()
        optimizer.step()

        # if batch_idx % cfg.SOLVER.PRINT_FREQ == 0:
        #     print(f"Epoch: {epoch} Batch: {batch_idx} Loss: {avg_loss.avg}")

    loss_dict = {
        'loss': avg_loss.avg,
        'smoothness_loss': smoothness_loss_meter.avg
    }

    return loss_dict

def test(model, test_loader, criterion, device):
    '''
    Test the model
    '''
    model.eval()
    avg_loss = AverageMeter()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            if cfg.MODEL.NAME.lower() == 'transformer':
                labels = labels.squeeze(1)[:,-1,:]
            loss = criterion(outputs, labels)
            avg_loss.update(loss.item(), data.size(0))

    return avg_loss.avg

def train(model, dataloaders, criterion, optimizer, epochs, logger, device):
    '''
    Train the model
    '''
    logger.info("Started Training ...")
    header = ['Epoch', 'Train Loss', 'Smoothness Loss', 'Val Loss']
    logger.info(''.join([f' {h:<20}' for h in header]))

    for epoch in range(epochs):

        train_loss = train_epoch(cfg, epoch, model, dataloaders['train'], criterion, optimizer, logger=logger, device=device)
        val_loss = test(model, dataloaders['val'], criterion, device=device)

        epoch_values = [epoch, train_loss['loss'], train_loss['smoothness_loss'], val_loss]
        logger.info(''.join([f' {h:<20}' for h in epoch_values]))

        # print(f"Epoch: {epoch} Train Loss: {train_loss['loss']} Smoothness Loss: {train_loss['smoothness_loss']} Val Loss: {val_loss}")
        # logger.info(f"Epoch: {epoch} Train Loss: {train_loss} Smoothness Loss: {train_loss['smoothness_loss']} Val Loss: {val_loss}")
        


def main(cfg, logger):

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    dataset = make_dataset(cfg)

    # make dataloader
    dataloaders = make_dataloader(cfg, dataset)

    # Initialize the model
    model = make_model(cfg)

    # initialize the model with weight from xaviar unifrom dist
    model.apply(weights_init)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    
    # Train the model
    train(model, dataloaders, criterion, optimizer, epochs=cfg.SOLVER.NUM_EPOCHS, logger=logger, device=device)

    # save the model
    torch.save(model.state_dict(), os.path.join(cfg.SOLVER.LOG_DIR, 'model.pth'))

def parse_arg():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arg()

    # only for 
    if cfg.DEBUG:
        cfg.SOLVER.LOG_DIR = "../debug"
        # set the config attribute of args to 
        
    
    # load config file
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.MODEL.NAME)
    cfg.freeze()

    # setup logging
    logger = create_logger(os.path.join(cfg.SOLVER.LOG_DIR, 'train.log'))

    logger.info(f"Running using Config:\n{cfg}\n\n")

    main(cfg, logger)