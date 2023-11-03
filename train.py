from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import make_dataset, make_dataloader
from config import cfg
from models import NeuroPose
import argparse
import os
from tqdm import tqdm

#setup logger
import logging


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

def train_epoch(cfg, epoch, model, train_loader, criterion, optimizer):

    model.train()

    avg_loss = AverageMeter()

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        avg_loss.update(loss.item(), data.size(0))

        loss.backward()
        optimizer.step()

        if batch_idx % cfg.SOLVER.PRINT_FREQ == 0:
            print(f"Epoch: {epoch} Batch: {batch_idx} Loss: {avg_loss.avg}")

        
    return avg_loss.avg

def test(model, test_loader, criterion):
    '''
    Test the model
    '''
    model.eval()
    avg_loss = AverageMeter()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            outputs = model(data)
            loss = criterion(outputs, labels)
            avg_loss.update(loss.item(), data.size(0))

    return avg_loss.avg

def train(model, dataloaders, criterion, optimizer, epochs, logger):
    '''
    Train the model
    '''
    print('Training the model')
    for epoch in range(epochs):
        train_loss = train_epoch(cfg, epoch, model, dataloaders['train'], criterion, optimizer)
        val_loss = test(model, dataloaders['val'], criterion)

        # if epoch % cfg.SOLVER.PRINT_FREQ == 0:

        print(f"Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss}")
        logger.info(f"Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss}")



def main(cfg, logger):

    # Load the dataset
    dataset = make_dataset(cfg)

    # make dataloader
    dataloaders = make_dataloader(cfg, dataset)

    # Initialize the model
    model = NeuroPose()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    
    # Train the model
    train(model, dataloaders, criterion, optimizer, epochs=cfg.SOLVER.NUM_EPOCHS, logger=logger)

def parse_arg():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='finger_pose_estimation/config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # setup logging
    logging.basicConfig(filename=os.path.join(cfg.SOLVER.LOG_DIR, 'train.log'), level=logging.INFO)
    logger = logging.getLogger(__name__)


    main(cfg, logger)