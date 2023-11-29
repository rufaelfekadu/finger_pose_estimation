
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import os
from tqdm import tqdm
#setup logger
import logging

# imports form FGR module
from util import create_logger, AverageMeter, AverageMeterList
from loss import make_loss
from data import make_dataset, make_dataloader
from config import cfg
from models import NeuroPose, TransformerModel, make_model


def weights_init(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0.1)
    


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
        loss_per_keypoint = criterion(output, target)
        loss = loss_per_keypoint.mean()
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

        if epoch % 5 == 0 and batch_idx == 0:
            # print sample output values
            print(f"Sample Output: {output[0]}")
            print(f"Sample Target: {target[0]}")
        # if batch_idx % cfg.SOLVER.PRINT_FREQ == 0:
        #     print(f"Epoch: {epoch} Batch: {batch_idx} Loss: {avg_loss.avg}")

    loss_dict = {
        'loss': avg_loss,
        'smoothness_loss': smoothness_loss_meter
    }

    return loss_dict

def test(model, test_loader, criterion, device):
    '''
    Test the model
    '''
    model.eval()
    avg_loss = AverageMeter()
    average_loss_per_keypoint = AverageMeterList(label_names=cfg.DATA.MANUS.KEY_POINTS)
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            if cfg.MODEL.NAME.lower() == 'transformer':
                labels = labels.squeeze(1)[:,-1,:]
            loss_per_keypoint = criterion(outputs, labels)
            average_loss_per_keypoint.update(loss_per_keypoint, data.size(0))
            loss = loss_per_keypoint.mean()
            avg_loss.update(loss.item(), data.size(0))

    return avg_loss, average_loss_per_keypoint

def train(model, dataloaders, criterion, optimizer, epochs, logger, device):
    '''
    Train the model
    '''
    logger.info("Started Training ...")
    header = ['Epoch', 'Train Loss', 'Smoothness Loss', 'Val Loss']
    logger.info(''.join([f' {h:<20}' for h in header]))

    for epoch in range(epochs):

        train_loss = train_epoch(cfg, epoch, model, dataloaders['train'], criterion, optimizer, logger=logger, device=device)
        val_loss, _ = test(model, dataloaders['val'], criterion, device=device)

        epoch_values = [epoch, str(train_loss['loss']), str(train_loss['smoothness_loss']), str(val_loss)]
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

    print(next(iter(dataloaders['train']))[0].shape)
    print(dataset.label_columns)
    cfg.DATA.MANUS.KEY_POINTS = list(dataset.label_columns)
    cfg.freeze()

    # # Initialize the model
    model = make_model(cfg)

    # initialize the model with weight from xaviar unifrom dist
    model.apply(weights_init)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = make_loss(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    
    # Train the model
    train(model, dataloaders, criterion, optimizer, epochs=cfg.SOLVER.NUM_EPOCHS, logger=logger, device=device)

    #final test
    test_loss, per_keypoint_loss = test(model, dataloaders['val'], criterion, device=device)
    print("----------------- Final Results -----------------")
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Loss per Keypoint:\n{per_keypoint_loss}")
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

    # setup logging
    logger = create_logger(os.path.join(cfg.SOLVER.LOG_DIR, 'train.log'))

    logger.info(f"Running using Config:\n{cfg}\n\n")

    main(cfg, logger)