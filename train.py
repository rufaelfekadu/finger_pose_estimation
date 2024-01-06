
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
from util import create_logger, AverageMeter, AverageMeterList, parse_arg
from loss import make_loss
from data import make_dataset, make_dataloader
from config import cfg
from models import make_model


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

    for batch_idx, (data, target, gesture) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        
        data, target  = data[0].to(device), target.to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(data)
        if cfg.MODEL.NAME.lower() == 'transformer':
            loss_per_keypoint = criterion(output, target)
            loss = loss_per_keypoint.mean()
        else:
            # loss_per_keypoint = criterion(output.squeeze()[:,-1,:], target.squeeze()[:, -1, :])

            loss = nn.functional.mse_loss(output, target)

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
        'loss': avg_loss,
        'smoothness_loss': smoothness_loss_meter
    }

    return loss_dict

def test(cfg, model, test_loader, criterion, device):
    '''
    Test the model
    '''
    model.eval()
    avg_loss = AverageMeter()
    average_loss_per_keypoint = AverageMeterList(label_names=cfg.DATA.LABEL_COLUMNS)
    
    with torch.no_grad():
        for i, (data, target, gestures) in enumerate(test_loader):
            data, target = data[0].to(device), target.to(device)

            output = model(data)

            if cfg.MODEL.NAME.lower() == 'transformer':
                target = target
                loss_per_keypoint = criterion(output, target)
            else:
                loss_per_keypoint = criterion(output.squeeze()[:,-1,:], target.squeeze()[:, -1, :])

            average_loss_per_keypoint.update(loss_per_keypoint, data.size(0))
            loss = loss_per_keypoint.mean()
            avg_loss.update(loss.item(), data.size(0))
            # if i  == len(test_loader) - 1:
            #     # print sample output values
            #     print(f"Sample Output: {output[0]}")
            #     print(f"Sample Target: {target[0]}")

    return avg_loss, average_loss_per_keypoint

def train(model, dataloaders, criterion, optimizer, epochs, logger, device):
    '''
    Train the model
    '''
    logger.info("Started Training ...")
    header = ['Epoch', 'Train Loss', 'Smoothness Loss', 'Val Loss']
    logger.info(''.join([f' {h:<20}' for h in header]))
    best_validation_loss = float('inf')
    counter = 0

    for epoch in range(epochs):

        train_loss = train_epoch(cfg, epoch, model, dataloaders['train'], criterion, optimizer, logger=logger, device=device)
        
        val_loss, _ = test(cfg, model, dataloaders['val'], criterion, device=device)

        epoch_values = [epoch, str(train_loss['loss']), str(train_loss['smoothness_loss']), str(val_loss)]
        logger.info(''.join([f' {h:<20}' for h in epoch_values]))

        # Check for improvement in validation loss
        if val_loss.avg < best_validation_loss:
            best_validation_loss = val_loss.avg
            counter = 0
            # save the model
            dict_to_save = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_validation_loss': best_validation_loss
            }
            torch.save(dict_to_save, os.path.join(cfg.SOLVER.LOG_DIR, f'model_best.pth'))
        else:
            counter += 1
            if counter >= cfg.SOLVER.PATIENCE:
                print(f'Early stopping after epoch {epoch}.')
                break

        # print(f"Epoch: {epoch} Train Loss: {train_loss['loss']} Smoothness Loss: {train_loss['smoothness_loss']} Val Loss: {val_loss}")
        # logger.info(f"Epoch: {epoch} Train Loss: {train_loss} Smoothness Loss: {train_loss['smoothness_loss']} Val Loss: {val_loss}")
    return best_validation_loss

def main(cfg, logger):

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    # dataset = make_dataset(cfg)

    # make dataloader
    dataloaders = make_dataloader(cfg)

    print(next(iter(dataloaders['train']))[1].shape)

    cfg.freeze()

    # # Initialize the model
    model = make_model(cfg)
    print(model)

    # initialize the model with weight from xaviar unifrom dist
    model.apply(weights_init)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = make_loss(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    
    # Train the model
    best_result = train(model, dataloaders, criterion, optimizer, epochs=cfg.SOLVER.NUM_EPOCHS, logger=logger, device=device)
    logger.info(f"Best Validation Loss: {best_result}")

    #final test
    model_best = make_model(cfg)
    model_best.load_pretrained(os.path.join(cfg.SOLVER.LOG_DIR, 'model_best.pth'))
    model_best = model_best.to(device)
    test_loss, per_keypoint_loss = test(cfg, model_best, dataloaders['test'], criterion, device=device)

    print("----------------- Final Results -----------------")
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Loss per Keypoint:\n{per_keypoint_loss}")



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

    # create log directory
    os.makedirs(cfg.SOLVER.LOG_DIR, exist_ok=True)

    # setup logging
    logger = create_logger(os.path.join(cfg.SOLVER.LOG_DIR, 'train.log'))

    logger.info(f"Running using Config:\n{cfg}\n\n")

    main(cfg, logger)