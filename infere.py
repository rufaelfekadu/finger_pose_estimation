import torch 
from data import read_saved_dataset
from util import create_logger, plot_time_series
import argparse
import os
from config import cfg
from models import NeuroPose, TransformerModel, make_model
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot(cfg, label, output):
    #plot the label and predictions
    
    fig, axs = plt.subplots(output.size(1), 1, figsize=(10, 10))
    print(output.size(1))
    for i in range(output.size(1)):
        axs[i].plot(label[:, i], label=f'label_{i}')
        axs[i].plot(output[:, i], label=f'output_{i}')
        axs[i].legend()
    plt.savefig(os.path.join(cfg.SOLVER.LOG_DIR, 'plot.png'))

def setup(cfg):

     # read the data
    data_path = os.path.join(cfg.SOLVER.LOG_DIR, 'test_dataset.pth')
    data, dataloader = read_saved_dataset(cfg, data_path)
    cfg.DATA.MANUS.KEY_POINTS = [i for i in range(15)]
    print(len(data))
    # load the model
    model = make_model(cfg)
    model.load_pretrained(os.path.join(cfg.SOLVER.LOG_DIR, 'model_best.pth'))

    return model, dataloader

def inference(cfg, logger):

    #setup device
    device = "cpu"

    model_transformer, data_transformer = setup(cfg)
    
    criterion = nn.MSELoss(reduction='mean')

    # set the model to eval mode
    model_transformer.eval()
    total_loss_trans = 0
    pred_trans = []
    to_plot_trans = []
    for i, (data, label) in tqdm(enumerate(data_transformer), total=len(data_transformer)):

        data, label = data.to(device), label.to(device)
        output = model_transformer(data)
        loss = criterion(output, label.squeeze(1)[:,-1,:])
        total_loss_trans += loss.item()

        pred_trans.append(output.detach().cpu())
        to_plot_trans.append(label.squeeze(1)[:,-1,:].detach().cpu())


    pred_trans = torch.cat(pred_trans, dim=0)
    to_plot_trans = torch.cat(to_plot_trans, dim=0)

    # torch.save(pred_cache, os.path.join(cfg.SOLVER.LOG_DIR, cfg.MODEL.NAME, 'pred_cache.pth'))
    plot(cfg, to_plot_trans[:200,:], pred_trans[:200])
    logger.info(f"Total loss for transformer: {total_loss_trans/len(data_transformer)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    if cfg.DEBUG:
        cfg.SOLVER.LOG_DIR = "../debug"
    #setup logger
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.MODEL.NAME)

    # cfg.freeze()
    logger = create_logger(os.path.join(cfg.SOLVER.LOG_DIR, 'inference.log'))

    inference(cfg, logger)
