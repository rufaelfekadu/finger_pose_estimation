import os

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
import numpy as np
# import stratified sampler
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from .EMGLeap import EMGLeap


def train_val_dataset(dataset, val_split=0.3):

    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split, )
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5)

    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)

    return datasets



def make_dataset(cfg):
    # data_path = os.path.join(cfg.DATA.PATH, "data_2023-10-02 14-59-55-627.edf")
    label_path = os.path.join(cfg.DATA.PATH, "label_2023-10-02_15-24-12_YH_lab_R.csv")

    if os.path.isfile(os.path.join(cfg.DATA.PATH, 'dataset.pth')):
        print("Loading saved dataset from {}".format(os.path.join(cfg.DATA.PATH, 'dataset.pth')))
        dataset = torch.load(os.path.join(cfg.DATA.PATH, 'dataset.pth'))
        cfg.DATA.LABEL_COLUMNS = dataset.label_columns
        
    else:
        if cfg.DEBUG:
            dataset = None
        else:
            args = {
                'data_path': cfg.DATA.PATH,
                'seq_len':cfg.DATA.SEGMENT_LENGTH,
                'num_channels':cfg.DATA.EMG.NUM_CHANNELS,
                'stride':cfg.DATA.STRIDE,
                'filter_data':cfg.DATA.FILTER_DATA,
                'fs':cfg.DATA.EMG.SAMPLING_RATE,
                'Q':cfg.DATA.EMG.Q,
                'low_freq':cfg.DATA.EMG.LOW_FREQ,
                'high_freq':cfg.DATA.EMG.HIGH_FREQ,
                'notch_freq':cfg.DATA.EMG.NOTCH_FREQ,
                'ica': cfg.DATA.ICA,
            }
            dataset = EMGLeap(kwargs=args)
            dataset.save_dataset()
            cfg.DATA.LABEL_COLUMNS = dataset.label_columns
            
    return dataset

def make_dataloader(cfg):

    dataset = make_dataset(cfg)
    dataset = train_val_dataset(dataset)

    #save train and val datasets
    torch.save(dataset['train'], os.path.join(cfg.SOLVER.LOG_DIR, 'train_dataset.pth'))
    torch.save(dataset['val'], os.path.join(cfg.SOLVER.LOG_DIR, 'val_dataset.pth'))
    torch.save(dataset['test'], os.path.join(cfg.SOLVER.LOG_DIR, 'test_dataset.pth'))
    dataloader = {}
    dataloader['train'] = DataLoader(dataset['train'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    dataloader['val'] = DataLoader(dataset['val'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    dataloader['test'] = DataLoader(dataset['test'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)

    return dataloader

def read_saved_dataset(cfg, path):
    
    dataset = torch.load(path)
    data_loader= DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)

    return dataset, data_loader

if __name__ == "__main__":
    print("hello")