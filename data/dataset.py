import os

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch

from sklearn.model_selection import train_test_split

from .EMGDataset import EMGDataset, TestDataset


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, )
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def make_dataset(cfg):
    data_path = os.path.join(cfg.DATA.PATH, "data_2023-10-02 14-59-55-627.edf")
    label_path = os.path.join(cfg.DATA.PATH, "label_2023-10-02_15-24-12_YH_lab_R.csv")
    dataset = EMGDataset(data_path=data_path, 
                         label_path=label_path,
                         transform=None,
                         data_source='emg',
                         label_source='manus',
                         seq_len=cfg.DATA.SEGMENT_LENGTH,
                         num_channels=cfg.DATA.EMG.NUM_CHANNELS)
    # dataset = TestDataset()
    return dataset

def make_dataloader(cfg, dataset):

    dataset = train_val_dataset(dataset)

    #save train and val datasets
    torch.save(dataset['train'], os.path.join(cfg.SOLVER.LOG_DIR, 'train_dataset.pth'))
    torch.save(dataset['val'], os.path.join(cfg.SOLVER.LOG_DIR, 'val_dataset.pth'))
    dataloader = {}
    dataloader['train'] = DataLoader(dataset['train'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    dataloader['val'] = DataLoader(dataset['val'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)

    return dataloader

def read_saved_dataset(cfg):
    dataset = {}
    train_dataset = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'train_dataset.pth'))
    val_dataset = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'val_dataset.pth'))

    dataset['train'] = DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    dataset['val'] = DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    
    return dataset
