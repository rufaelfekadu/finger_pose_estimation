import os

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
import numpy as np
# import stratified sampler
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from .EMGLeap import EMGLeap

exp_setups = {

    'exp0': None,

    'exp1': {
        'train': ['S1/p4', 'S1/p2'],
        'test': ['S1/p1']
    },

    'exp2': {
        'train': ['S1/p1', 'S1/p2', 'S1/p3'],
        'test': ['S1/p4']
    },
}

def make_args(cfg):
    data_args = {
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
    return data_args

def get_dirs_for_exp(cfg):

    data_path = cfg.DATA.PATH

    if cfg.DATA.EXP_SETUP not in exp_setups.keys():
        raise ValueError(f'Invalid experiment setup {cfg.DATA.EXP_SETUP}')
    
    train_dirs = []
    test_dirs = []

    for dir in exp_setups[cfg.DATA.EXP_SETUP]['train']:
        train_dirs.append(os.path.join(data_path, dir))

    for dir in exp_setups[cfg.DATA.EXP_SETUP]['test']:
        test_dirs.append(os.path.join(data_path, dir))

    return train_dirs, test_dirs


def make_exp_dataset(cfg,):

    if not cfg.DATA.EXP_SETUP or cfg.DATA.EXP_SETUP == 'exp0':
        #  do default train val test split
        dataset = make_dataset(cfg)
        print(f'Running experiment setup {cfg.DATA.EXP_SETUP}')
    else:

        train_dirs, test_dirs = get_dirs_for_exp(cfg)
        args = make_args(cfg)

        args['data_path'] = train_dirs
        train_dataset = EMGLeap(args)

        args['data_path'] = test_dirs
        test_dataset = EMGLeap(args)

        dataset = train_val_test(train_dataset, val_split=0.3)
        dataset['test'] = test_dataset
        cfg.DATA.LABEL_COLUMNS = train_dataset.label_columns
        print(f"Runnig experiment setup {cfg.DATA.EXP_SETUP} with \n\ntrain: {exp_setups[cfg.DATA.EXP_SETUP]['train']}\nand test: {exp_setups[cfg.DATA.EXP_SETUP]['test']}\n\n")

        #  print some statistics about the dataset
        print(f"Number of training examples: {len(dataset['train'].dataset)}")
        print(f"Number of validation examples: {len(dataset['val'].dataset)}")
        print(f"Number of test examples: {len(dataset['test'])}")
        print(f"Number of classes: {len(dataset['test'].label_columns)}")


    return dataset

def train_val_test(dataset, val_split=0.3, test_split=None):

    datasets = {}
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, )
    if test_split is not None:
        val_idx, test_idx = train_test_split(val_idx, test_size=test_split)
        datasets['test'] = Subset(dataset, test_idx)

    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)

    return datasets


def make_dataset(cfg):

    if os.path.isfile(os.path.join(cfg.DATA.PATH, 'dataset.pth')):
        print("Loading saved dataset from {}".format(os.path.join(cfg.DATA.PATH, 'dataset.pth')))
        dataset = torch.load(os.path.join(cfg.DATA.PATH, 'dataset.pth'))
        cfg.DATA.LABEL_COLUMNS = dataset.label_columns
        
    else:
        if cfg.DEBUG:
            dataset = None
        else:
            args = make_args(cfg)
            dataset = EMGLeap(kwargs=args)
            dataset.save_dataset(os.path.join(cfg.DATA.PATH, 'dataset.pth'))
            cfg.DATA.LABEL_COLUMNS = dataset.label_columns

    dataset = train_val_test(dataset, val_split=0.3, test_split=0.5)

    return dataset

def make_dataloader(cfg):

    dataset = make_exp_dataset(cfg)

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