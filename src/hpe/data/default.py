import os

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
import numpy as np
# import stratified sampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from .EMGLeap import EMGLeap
from .transforms import make_transform

exp_setups = {

    'exp0': None,

    'pretrain':{
        'train': ['S1/p1', 'S1/p2', 'S1/p3'],
        'test': ['S1/p3']
    },

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
                'filter_data':cfg.DATA.FILTER,
                'fs':cfg.DATA.EMG.SAMPLING_RATE,
                'Q':cfg.DATA.EMG.Q,
                'low_freq':cfg.DATA.EMG.LOW_FREQ,
                'high_freq':cfg.DATA.EMG.HIGH_FREQ,
                'notch_freq':cfg.DATA.EMG.NOTCH_FREQ,
                'ica': cfg.DATA.ICA,
                'transform': make_transform(cfg),
                'target_transform': None,
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

def train_test_gesture_split(dataset, test_gestures):

    datasets = {}
    train_idx = []
    test_idx = []
    val_idx = []
    for idx, gesture in enumerate(dataset.gestures):
        if dataset.gesture_names_mapping[gesture.item()] in test_gestures:
            test_idx.append(idx)
        elif 'rest' in dataset.gesture_names_mapping[gesture.item()]:
            continue
        else:
            train_idx.append(idx)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, shuffle=True)
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)

    return datasets

def train_val_test(dataset, val_split=0.3, test_split=None):

    datasets = {}
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, shuffle=False)
    if test_split is not None:
        val_idx, test_idx = train_test_split(val_idx, test_size=test_split, shuffle=False)
        datasets['test'] = Subset(dataset, test_idx)

    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)

    return datasets


def make_dataset(cfg):

    save_path = os.path.join(cfg.DATA.PATH, f'dataset_segment_{cfg.DATA.SEGMENT_LENGTH}_stride_{cfg.DATA.STRIDE}.pth')
    if os.path.isfile(save_path):
        print("Loading saved dataset from {}".format(save_path))
        dataset = torch.load(save_path)
        cfg.DATA.LABEL_COLUMNS = dataset.label_columns
        
    else:
        if cfg.DEBUG:
            dataset = None
        else:
            args = make_args(cfg)
            dataset = EMGLeap(kwargs=args)
            dataset.save_dataset(save_path=save_path)
            cfg.DATA.LABEL_COLUMNS = dataset.label_columns

    # rep = np.random.randint(1,5)
    rep = 1
    unique_gestures = np.unique([i.split('_')[0] for i in dataset.gesture_names_mapping.values()])
    # select the rep-th repetition of the gestures in the test set
    
    test_gestures = [i+f'_{rep}' for i in unique_gestures]
    dataset = train_test_gesture_split(dataset, test_gestures=test_gestures)

    # dataset statistics
    print("Number of training examples: {}".format(len(dataset['train'])))
    print("Testing on rep {}".format(rep))
    print("Number of validation examples: {}".format(len(dataset['val'])))
    print("Number of test examples: {}".format(len(dataset['test'])))
    return dataset

def build_dataloader(cfg, save=False, shuffle=False):

    dataset = make_exp_dataset(cfg)

    #save train and val datasets
    if save:
        torch.save(dataset['train'], os.path.join(cfg.SOLVER.LOG_DIR, 'train_dataset.pth'))
        torch.save(dataset['val'], os.path.join(cfg.SOLVER.LOG_DIR, 'val_dataset.pth'))
        torch.save(dataset['test'], os.path.join(cfg.SOLVER.LOG_DIR, 'test_dataset.pth'))

    dataloader = {}
    dataloader['train'] = DataLoader(dataset['train'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.SOLVER.NUM_WORKERS, persistent_workers=True)
    dataloader['val'] = DataLoader(dataset['val'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=cfg.SOLVER.NUM_WORKERS, persistent_workers=True)
    dataloader['test'] = DataLoader(dataset['test'], batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=cfg.SOLVER.NUM_WORKERS, persistent_workers=True)

    return dataloader

def read_saved_dataset(cfg, path):
    dataset = torch.load(path)
    data_loader= DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    return dataset, data_loader

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from config import cfg
    cfg.DATA.PATH = './dataset/FPE/S1/p3'
    cfg.DATA.SEGMENT_LENGTH = 100
    cfg.DATA.STRIDE = 10
    cfg.DEBUG = False
    dataset = make_dataset(cfg)
    