import torch
from torch.fft import fft, rfft
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from hpe.util.data import read_dirs, train_test_gesture_split, strided_array
from hpe.data.transforms import JitterTransform, FrequencyTranform


class EmgDataset(Dataset):
    def __init__(self, cfg, training_mode='pretrain', transform_t=None, transform_f=None):

        merged_data = np.array([])
        self.transform_t = transform_t
        self.transform_f = transform_f
        self.seq_len = cfg.DATA.SEGMENT_LENGTH
        self.stride = cfg.DATA.STRIDE
        self.training_mode = training_mode

        for edf_file, _ in tqdm(zip(*read_dirs(cfg.DATA.PATH))):
            file_name = ".npz"
            np_file = os.path.splitext(edf_file)[0] + file_name
            if os.path.isfile(np_file):
                # load dataset
                loaded = np.load(np_file,  allow_pickle=True)
                data = {key: loaded[key] for key in loaded.files}
                d = self.discritise_data(data['data'], seq_len=self.seq_len, stride=self.stride)
                merged_data = np.concatenate((merged_data, d), axis=0) if len(merged_data) > 0 else d
            else:
                # skip
                continue

        self.label_columns = data['label_columns']
        self.data_columns = data['data_columns']
        self.gesture_mapping = data['gesture_mapping']
        self.gesture_mapping_class = data['gesture_mapping_class']

        # discritise data grouped by the last column
        # merged_data = self.discritise_data(merged_data, seq_len=self.seq_len, stride=self.stride)

        self.data = merged_data[:, :, 0:len(self.data_columns)]
        self.label = merged_data[:, :, len(self.data_columns):-2]
        self.gesture_class = merged_data[:, :, -2]
        self.gestures = merged_data[:, :, -1]


        #  to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.gesture_class = torch.tensor(self.gesture_class, dtype=torch.long)
        self.gestures = torch.tensor(self.gestures, dtype=torch.long)

        #  fft 
        self.data_f = fft(self.data, dim=1).abs()
        
        if self.training_mode == 'pretrain':
            # time augmentations apply jitter augmentation
            self.aug1_t = self.transform_t(self.data).float()
            # frequency augmentations 
            self.aug1_f = self.transform_f(self.data_f).float()

    def discritise_data(self, data, seq_len=150, stride=5):
        data = pd.DataFrame(data)
        grouped = data.groupby(data.iloc[:, -1])

        # Initialize an empty list to store the strided arrays
        strided_arrays = []

        # Iterate over the groups
        for _, group in grouped:
            # Convert the group to a numpy array
            array = np.array(group)
            # Generate the strided array and append it to the list
            # assert the shape of the array is greater than the sequence length
            if array.shape[0] > seq_len:
                strided_arrays.append(strided_array(array, seq_len, stride))
            else:
                print(f'Skipping {group.iloc[0]["gesture"]}, not enough data')

        # Concatenate the strided arrays into a single array and return it
        return np.concatenate(strided_arrays, axis=0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.training_mode == 'pretrain':
            return self.data[idx], self.aug1_t[idx], self.data_f[idx], \
                self.aug1_f[idx], self.gesture_class[idx]
        else:
            return self.data[idx], self.data_f[idx], self.label[idx], self.gesture_class[idx] 


exp_setups = {
    'pretrain':{
        'pretrain': ['003'],
        'train': ['004/S1/P2', '004/S1/P3'],
        'test': ['004/S1/P4']
    },
}
def get_dirs_for_exp(cfg):

    data_path = cfg.DATA.PATH

    if cfg.DATA.EXP_SETUP not in exp_setups.keys():
        raise ValueError(f'Invalid experiment setup {cfg.DATA.EXP_SETUP}')
    
    pretrain_dirs = []
    train_dirs = []
    test_dirs = []

    for dir in exp_setups[cfg.DATA.EXP_SETUP]['pretrain']:
        pretrain_dirs.append(os.path.join(data_path, dir))

    for dir in exp_setups[cfg.DATA.EXP_SETUP]['train']:
        train_dirs.append(os.path.join(data_path, dir))

    for dir in exp_setups[cfg.DATA.EXP_SETUP]['test']:
        test_dirs.append(os.path.join(data_path, dir))

    return pretrain_dirs, train_dirs, test_dirs

def build_dataloaders(cfg):

    pretrain_dirs, train_dirs, test_dirs = get_dirs_for_exp(cfg)
    dataloaders = {}

    cfg.DATA.PATH = pretrain_dirs
    pretrain_set = EmgDataset(cfg, training_mode='pretrain',
                              transform_f=FrequencyTranform(fs=cfg.DATA.EMG.SAMPLING_RATE, pertub_ratio=cfg.DATA.FREQ_PERTUB_RATIO), 
                              transform_t=JitterTransform(scale=cfg.DATA.JITTER_SCALE))
    
    cfg.DATA.PATH = train_dirs
    cfg.DATA.LABEL_COLUMNS = pretrain_set.label_columns.tolist()
    train_set = EmgDataset(cfg, training_mode='hpe')

    rep = np.random.randint(1,5)

    #  use one of the repititions as validation
    unique_gestures = np.unique(train_set.gesture_mapping_class)
    test_gestures = [i+f'_{rep}' for i in unique_gestures]

    train_set, val_set, test_set = train_test_gesture_split(train_set, test_gestures=test_gestures)

    cfg.DATA.PATH = test_dirs
    test_set_2 = EmgDataset(cfg, training_mode='hpe')

    dataloaders['pretrain'] = torch.utils.data.DataLoader(pretrain_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True)
    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    dataloaders['test_2'] = torch.utils.data.DataLoader(test_set_2, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)
    # split test into validation and test

    return dataloaders

