from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, stft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import json
from torchvision import transforms

import os
import glob

import sys
from hpe.util.data import *
from hpe.data.base import BaseDataset
from hpe.data.transforms import FilterTransform, StandardScalerTransform, FastICATransform
from memory_profiler import profile

# Add data sources here
# TODO: 
# 1. Add ultraleap data source
# 2. Add video data source


DATA_SOURCES = {
    'manus': read_manus,
    'emg': read_emg,
    'leap': read_leap,
}

class EMGLeap(BaseDataset):
    def __init__(self,kwargs):
        super().__init__( **kwargs)

        # read the data
        edf_files, csv_files = self.read_dirs()

        if len(edf_files) == 0:
            raise ValueError(f'No edf files found in {self.data_path}')
        if len(csv_files) == 0:
            raise ValueError(f'No csv files found in {self.data_path}')
        

        threads = [None]*len(edf_files)
        results = {
            'data': [None]*len(edf_files),
            'label': [None]*len(edf_files),
            'gestures': [None]*len(edf_files)
        }

        #  read the data
        self.data, self.label, self.gestures = [], [], []
        for i in range(len(edf_files)):
            print(f'Reading data from {edf_files[i]} and {csv_files[i]}')
            thread = Thread(target=self.prepare_data, args=(edf_files[i], csv_files[i], results, i))
            threads[i] = thread

        for i in range(len(edf_files)):
            threads[i].start()

        for i in range(len(edf_files)):
            threads[i].join()
        
        self.data = np.concatenate(results['data'], axis=0)
        self.label = np.concatenate(results['label'], axis=0)
        self.gestures = np.concatenate(results['gestures'], axis=0)


        #  print dataset specs
        self.print_dataset_specs()

        # if self.ica:
        #     self.apply_ica_to_emg()
        # else:
        #     self.data_ica = None
        #     self.mixing_matrix = None

        if self.transform:
            self.data = self.transform(self.data)

        if self.target_transform:
            self.label = self.target_transform(self.label)
        
        # label encode gestures
        self.label_encoder = LabelEncoder()
        self.gestures = self.label_encoder.fit_transform(self.gestures)
        self.gesture_names_mapping = {i: gesture for i, gesture in enumerate(self.label_encoder.classes_)}

        # to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.gestures = torch.tensor(self.gestures, dtype=torch.long)
        
        # unfold data and label
        self.data = self.data.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)
        self.label = self.label.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)
        self.gestures = self.gestures.unfold(0, self.seq_len, self.stride)[:,-1] # get the last label in the sequence

    def read_dirs(self):

        if isinstance(self.data_path, str):
            self.data_path = [self.data_path]
        all_files = []
        for path in self.data_path:
            if not os.path.isdir(path):
                raise ValueError(f'{path} is not a directory')
            else:
                print(f'Reading data from {path}')
                all_files += [f for f in glob.glob(os.path.join(path, '**/*'), recursive=True) if os.path.splitext(f)[1] in ['.edf', '.csv']]
        
        # # Traverse through all the directories and read the data
        # all_files = [f for f in glob.glob(os.path.join(self.data_path, '**/*'), recursive=True) if os.path.splitext(f)[1] in ['.edf', '.csv']]
        # Separate .edf and .csv files
                
        edf_files = sorted([file for file in all_files if file.endswith('.edf')])
        csv_files = sorted([file for file in all_files if file.endswith('.csv')])

        return edf_files, csv_files
    

    def print_dataset_specs(self):
        print("data shape: ", self.data.shape)

    
    def prepare_data(self, data_path, label_path, results={}, index=0):

        #  read json file and get start and end time
        
        # stat = json.load(open(os.path.join(data_path, 'log.json'), 'r'))


        data =  DATA_SOURCES['emg'](data_path)
        label = DATA_SOURCES['leap'](label_path, rotations=True, positions=False)

        if index == 0:
            #save the column names for the label
            self.label_columns = list(label.columns)
            self.data_columns = list(data.columns)
            #  remove gesture column
            self.data_columns.remove('gesture')
        
        # set the start and end of experiment
        start_time = max(min(data.index), min(label.index))
        end_time = min(max(data.index), max(label.index))

        # select only the data between start and end time
        data = data.loc[start_time:end_time]
        label = label.loc[start_time:end_time]

        # merge dataframes
        merged_df = pd.merge_asof(data, label, left_index=True, right_index=True, direction='forward')

        #  remove rest positions
        merged_df = merged_df[merged_df['gesture'] != 'rest']

        # data, label, gestures = create_windowed_dataset(data, label, annotations, self.seq_len, self.stride)
        # label, gestures = find_closest(label, label_index, annotations)

        gestures = merged_df['gesture'].values
        data = merged_df[self.data_columns].values
        label = merged_df[self.label_columns].values

        # normalize the data
        # data, label = self.normalize_and_filter(data, label)

        results['data'][index] = data
        results['label'][index] = label
        results['gestures'][index] = gestures

        return data, label, gestures
    
    def normalize_and_filter(self, data=None, label=None):

        N, C, L = data.shape
        data_sliced = data.reshape(-1, L)

        # normalize the data
        self.data_scaler = StandardScaler()
        data_sliced = self.data_scaler.fit_transform(data_sliced)

        # self.label_scaler = StandardScaler()
        # label = self.label_scaler.fit_transform(label)

        print("Filtering data...")
        # filter the data
        if self.filter_data:
            data_sliced = self._filter_data(data_sliced)

        return data_sliced.reshape(N, C, L), label
    
    def apply_ica_to_emg(self):
        # TODO: apply ICA to the EMG data
        # Reshape data to 2D
        N, L, C = self.data.shape
        #  copy data
        import copy
        data = copy.deepcopy(self.data)
        data = data.reshape(-1, C)

        # Apply ICA
        ica = FastICA(n_components=C)
        data = ica.fit_transform(data)

        # Get the mixing matrix
        self.mixing_matrix = ica.mixing_

        # Reshape data back to 3D
        self.data_ica = data.reshape(N, L, C)

        # save the mixing matrix
        # torch.save(self.mixing_matrix, os.path.join(self.data_path, 'mixing_matrix.pth'))
    
    def plot_data(self, save_dir=None):

        '''
        get random index and plot data
        '''

        idx = np.random.randint(0, self.data.shape[0])
        data = self.data[idx]
        label = self.label[idx]
        gestures = self.gestures[idx]
        data_ica = self.data_ica[idx]

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        # plot as heatmap
        axs[0].imshow(data.T, aspect='auto')
        axs[0].set_title('EMG data')

        axs[1].plot(label)

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'plot_{idx}.png'))
    
    def save_dataset(self, save_path=None):
        if save_path is None:
            raise ValueError('save_dir cannot be None')
        torch.save(self, save_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # if self.ica:
        #     return (self.data[idx], self.data_ica[idx]), self.label[idx], self.gestures[idx]
        # else:
        return self.data[idx], self.label[idx], self.gestures[idx]
        

if __name__ == '__main__':

    train_transform = transforms.Compose([
        StandardScalerTransform(),
        FilterTransform(fs=250, notch_freq=50, Q=30, lowcut=20, highcut=55),
        transforms.ToTensor(),
    ])

    kwargs = {
        'data_path': './dataset/FPE/S1/p3',
        'seq_len': 50,
        'num_channels': 16,
        # filter info
        'filter_data': True,
        'fs': 150,
        'notch_freq': 50,
        'Q': 30,
        'low_freq': 20,
        'high_freq': 55,
        'stride': 25,
        'data_source': 'emg',
        'ica': False,
        'transform': None,
        'target_transform': None,
    }

    dataset = EMGLeap(kwargs=kwargs)
    print(dataset.data.shape)
