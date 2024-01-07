from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, stft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread

import os
import glob

import sys
sys.path.append('/Users/rufaelmarew/Documents/tau/finger_pose_estimation')
from util.data import *
from config import cfg
from .base import BaseDataset

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
            thread.start()
            # data, label, gestures = self.prepare_data(edf_files[i], csv_files[i], results, i)
            # self.data.append(data)
            # self.label.append(label)
            # self.gestures.append(gestures)


        for i in range(len(edf_files)):
            threads[i].join()

        # self.data = np.concatenate(self.data, axis=0)
        # self.label = np.concatenate(self.label, axis=0)
        # self.gestures = np.concatenate(self.gestures, axis=0)
        
        self.data = np.concatenate(results['data'], axis=0)
        self.label = np.concatenate(results['label'], axis=0)
        self.gestures = np.concatenate(results['gestures'], axis=0)


        #  print dataset specs
        self.print_dataset_specs()

        if self.ica:
            self.apply_ica_to_emg()
        else:
            self.data_ica = None
            self.mixing_matrix = None

        # to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)

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

        data, annotations, header =  DATA_SOURCES['emg'](data_path)
        label, _, _ = DATA_SOURCES['leap'](label_path)

        if index == 0:
            #save the column names for the label
            self.label_columns = list(label.columns)
            self.data_columns = list(data.columns)
        
        # set the start and end of experiment
        start_time = max(min(data.index), min(label.index))
        end_time = min(max(data.index), max(label.index))

        # select only the data between start and end time
        data = data.loc[start_time:end_time]
        label = label.loc[start_time:end_time]

        data, label, gestures = create_windowed_dataset(data, label, annotations, self.seq_len, self.stride)

        # label, gestures = find_closest(label, label_index, annotations)

        # normalize the data
        data = self.normalize_and_filter(data)

        results['data'][index] = data
        results['label'][index] = label
        results['gestures'][index] = gestures

        # convert to tensor
        # self.data = torch.tensor(self.data, dtype=torch.float32)
        # self.label = torch.tensor(self.label, dtype=torch.float32)
        return data, label, gestures
    
    def normalize_and_filter(self, data=None):

        N, C, L = data.shape
        data_sliced = data.reshape(-1, L)

        # normalize the data
        scaler = StandardScaler()
        data_sliced = scaler.fit_transform(data_sliced)

        print("Filtering data...")
        # filter the data
        if self.filter_data:
            data_sliced = self._filter_data(data_sliced)

        return data_sliced.reshape(N, C, L)
    
    def apply_ica_to_emg(self):
        # TODO: apply ICA to the EMG data
        # Reshape data to 2D
        N, L, C = self.data.shape
        #  copy data
        data = self.data.clone()
        data = data.reshape(-1, C)

        # Apply ICA
        ica = FastICA(n_components=C)
        data = ica.fit_transform(data)

        # Get the mixing matrix
        self.mixing_matrix = ica.mixing_

        # Reshape data back to 3D
        self.data_ica = data.reshape(N, L, C)

        # save the mixing matrix
        torch.save(self.mixing_matrix, os.path.join(self.data_path, 'mixing_matrix.pth'))
    
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

        axs[1].imshow(data_ica.T, aspect='auto')
        axs[1].set_title('ICA transformed data')

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'plot_{idx}.png'))
    
    def save_dataset(self, save_path=None):
        if save_path is None:
            raise ValueError('save_dir cannot be None')
        torch.save(self, save_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.ica:
            if self.transform:
                return self.transform(self.data[idx]), self.data_ica[idx], self.label[idx], self.gestures[idx]
            return self.data[idx], self.data_ica[idx], self.label[idx], self.gestures[idx]
        else:
            return (self.data[idx], np.NAN),  self.label[idx], self.gestures[idx]

def get_ica_components(data, mixing_matrix):
    #  return the ICA components
    return np.matmul(data, mixing_matrix)

class ICATransform(object):
    def __init__(self, mixing_matrix):
        self.mixing_matrix = mixing_matrix
        pass

    def __call__(self, sample):
        #  return the sample as well as the ICA transformed sample
        sample = torch.tensor(sample, dtype=torch.float32)
        sample_ica = torch.matmul(sample, self.mixing_matrix)
        return sample, sample_ica
    
if __name__ == '__main__':

    kwargs = {
        'data_path': '/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/data_2023-10-02 14-59-55-627.edf',
        'label_path': '/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/label_2023-10-02_15-24-12_YH_lab_R.csv',
        'seq_len': 150,
        'num_channels': 16,
        # filter info
        'filter_data': True,
        'fs': 150,
        'notch_freq': 50,
        'Q': 30,
        'low_freq': 20,
        'high_freq': 55,
        'stride': 1,
        'label_source': 'manus',
        'data_source': 'emg'
    }

    dataset = EMGLeap(ica=False, kwargs=kwargs)
    print(dataset.data.shape)
