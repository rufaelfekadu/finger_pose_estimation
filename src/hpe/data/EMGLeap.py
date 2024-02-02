from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, stft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Lock, Event
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
    'emg': read_emg_v1,
    'leap': read_leap,
}

class EMGLeap(BaseDataset):

    def __init__(self,kwargs):
        super().__init__( **kwargs)

        self.discritise = True
        
        # read the data
        edf_files, csv_files = self.read_dirs()

        if len(edf_files) == 0:
            raise ValueError(f'No edf files found in {self.data_path}')
        if len(csv_files) == 0:
            raise ValueError(f'No csv files found in {self.data_path}')
        
        self.data_columns = build_emg_columns()
        self.label_columns = build_leap_columns(full=self.visualize)
        self.columns = self.data_columns + self.label_columns + ['gesture_class', 'gesture']

        threads = [None]*len(edf_files)
        results = {
            'merged_data': [None]*len(edf_files),
        }
        self.label_encoder = LabelEncoder()
        self.label_encoder_class = LabelEncoder()
        lock = Lock()
        event = Event()
        #  read the data
        self.data, self.label, self.gestures, self.gesture_label = [], [], [], []
        for i in range(len(edf_files)):
            print(f'Reading data from {edf_files[i]} and {csv_files[i]}')
            thread = Thread(target=self.prepare_data, args=(edf_files[i], csv_files[i], results, i, lock, event))
            threads[i] = thread

        for i in range(len(edf_files)):
            threads[i].start()

        for i in range(len(edf_files)):
            threads[i].join()
        
        merged_df = np.concatenate(results['merged_data'], axis=0)

        if self.discritise:
            self.data = merged_df[:,:,0:len(self.data_columns)]
            self.label = merged_df[:,:,len(self.data_columns):-2]
            self.gesture_label = merged_df[:,0,-2]
            self.gestures = merged_df[:,0,-1]
        else:
            self.data = merged_df[:,0:len(self.data_columns)]
            self.label = merged_df[:,len(self.data_columns):-2]
            self.gesture_label = merged_df[:, -2]
            self.gestures = merged_df[:, -1]

        #  print dataset specs
        self.print_dataset_specs()

        # if self.transform:
        #     self.data = self.transform(self.data)

        if self.target_transform:
            self.label = self.target_transform(self.label)

        # to tensor
        self.data = torch.tensor(self.data.copy(), dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.gestures = torch.tensor(self.gestures, dtype=torch.long)
        self.gesture_label = torch.tensor(self.gesture_label, dtype=torch.long)
        # unfold data and label
        # self.data = self.data.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)
        # self.label = self.label.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)
        # self.gestures = self.gestures.unfold(0, self.seq_len, self.stride)[:,-1] # get the last label in the sequence

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
        
        edf_files = sorted([file for file in all_files if file.endswith('.edf')])
        csv_files = sorted([file for file in all_files if file.endswith('.csv')])

        return edf_files, csv_files
    
    @staticmethod
    def merge_data(emg_data, leap_data):
        #  ajust the time
        start_time = max(min(emg_data.index), min(leap_data.index))
        end_time = min(max(emg_data.index), max(leap_data.index))

        emg_data = emg_data[start_time:end_time]
        leap_data = leap_data[start_time:end_time]

        data = pd.merge_asof(emg_data, leap_data, left_index=True, right_index=False, right_on='time', direction='backward', tolerance=pd.to_timedelta(10, unit='ms'))
        data['gesture_class'] = data['gesture'].apply(lambda x: x.split('_')[0])
        
        # data['time_diff'] = (data.index - data['time_leap']).dt.total_seconds()
        # data.drop(columns=['timestamp', 'frame_id', 'time_leap'], inplace=True)


        #  reorder columns to have gesture at the end
        if 'gesture' in data.columns:
            data = data[[col for col in data.columns if col != 'gesture'] + ['gesture']]

        return data

    def print_dataset_specs(self):
        print("-----------------Dataset specs-----------------")
        print(f"Number of examples: {self.data.shape[0]}")
        print(f"Sequence length: {self.seq_len}")
        print(f"Number of channels: {self.data.shape[2]}")
        print(f"Number of gestures: {self.gestures.max().item()+1}")
        print(f"Number of classes: {len(self.label_columns)}")
        print(f"Label columns: {self.label_columns}")

    @staticmethod
    def interpolate_missing_values(self, data):
        #  drop group if count of nulls > 30%
        return data.groupby('gesture').filter(lambda x: x.isnull().sum()[self.label_columns[0]] < 300 ).apply(lambda x: x.fillna(x.mean()))

    @staticmethod
    def discritise_data(data, seq_len=150, stride=5):
        # assert gesture is in the last column
        assert data.columns[-1] == 'gesture', 'gesture should be the last column'
        grouped = data.groupby('gesture')

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
    
    def prepare_data(self, data_path, label_path, results={}, index=0, lock=None, event=None):

        data =  DATA_SOURCES['emg'](data_path)
        label = DATA_SOURCES['leap'](label_path, rotations=True, positions=False, visualisation=self.visualize)

        merged_df = self.merge_data(data, label)

        # label encoder for gesture
        
        if index == 0:
            lock.acquire()
            try:
                    self.label_encoder.fit(merged_df['gesture'].unique())
                    self.label_encoder_class.fit(merged_df['gesture_class'].unique())
                    self.gesture_names_mapping = {i: gesture for i, gesture in enumerate(self.label_encoder.classes_)}
                    self.gesture_names_mapping_class = {i: gesture for i, gesture in enumerate(self.label_encoder_class.classes_)}
                
            finally:
                lock.release()
            event.set()
        else:
            event.wait()
            
        #  drop null enries
        # merged_df.dropna(inplace=True)

        merged_df['gesture'] = self.label_encoder.transform(merged_df['gesture'])
        merged_df['gesture_class'] = self.label_encoder_class.transform(merged_df['gesture_class'])

        #  interpolate missing values
        merged_df = self.interpolate_missing_values(self,merged_df)

        # apply transform to emg data
        if self.transform:
            lock.acquire()
            try:
                merged_df[self.data_columns] = self.transform(merged_df[self.data_columns].values)
            finally:
                lock.release()
        merged_df = merged_df[self.columns]

        #  discritise data
        if self.discritise:
            merged_data = self.discritise_data(data=merged_df, seq_len=self.seq_len, stride=self.stride)
        else:
            merged_data = merged_df.values

        results['merged_data'][index] = merged_data
        
        return merged_df
    
    
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
        return self.data[idx], self.label[idx], (self.gesture_label[idx], self.gestures[idx])
    # def __getitem__(self, idx):
    #     return None, self.label[idx], None

if __name__ == '__main__':

    train_transform = transforms.Compose([
        StandardScalerTransform(),
        FilterTransform(fs=250, notch_freq=50, Q=30, lowcut=20, highcut=55),
        transforms.ToTensor(),
    ])

    kwargs = {
        'data_path': './dataset/emgleap/003/S1',
        'seq_len': 150,
        'num_channels': 16,
        # filter info
        'filter_data': True,
        'fs': 150,
        'notch_freq': 50,
        'Q': 30,
        'low_freq': 20,
        'high_freq': 55,
        'stride': 10,
        'data_source': 'emg',
        'ica': False,
        'transform': None,
        'target_transform': None,
        'visualize': False
    }

    dataset = EMGLeap(kwargs=kwargs)
    print(dataset.data.shape)
