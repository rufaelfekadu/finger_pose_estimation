from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, stft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import os

import sys
sys.path.append('/Users/rufaelmarew/Documents/tau/finger_pose_estimation')
from util.data import read_emg, read_manus, read_leap, build_leap_columns, build_manus_columns
from config import cfg

# Add data sources here
# TODO: 
# 1. Add ultraleap data source
# 2. Add video data source


DATA_SOURCES = {
    'manus': read_manus,
    'emg': read_emg,
    'leap': read_leap,
}

class BaseDataset(Dataset):
    def __init__(self, transform=None, **kwargs):
        self.fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        self.joints = ['CMC', 'MCP', 'PIP', 'DIP']
        self.movements = ['Flex', 'Spread']
        self.ROTATIONS = ['x', 'y', 'z', 'w']

        self.data_path = data_path

        self.seq_len = seq_len
        self.stride = stride
        self.num_channels = num_channels

        #filter info
        self.filter_data = filter_data
        self.fs = sampling_freq
        self.notch_freq = notch_freq
        self.Q = Q
        self.low_freq = low_freq
        self.high_freq = high_freq

        self.transform = transform

        self.data_source = data_source # emg or imu
        self.label_source = label_source   # manus, video, or ultraleap 

    def apply_ica_to_emg(self):
        pass
        
    def prepare_data(self):
        pass

    def _filter_data(self, data: np.ndarray, buff_len: int = 0) -> np.ndarray:

        # Calculate the normalized frequency and design the notch filter
        w0 = self.notch_freq / (self.fs / 2)
        b_notch, a_notch = iirnotch(w0, self.Q)

        #calculate the normalized frequencies and design the highpass filter
        cutoff = self.low_freq / (self.fs / 2)
        sos = butter(5, cutoff, btype='highpass', output='sos')

        # apply filters using 'filtfilt' to avoid phase shift
        data = sosfiltfilt(sos, data, axis=0, padtype='even')
        data = filtfilt(b_notch, a_notch, data)

        return data

         


class EMGDataset(BaseDataset):
    def __init__(self, kwargs):
        super().__init__(**kwargs)

        self.emg_columns = ['channel {}'.format(i) for i in range(16)]
        if self.label_source == 'manus':
            self.label_columns = build_manus_columns()
        elif self.label_source == 'leap':
            self.label_columns = build_leap_columns()
        
        self.prepare_data()

        self.discritize_data() # discritize the data into sequences of length seq_len using torch
        print("data shape: ", self.data.shape)

    
    def prepare_data(self):
        data, annotations, header =  DATA_SOURCES[self.data_source](self.data_path)
        label, _, _ = DATA_SOURCES[self.label_source](self.label_path)
        
        # set the start and end of experiment
        start_time = max(min(data.index), min(label.index))
        end_time = min(max(data.index), max(label.index))

        # select only the data between start and end time
        data = data.loc[start_time:end_time]
        label = label.loc[start_time:end_time]

        # make sure the dataframes are of the same length for the merge
        df = pd.merge_asof(data, label, on='time', direction='nearest')

        assert df.shape[0] == data.shape[0] & df.shape[0] == label.shape[0], 'Dataframes are not of the same length'
        del df

        #reset index to numeric values
        data.reset_index(drop=True, inplace=True)
        label.reset_index(drop=True, inplace=True)

        #save the column names for the label
        self.label_columns = label.columns

        #convert to numpy arrays
        data = data.to_numpy()
        label = label.to_numpy()
        print("data shape original: ", data.shape)

        # normalize the data
        print(f'max before scaling: {np.max(data)}\nmin before scaling: {np.min(data)}')
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        print(f'max after scaling: {np.max(data)}\nmin after scaling: {np.min(data)}')

        # # discritize the data into sequences of length seq_len
        # data = self.unfold(data, self.seq_len)
        # label = self.unfold(label, self.seq_len)
        # print("data shape unfolded: ", data.shape)

        print("filtering data")
        #filter the data
        if self.filter_data:
            data = self._filter_data(data, fs=self.fs)
        
        # convert to tensor
        self.data = torch.tensor(data.copy(), dtype=torch.float32)
        self.label = torch.tensor(label.copy(), dtype=torch.float32)


    #discritize the data into sequences of length seq_len using torch
    def discritize_data(self):
        self.data = self.data.unfold(0, self.seq_len, self.stride).permute(0, 2, 1).unsqueeze(1)
        self.label = self.label.unfold(0, self.seq_len, self.stride).permute(0, 2, 1).unsqueeze(1)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label
    
    @staticmethod
    def _filter_data(data: np.ndarray, fs: float, notch: float=50, low_freq: float=20.0, high_freq: float=55,
                     buff_len: int = 0) -> np.ndarray:
        # Define the notch frequency and quality factor
        notch_freq = 50  # Hz
        Q = 30

        # Calculate the normalized frequency and design the notch filter
        w0 = notch_freq / (fs / 2)
        b_notch, a_notch = iirnotch(w0, Q)

        #calculate the normalized frequencies and design the highpass filter
        cutoff = low_freq / (fs / 2)
        sos = butter(5, cutoff, btype='highpass', output='sos')

        # apply filters using 'filtfilt' to avoid phase shift
        data = sosfiltfilt(sos, data, axis=0, padtype='even')
        data = filtfilt(b_notch, a_notch, data)

        return data 
        
    @staticmethod
    def unfold(data, seq_len):
        '''
        Unfold the data into segments of length seq_len
        Input: data: numpy array of shape (num_samples, num_features)
                seq_len: length of each segment
        Output: segments: numpy array of shape (num_segments, seq_len, num_features)
        '''
        original_length, num_features = data.shape
        num_segments = (original_length - seq_len + 1)

        # Reshape the data to (num_segments, seq_len, num_features)
        segments = np.lib.stride_tricks.sliding_window_view(data, (seq_len, num_features))
        segments = segments.squeeze(1)
        return segments
    
    @staticmethod
    def fold(data):
        '''
        Fold the data into a single array
        Input: segments: numpy array of shape (num_segments, seq_len, num_features)
        Output: segments: numpy array of shape (num_samples, num_features)
        '''
        num_segments, seq_len, num_features = data.shape
        original_length = int(num_segments) + seq_len - 1

        # Reshape the data to (num_segments, seq_len, num_features)
        segments = np.lib.stride_tricks.as_strided(data, shape=(original_length, num_features), strides=(1,1))
        return segments
    
    def plot_data(self, seg_len=800, num_channels=4, channels=None, save_dir=None):
        '''
        Plot EMG data
        '''
        to_plot = self.data.numpy().squeeze()[:,0,:]
        t = np.linspace(0, 1, to_plot[:seg_len,:].shape[0])
        if not channels:
            channels = np.arange(num_channels)

        start = np.random.randint(0, to_plot.shape[0] - seg_len)
        to_plot = to_plot[start:start+seg_len,:]

        # setup figure with subplots
        fig, axs = plt.subplots(num_channels, 1, figsize=(10, 10))
        for i in channels:
            axs[i].plot(t, to_plot[:,i])
            axs[i].set_title(f'Channel {i}')
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'data.png'))
        plt.show()

class TestDataset(EMGDataset):
    def __init__(self, seq_len=150, num_channels=16, filter_data=False):
        super().__init__(data_path='/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/data_2023-10-02 14-59-55-627.edf', 
                      label_path='/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/label_2023-10-02_15-24-12_YH_lab_R.csv',
                      seq_len=seq_len, num_channels=num_channels, filter_data=filter_data
                      )
        # tale only the first 1000 samples
        self.data = self.data[:1000]
        self.label = self.label[:1000]
    
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

    dataset = EMGDataset(kwargs)
    print(dataset.data.shape)
