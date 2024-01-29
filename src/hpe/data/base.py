from torch.utils.data import Dataset
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, stft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.decomposition import FastICA
import os

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        self.joints = ['CMC', 'MCP', 'PIP', 'DIP']
        self.movements = ['Flex', 'Spread']
        self.ROTATIONS = ['x', 'y', 'z', 'w']

        # unpack kwargs
        self.data_path = kwargs['data_path']
        self.seq_len = kwargs['seq_len']
        self.stride = kwargs['stride']
        self.num_channels = kwargs['num_channels']

        # filter info
        self.filter_data = kwargs['filter_data']
        self.fs = kwargs['fs']
        self.notch_freq = kwargs['notch_freq']
        self.Q = kwargs['Q']
        self.low_freq = kwargs['low_freq']
        self.high_freq = kwargs['high_freq']
        self.ica = kwargs['ica']
        
        self.transform = kwargs['transform']
        self.target_transform = kwargs['target_transform']
        self.visualize = kwargs['visualize']

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