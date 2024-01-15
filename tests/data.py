from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, stft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import mne
import os
import glob

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

def build_leap_columns(positions=False, rotations=False):
    
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    joints = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    # rotations = ['x', 'y', 'z', 'w']
    # positions = ['x', 'y', 'z']
    leap_columns = []
    if positions:
        for finger in fingers:
            for joint in joints:
                leap_columns.append(f'{finger}_{joint}_position_x')
                leap_columns.append(f'{finger}_{joint}_position_y')
                leap_columns.append(f'{finger}_{joint}_position_z')
    if rotations:
        for finger in fingers:
            for joint in joints:
                # leap_columns.append(f'{finger}_{joint}_rotation_z')
                # leap_columns.append(f'{finger}_{joint}_rotation_x')
                # leap_columns.append(f'{finger}_{joint}_rotation_y')
                leap_columns.append(f'{finger}_{joint}_rotation_w')

    return leap_columns

def read_emg(path, start_time=None, end_time=None, fs: int=250):

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    # get header
    header = raw.info

    if start_time is None:
        start_time = header['meas_date']
        # convert to pd.datetime from datetime.datetime
        start_time = pd.to_datetime(start_time).tz_localize(None)
        # remove 2 hours
        # start_time = start_time - pd.to_timedelta(2, unit='h')
        print(start_time)
    
    #  get annotations
    annotations = raw.annotations
    annotations.onset = start_time + pd.to_timedelta(annotations.onset, unit='s')
    
    # get annotations as df
    to_append = []
    for ind, (i,j) in enumerate(zip(annotations.onset, annotations.description)):
        if 'start_' in j:
            if 'end_' in annotations.description[ind+1] and j.replace('start_', '') == annotations.description[ind+1].replace('end_', ''):
                new_j = j.replace('start_', '')
                #  add 1 sec ofset to onset and append
                offset = pd.to_timedelta(1, unit='s')
                to_append.append([annotations.onset[ind]+offset, annotations.onset[ind+1]+offset, new_j])
    
    #  append rest periods
    for i in range(len(to_append)-1):
        if to_append[i][1] != to_append[i+1][0]:
            to_append.append([to_append[i][1], to_append[i+1][0], f'{i}_rest'])

    ann_df = pd.DataFrame(to_append, columns=['start_time', 'end_time', 'gesture'])
    # sort by start time
    ann_df.sort_values(by='start_time', inplace=True)
    #  if duration is greater than 10 sec, drop
    ann_df = ann_df[ann_df['end_time'] - ann_df['start_time'] < pd.to_timedelta(10, unit='s')]


    emg_df = raw.to_data_frame()
    emg_df['time'] = pd.to_datetime(emg_df['time'], unit='s', origin=start_time)
    emg_df.set_index('time', inplace=True)

    # start data from first annotation
    # start_time = ann_df['start_time'].iloc[0]
    emg_df = emg_df[start_time:]
    
    #  resample emg data to fs Hz
    emg_df = emg_df.resample(f'{int(1000/fs)}ms', origin='start').mean()

    return emg_df, ann_df, header

def read_leap(path, fs=250, positions=False, rotations=True):

    leap_df = pd.read_csv(path, index_col=False)

    # drop null and duplicates
    leap_df.dropna(inplace=True)
    leap_df.drop_duplicates(inplace=True, subset=['time'])

    leap_df['time'] = pd.to_datetime(leap_df['time'])
    leap_df['time'] = leap_df['time'].dt.tz_localize(None)
    leap_df = leap_df.set_index('time')

    # calculate relative position
    for i in leap_df.columns:
        if 'position_x' in i:
            leap_df[i] = leap_df[i] - leap_df['palm_x']
        elif 'position_y' in i:
            leap_df[i] = leap_df[i] - leap_df['palm_y']
        elif 'position_z' in i:
            leap_df[i] = leap_df[i] - leap_df['palm_z']
        else:
            continue
    
    # leap_df = leap_df.resample(f'{int(1000/fs)}ms', origin='start').ffill()
    

    valid_columns = build_leap_columns(positions=positions, rotations=rotations)
    # distal = [i for i in leap_df.columns if "distal" in i.lower()]
    if len(valid_columns) != 0:
        leap_df = leap_df[valid_columns]
        # leap_df = leap_df[distal]
    
    if rotations and len(valid_columns) != 0 and not positions:
        leap_df = leap_df.apply(lambda x: np.rad2deg(x))
        # add offset value of 50 degrees to all angles
        leap_df = leap_df.apply(lambda x: x - 50)
        #  proximal columns
        proximal = [i for i in leap_df.columns if "proximal" in i.lower()]
        leap_df[proximal] = leap_df[proximal].apply(lambda x: x-45)
        #  remove distal columns
        # distal = [i for i in leap_df.columns if "distal" in i.lower()]
        # leap_df.drop(columns=distal, inplace=True)

        # leap_df = leap_df.apply(lambda x: x - 180 if x > 180 else x)
        # mcp = [i for i in leap_df.columns if "metacarpal" in i.lower() and "thumb" not in i.lower()]
        # leap_df[mcp] = leap_df[mcp].apply(lambda x: 0)
    #  convert radians to degrees
    # leap_df = leap_df.apply(np.degrees)
    
    #  normalise the data
    return leap_df, None, None

DATA_SOURCES = {
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
        

        results = {
            'data': [None]*len(edf_files),
            'label': [None]*len(edf_files),
            'gestures': [None]*len(edf_files)
        }

        #  read the data
        self.data, self.label, self.gestures = [], [], []
        for i in range(len(edf_files)):
            print(f'Reading data from {edf_files[i]} and {csv_files[i]}')

            data, label, gestures = self.prepare_data(edf_files[i], csv_files[i], results, i)
            results['data'][i] = data
            results['label'][i] = label
            results['gestures'][i] = gestures
        
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
        #  use torch fold to turn in to sequences
        
        # to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)

        #  convert into shape NxSxC with a sliding window using torch roll
        self.data = self.data.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)
        self.label = self.label.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)

        if self.transform:
            self.data = self.transform(self.data)

        if self.target_transform:
            self.label = self.target_transform(self.label)
            
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
        label, _, _ = DATA_SOURCES['leap'](label_path, rotations=True, positions=False)

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
        #  get the biggest time gap in the label
        time_diff = label.index.to_series().diff().max()
        merged_df = pd.merge_asof(data, label, on='time', direction='forward')
        
        #  assign each data point annotation
        # annotations = annotations[annotations['start_time'] >= start_time]
        # annotations = annotations[annotations['end_time'] <= end_time]
        #  merge the annotations with the data

        # merged_df = pd.merge_asof(merged_df, annotations['gesture'], on='time', direction='forward')

        data = merged_df[data.columns].to_numpy()
        label = merged_df[label.columns].to_numpy()


        # normalize the data
        data = self.normalize_and_filter(data)

        #  remove all rest gestures


        # convert to tensor
        # self.data = torch.tensor(self.data, dtype=torch.float32)
        # self.label = torch.tensor(self.label, dtype=torch.float32)
        return data, label, annotations
    
    def normalize_and_filter(self, data=None):

        # N, C, L = data.shape
        # data_sliced = data.reshape(-1, L)

        # normalize the data
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        print("Filtering data...")
        # filter the data
        if self.filter_data:
            data = self._filter_data(data)

        return data
    
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
    
    

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.gestures[idx]


class ICATransform(object):
    def __init__(self, mixing_matrix):
        self.mixing_matrix = mixing_matrix
        pass

    def __call__(self, sample):
        #  return the sample as well as the ICA transformed sample
        sample = torch.tensor(sample, dtype=torch.float32)
        sample_ica = torch.matmul(sample, self.mixing_matrix)
        return sample, sample_ica
    
def get_data():
    kwargs = {
        'data_path': '../finger_pose_estimation/dataset/FPE/S2/p3',
        'seq_len': 250,
        'num_channels': 16,
        # filter info
        'filter_data': True,
        'fs': 250,
        'notch_freq': 50,
        'Q': 30,
        'low_freq': 20,
        'high_freq': 55,
        'stride': 1,
        'data_source': 'emg',
        'ica': False,
        'transform': None,
        'target_transform': None,
    }
    return EMGLeap(kwargs=kwargs)
if __name__ == '__main__':

    kwargs = {
        'data_path': '../finger_pose_estimation/dataset/FPE/S2/p3',
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
        'data_source': 'emg',
        'ica': False,
        'transform': None,
        'target_transform': None,
    }

    dataset = EMGLeap(kwargs=kwargs)
    print(dataset.data.shape)

