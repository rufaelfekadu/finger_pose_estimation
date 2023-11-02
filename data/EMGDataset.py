from torch.utils.data import Dataset
import pandas as pd
import torch

import sys
sys.path.append('../')
from util.data import read_emg, read_manus

# Add data sources here
# TODO: 
# 1. Add ultraleap data source
# 2. Add video data source


DATA_SOURCES = {
    'manus': read_manus,
    'emg': read_emg,
}

class EMGDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None, data_source='emg', label_source='manus', seq_len=1000, num_channels=16):

        self.seq_len = seq_len
        self.num_channels = num_channels
        self.data_path = data_path
        self.label_path = label_path

        self.transform = transform

        self.data_source = data_source # emg or imu
        self.label_source = label_source   # manus, video, or ultraleap 

        self.emg_columns = ['channel {}'.format(i) for i in range(16)]
        self.mauns_columns = ['Pinch_ThumbToIndex','Pinch_ThumbToMiddle', 'Pinch_ThumbToRing',
                        'Pinch_ThumbToPinky', 'Thumb_CMC_Spread', 'Thumb_CMC_Flex', 'Thumb_PIP_Flex', 'Thumb_DIP_Flex',
                        'Index_MCP_Spread', 'Index_MCP_Flex', 'Index_PIP_Flex', 'Index_DIP_Flex', 'Middle_MCP_Spread',
                        'Middle_MCP_Flex', 'Middle_PIP_Flex', 'Middle_DIP_Flex', 'Ring_MCP_Spread', 'Ring_MCP_Flex',
                        'Ring_PIP_Flex', 'Ring_DIP_Flex', 'Pinky_MCP_Spread', 'Pinky_MCP_Flex', 'Pinky_PIP_Flex',
                        'Pinky_DIP_Flex','time']
        
        self.prepare_data()
        self.discritize_data() # discritize the data into sequences of length seq_len using torch

        
    def prepare_data(self):
        data =  DATA_SOURCES[self.data_source](self.data_path)
        label = DATA_SOURCES[self.label_source](self.label_path)
        
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

        # convert to tensor
        self.data = torch.tensor(data.values)
        self.label = torch.tensor(label.values)

    #discritize the data into sequences of length seq_len using torch
    def discritize_data(self):
        self.data = self.data.unfold(0, self.seq_len, 1).permute(0, 2, 1).unsqueeze(1)
        self.label = self.label.unfold(0, self.seq_len, 1).permute(0, 2, 1).unsqueeze(1)

    


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

if __name__ == '__main__':
    data = EMGDataset('/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/ data_2023-10-02 14-59-55-627.edf', 
                      '/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/label_2023-10-02_15-24-12_YH_lab_R.csv')
    print(data.data.shape)
    print(data.label.shape)
