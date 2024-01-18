import os
import glob
from threading import Thread
import numpy as np
import torch
import pandas as pd
import sys 
from sklearn.preprocessing import StandardScaler

from config import cfg
from data import BaseDataset
from util.data import read_emg, read_leap, create_windowed_dataset
# Assuming you have the necessary functions (DATA_SOURCES, create_windowed_dataset, normalize_and_filter, etc.)

DATA_SOURCES = {
    'emg': read_emg,
    'leap': read_leap,
}

class EMGLeap(BaseDataset):
    def __init__(self, kwargs):
        super().__init__(**kwargs)

        # read the data
        edf_files, csv_files = self.read_dirs()

        if len(edf_files) == 0:
            raise ValueError(f'No edf files found in {self.data_path}')
        if len(csv_files) == 0:
            raise ValueError(f'No csv files found in {self.data_path}')

        threads = [None] * len(edf_files)
        results = {
            'data': [None] * len(edf_files),
            'label': [None] * len(edf_files),
            'gestures': [None] * len(edf_files)
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
                all_files += [f for f in glob.glob(os.path.join(path, '**/*'), recursive=True) if
                              os.path.splitext(f)[1] in ['.edf', '.csv']]

        # # Traverse through all the directories and read the data
        # all_files = [f for f in glob.glob(os.path.join(self.data_path, '**/*'), recursive=True) if os.path.splitext(f)[1] in ['.edf', '.csv']]
        # Separate .edf and .csv files

        edf_files = sorted([file for file in all_files if file.endswith('.edf')])
        csv_files = sorted([file for file in all_files if file.endswith('.csv')])

        return edf_files, csv_files

    def print_dataset_specs(self):
        print("data shape: ", self.data.shape)

    def prepare_data(self, data_path, label_path, results={}, index=0):
        data, annotations, header = DATA_SOURCES['emg'](data_path)
        label, _, _ = DATA_SOURCES['leap'](label_path, rotations=True, positions=False)

        if index == 0:
            # save the column names for the label
            self.label_columns = list(label.columns)
            self.data_columns = list(data.columns)

        # set the start and end of experiment
        start_time = max(min(data.index), min(label.index))
        end_time = min(max(data.index), max(label.index))

        # select only the data between start and end time
        data = data.loc[start_time:end_time]
        label = label.loc[start_time:end_time]

        self.label_columns = list(label.columns)
        # Merge the two DataFrames based on the 'time' column
        merged_df = pd.merge_asof(data, label, on='time', direction='forward')
        data = merged_df[data.columns].to_numpy()
        label = merged_df[label.columns].to_numpy()

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        #  convert into shape NxSxC with a sliding window using torch roll
        data = data.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)
        label = label.unfold(0, self.seq_len, self.stride).permute(0, 2, 1)

        # data, label, gestures = create_windowed_dataset(merged_df, annotations, self.seq_len, self.stride)
        #  convert into shape NxSxC with a sliding window


        # normalize the data
        data = self.normalize_and_filter(data)

        results['data'][index] = data
        results['label'][index] = label
    
    
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
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)
    

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming you have your training data in train_loader and validation data in val_loader
# Make sure to replace this with your actual DataLoader instances

# Example DataLoader creation (replace this with your actual data loading code)
# train_data = ...  # Your training data
# val_data = ...    # Your validation data
# train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)

# Hyperparameters
input_dim = 100    # Replace with the actual size of your input vocabulary
output_dim = 3     # Assuming 3 for x, y, z coordinates in pose estimation
hidden_dim = 256
num_layers = 3
num_heads = 8
lr = 0.001
batch_size = 32
epochs = 10

class TransformerEncoder(nn.Module):
    # ... (unchanged)

class TransformerDecoder(nn.Module):
    # ... (unchanged)

class PositionalEncoding(nn.Module):
    # ... (unchanged)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        self.transformer_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers)
        
        self.decoder = TransformerDecoder(output_dim, hidden_dim, num_layers, num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, y):
        x = self.embedding(x)
        x = x + self.pos_encoder(x)
        
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        encoder_output = self.transformer(x)
        
        y = y.permute(1, 0, 2)  # (seq_len, batch_size, output_dim)
        decoder_output = self.decoder(y, encoder_output)
        decoder_output = decoder_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        
        output = self.fc(decoder_output)
        
        return output




if __name__ == "__main__":

    kwargs = {
                'data_path': 'dataset/FPE/S1/p3',
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
                'transform': None,
                'target_transform': None,
            }
    dataset= EMGLeap(kwargs)


    # Instantiate the model
    model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads)

    # Loss and optimizer (using Mean Absolute Error for regression)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output = model(input_seq, target_seq[:, :-1, :])  # Exclude the last pose from the target

            # Compute the loss
            loss = criterion(output, target_seq[:, 1:, :])  # Exclude the first pose from the target
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {average_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                # Forward pass
                output = model(input_seq, target_seq[:, :-1, :])

                # Compute the loss
                loss = criterion(output, target_seq[:, 1:, :])
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {average_val_loss:.4f}")

    # After training, you can save the model if needed
    torch.save(model.state_dict(), 'transformer_model.pth')

    print(dataset.data.shape)
    print(dataset.label.shape)