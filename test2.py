import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from threading import Thread
import os
import glob

from util.data import *
from config import cfg
from data import BaseDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_SOURCES = {
    'emg': read_emg,
    'leap': read_leap,
}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.generate_encoding(d_model, max_len)

    def generate_encoding(self, d_model, max_len):
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        return encoding

    def forward(self, x):
        seq_length = x.size(1)
        return self.encoding[:, :seq_length].to(x.device)

import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        output = self.encoder(x)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim)
        self.decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)

    def generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, encoder_output):
        x = self.embedding(x)
        x = x + self.pos_decoder(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)

        # Generate causal mask
        tgt_mask = self.generate_mask(x.size(0)).to(x.device)

        output = self.decoder(x, encoder_output, tgt_mask=tgt_mask)
        return output


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads)
        self.decoder = TransformerDecoder(output_dim, hidden_dim, num_layers, num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

    def forward(self, x, y):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(y, encoder_output)
        output = self.fc(decoder_output)
        # reshape back to batch_size x seq_len x num_channels
        output = output.permute(1, 0, 2)
        return output

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


# Assuming you have your training data in train_loader and validation data in val_loader
# Make sure to replace this with your actual DataLoader instances

# Example DataLoader creation (replace this with your actual data loading code)
# train_data = ...  # Your training data
# val_data = ...    # Your validation data
# train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)

# Hyperparameters
input_dim = 16    # Replace with the actual size of your input vocabulary
output_dim = 20     # Assuming 3 for x, y, z coordinates in pose estimation
hidden_dim = 256
num_layers = 4
num_heads = 8
lr = 0.001
batch_size = 32
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kwargs = {
        'data_path': 'dataset/FPE/S1/p3',
        'seq_len':200,
        'num_channels':16,
        'stride':5,
        'filter_data':True,
        'fs':cfg.DATA.EMG.SAMPLING_RATE,
        'Q':cfg.DATA.EMG.Q,
        'low_freq':cfg.DATA.EMG.LOW_FREQ,
        'high_freq':cfg.DATA.EMG.HIGH_FREQ,
        'notch_freq':cfg.DATA.EMG.NOTCH_FREQ,
        'ica': False,
        'transform': None,
        'target_transform': None,
    }
dataset = EMGLeap(kwargs=kwargs)
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads)
model = model.to(device)
# Loss and optimizer (using Mean Absolute Error for regression)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
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
