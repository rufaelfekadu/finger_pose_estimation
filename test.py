import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TransformerModelWithPositionalEncoding(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, hidden_dim=128, num_layers=4):
        super(TransformerModelWithPositionalEncoding, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.output_embedding = nn.Linear(output_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, emg_signal, target):
        # Input embedding and positional encoding
        embedded = self.embedding(emg_signal.transpose(0, 1))  # Transpose for correct shape
        pos_enc = self.positional_encoding(embedded)

        # Add positional encoding to the input
        embedded_with_pos = embedded + pos_enc

        # Transformer Encoder
        encoded = self.encoder(embedded_with_pos)

        # Transformer Decoder
        embedded_target = self.output_embedding(target.transpose(0, 1))  # Transpose for correct shape
        pos_enc_target = self.positional_encoding(embedded_target)

        # Add positional encoding to the target input
        embedded_target_with_pos = embedded_target + pos_enc_target

        decoded = self.decoder(embedded_target_with_pos, encoded)

        # Output layer
        output = self.fc(decoded)

        return output



import torch
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset for the test case
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, emg_channels, keypoint_positions):
        self.num_samples = num_samples
        self.emg_channels = emg_channels
        self.keypoint_positions = keypoint_positions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic EMG signal data and key point positions
        emg_signal = torch.randn(self.emg_channels, 100)  # Example: 100 timesteps for each channel
        keypoint_position = torch.randn(21, 100)  # Example: 100 timesteps for each key point

        return emg_signal, keypoint_position


model = TransformerModel(input_dim=16, output_dim=21)

# Create a synthetic test dataset with 10 samples
test_dataset = SyntheticDataset(num_samples=10, emg_channels=16, keypoint_positions=21)

# Create a DataLoader for the test dataset
batch_size = 4
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the test DataLoader to get a batch of data
for batch_idx, (emg_batch, keypoint_batch) in enumerate(test_loader):
    # Pass the batch through the model for testing
    # Assuming you've already instantiated and trained your TransformerModel (model)
    output = model(emg_batch, keypoint_batch)
    
    # Perform any necessary processing or analysis with the output
    # For a test case, you might print the output shape or inspect it further
    print(f"Output shape for batch {batch_idx}: {output.shape}")
    
    # Break after the first batch for demonstration purposes
    break
