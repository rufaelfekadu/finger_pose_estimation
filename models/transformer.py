import torch
import torch.nn as nn
import time

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


class TransformerModel(nn.Module):
    def __init__(self, input_size, seq_length, num_channels, output_size):
        super(TransformerModel, self).__init__()

        self.d_model = 128
        self.nhead = 4
        self.num_layers = 4
        self.dropout = 0.1

        self.embedding = nn.Linear(input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=seq_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_model, self.dropout),
            num_layers=self.num_layers
        )
        self.decoder = nn.Linear(self.d_model * seq_length, output_size)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.embedding(x)
        x = (x + self.pos_encoder(x)).permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x.flatten(start_dim=1)
        x = self.decoder(x)
        x.unsqueeze(1)
        return x
    
    def load_pretrained(self, path):
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        print('Pretrained model loaded')

        del pretrained_dict

def make_transformer_model(cfg):
    model = TransformerModel(input_size=cfg.DATA.EMG.NUM_CHANNELS, 
                             seq_length=cfg.DATA.SEGMENT_LENGTH, 
                             num_channels=cfg.DATA.EMG.NUM_CHANNELS, 
                             output_size=cfg.DATA.MANUS.NUM_JOINTS)
    return model
if __name__ == '__main__':
    # Example usage
    N = 100  # Number of training examples
    S = 500   # Sequence length
    C = 16   # Number of channels
    output_dim = 20

    # Generate random input data
    input_data = torch.randn(N,1, S, C)

    # Create the Transformer model
    model = TransformerModel(input_size=C, seq_length=S, num_channels=C, output_size=output_dim)

    start = time.time()
    # Pass the input through the model
    output = model(input_data)
    print(f"Time taken: {time.time() - start}")

    print(output.shape)  # Check the output shape
