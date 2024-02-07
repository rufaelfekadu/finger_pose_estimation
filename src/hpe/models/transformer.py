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

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class BoundedActivation(nn.Module):
    def __init__(self, a_values, b_values, label_dim):
        super(BoundedActivation, self).__init__()
        assert len(a_values) == len(b_values) == label_dim, "Length of a_values and b_values must be equal and equal to label_dim"
        self.a_values = nn.Parameter(torch.Tensor(a_values).view(1, -1))
        self.b_values = nn.Parameter(torch.Tensor(b_values).view(1, -1))
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(x) * (self.b_values - self.a_values) + self.a_values
    
class TransformerModel(nn.Module):

    def __init__(self, input_size, seq_length, output_size, stage='pretrain'):
        super(TransformerModel, self).__init__()

        self.d_model = 128
        self.nhead = 2
        self.num_layers = 2
        self.dropout = 0.1

        self.embedding = nn.Linear(input_size, self.d_model, bias=False)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=seq_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_model, self.dropout, batch_first=True),
            num_layers=self.num_layers,
        )

        self.decoder = nn.Linear(self.d_model*seq_length, self.d_model*4)
        
        self.mlp_head = MLP(self.d_model*4, output_size)
        a = [0, -15, 0, -15, 0, -15, 0, 0, -15, 0, 0, -15, 0, 0, -15, 0]
        b = [90, 15 , 90, 15, 90, 15, 110, 90, 15, 110, 90, 15, 110, 90, 15, 110]
        # self.bact = BoundedActivation(a_values=a, b_values=b, label_dim=output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        x = self.embedding(x)
        x = (x + self.pos_encoder(x))
        x = self.transformer_encoder(x)
        # x = x.permute(1, 0, 2)
        x = self.decoder(x.flatten(start_dim=1))
        x = self.mlp_head(x)
        x.unsqueeze(1)
        # x = self.bact(x)
        return x
    
    def load_pretrained(self, path):

        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        print('Pretrained model loaded')

        del pretrained_dict

def make_transformer(cfg):

    return TransformerModel(input_size=cfg.DATA.EMG.NUM_CHANNELS, 
                             seq_length=cfg.DATA.SEGMENT_LENGTH, 
                             output_size=len(cfg.DATA.LABEL_COLUMNS),
                             stage=cfg.STAGE)

if __name__ == '__main__':
    # Example usage
    N = 32  # Number of training examples
    S = 500   # Sequence length
    C = 16   # Number of channels
    output_dim = 10

    # Generate random input data
    input_data = torch.randn(N, S, C)
    label = torch.randint(0, 10, (N, output_dim))

    # Create the Transformer model
    model = TransformerModel(input_size=C, seq_length=S, output_size=output_dim)
    soft_max = nn.Softmax(dim=1)

    start = time.time()
    # Pass the input through the model
    output = model(input_data)
    # output = soft_max(output)
    pred = output.argmax(dim=1)
    print(f"Time taken: {time.time() - start}")

    print(output.shape)  # Check the output shape
