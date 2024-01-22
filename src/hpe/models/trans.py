import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        output = self.encoder(x)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)

    def forward(self, x, encoder_output):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        output = self.decoder(x, encoder_output)
        return output

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
        return output

def make_transformer(cfg):
    return Transformer(
        input_dim = cfg.DATA.EMG.NUM_CHANNELS,
        output_dim = len(cfg.DATA.LABEL_COLUMNS),
        hidden_dim = cfg.MODEL.HIDDEN_DIM,
        num_layers = cfg.MODEL.NUM_LAYERS,
        num_heads = cfg.MODEL.NUM_HEADS,
    )
if __name__ == '__main__':
    
    input_dim = 16
    output_dim = 20
    hidden_dim = 128
    num_layers = 3
    num_heads = 4
    batch_size = 32
    seq_len = 200
    x = torch.randint(0, input_dim, (batch_size, seq_len))
    y = torch.randint(0, output_dim, (batch_size, seq_len))
    model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads)
    output = model(x, y)
