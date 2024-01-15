import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
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
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
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
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout=0.1)
        self.decoder = TransformerDecoder(output_dim, hidden_dim, num_layers, num_heads, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(y, encoder_output)
        output = self.fc(decoder_output)
        output = self.dropout(output)
        # reshape back to batch_size x seq_len x num_channels
        output = output.permute(1, 0, 2)

        return output
    
    def inference(self, x, y):

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(y, encoder_output)
        output = self.fc(decoder_output)
        # reshape back to batch_size x seq_len x num_channels
        output = output.permute(1, 0, 2)
        return output

def get_model():
    input_dim = 16
    output_dim = 20
    hidden_dim = 512
    num_layers = 2
    num_heads = 8

    model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads)
    return model
if __name__ == "__main__":
    # Test
    input_dim = 16
    output_dim = 20
    hidden_dim = 512
    num_layers = 2
    num_heads = 8

    model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads)
    x = torch.randn(32, 10, input_dim)
    y = torch.randn(32, 9, output_dim)

    output = model(x, y)
    print(output.shape)