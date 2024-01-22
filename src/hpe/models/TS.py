from torch import nn
import torch

#  model for time space consistancy contrastive learining Framework
class TS(nn.modules):
    def __init__(self, cfg):
        super(TS, self).__init__()
        self.cfg = cfg
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 128, 0.5),
            num_layers=4
        )

        self.ica_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 128, 0.5),
            num_layers=4
        )

        self.projection_t = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.projection_ica = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x, x_ica):
        x = self.time_encoder(x)
        x_ica = self.ica_encoder(x_ica)

        h_t = x.flatten(start_dim=1)
        h_ica = x_ica.flatten(start_dim=1)

        z_t = self.projection_t(x)
        z_ica = self.projection_ica(x_ica)

        return h_t, h_ica, z_t, z_ica

