import torch
from torch import nn


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return x + self.block(x)
    
# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2,2)):
        super(EncoderLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,2), stride=(1,1), padding=(2,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=scale_factor),
        )
    def forward(self, x):
        x = self.encoder(x.float())
        return x
    
class Conv3DEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2,2)):
        super(Conv3DEncoderLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,2), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=scale_factor),
        )
    
# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, last=False, scale_factor=(2,2), output_shape=(1000,20)):
        super(DecoderLayer, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3,2), stride=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if last:
            self.decoder.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=False))
        else:
            self.decoder.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.decoder(x)
        return x
    
# Construct a model with 3 conv layers 3 residual blocks and 3 deconv layers using the ResNet architecture
class NeuroPose(nn.Module):
    def __init__(self, in_channels=1, num_residual_blocks=3, output_shape=(1000,20)):
        super(NeuroPose, self).__init__()
        
        encoder_channels = [in_channels, 32, 128, 256]
        scale_factors = [(5,2), (4,2), (2,2)]

        self.encoder = self.make_encoder_layers(channels=encoder_channels, scale_factors=scale_factors)

        # get last number of filters from encoder
        resnet_channels = encoder_channels[-1]
        self.resnet = self.make_resnet_layers(channel=resnet_channels, num_layers=num_residual_blocks)
        
        self.decoder = self.make_decoder_layers(channels=encoder_channels[::-1], scale_factors=scale_factors[::-1], output_shape=output_shape)

    def make_encoder_layers(self, channels = [1, 32, 128, 256], scale_factors = [(5,2), (4,2), (2,2)]):
        # sequence of encoder layers
        layers = []
        for i in range(len(channels)-1):
            layers.append(EncoderLayer(channels[i], channels[i+1], scale_factor=scale_factors[i]))

        return nn.Sequential(*layers)

    def make_decoder_layers(self, channels = [256, 128, 32, 16], scale_factors = [(2,2), (4,2), (5,2)], output_shape=(1000,20)):
        # sequence of decoder layers
        layers = []
        for i in range(len(channels)-2):
            layers.append(DecoderLayer(channels[i], channels[i+1], scale_factor=scale_factors[i]))
        layers.append(DecoderLayer(channels[-2], channels[-1], last=True, output_shape=output_shape))

        return nn.Sequential(*layers)

    def make_resnet_layers(self, channel=256, num_layers=3):
        # sequence of resnet layers
        layers = []
        for i in range(num_layers):
            layers.append(ResidualBlock(channel, channel))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.resnet(x)
        x = self.decoder(x)
        return x
    
    #load from pretrained weights
    def load_pretrained(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

        del pretrained_dict
    
def make_neuropose_model(cfg):
    model = NeuroPose(output_shape=(cfg.DATA.EMG.SEGMENT_LENGTH, cfg.DATA.MANUS.NUM_JOINTS))
    return model

if __name__ == '__main__':
    model = NeuroPose()
    print(model)
    x = torch.randn(1, 1, 1000, 24)
    print(model(x).shape)