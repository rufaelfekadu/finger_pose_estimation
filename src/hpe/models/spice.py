import torch
import torch.nn as nn
import torchvision.models as models

class ContrastiveModel(nn.Module):
    def __init__(self, num_classes):
        super(ContrastiveModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove the last fully connected layer

    def forward(self, x):
        # x.shape = (batch_size, 1, 4, 4)
        features_1 = self.backbone(x[0])
        features_2 = self.backbone(x[1])
        embeddings_1 = self.projection_head(features_1)
        embeddings_2 = self.projection_head(features_2)

        return features_1, features_2, embeddings_1, embeddings_2


if __name__ == "__main__":
    model = ContrastiveModel(num_classes=128)
    x = torch.randn(2, 1, 4, 4)
    features_1, features_2, embeddings_1, embeddings_2 = model(x)
    print(features_1.shape)
    print(features_2.shape)
    print(embeddings_1.shape)
    print(embeddings_2.shape)