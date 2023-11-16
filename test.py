import torch
from models import NeuroPose
from data import make_dataset, make_dataloader, read_saved_dataset
from torchvision import transforms
from config import cfg

# Load the trained model
model = NeuroPose()
model.load_pretrained(cfg.MODEL.PRETRAINED_PATH)

# Set the model to evaluation mode
model.eval()

# Load the test dataset
dataset = read_saved_dataset(cfg)
# Define the transformation to be applied to the input images


# Iterate over the test dataset and make predictions
with torch.no_grad():
    for data, labels in dataset['val']:

        # Make predictions using the trained model
        outputs = model(data)
        
