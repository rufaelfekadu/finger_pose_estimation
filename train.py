import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import make_dataset, make_dataloader
from config import cfg
from models import NeuroPose

# Define hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Load the dataset
dataset = make_dataset(cfg)

# make dataloader
train_loader = make_dataloader(cfg, dataset)

# Initialize the model
model = NeuroPose()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# Save the model
torch.save(model.state_dict(), 'finger_pose_model.pth')

def train(model, train_loader, criterion, optimizer, num_epochs):

    '''
    Train the model
    '''
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

def test(model, test_loader, criterion):
    
    '''
    Test the model
    '''
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            outputs = model(data)
            loss = criterion(outputs, labels)
            print('Loss: {:.4f}'.format(loss.item()))

