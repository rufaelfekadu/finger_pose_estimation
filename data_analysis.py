import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from data import make_dataset
from config import cfg
from models import NeuroPose

from torch import nn

def plot_loss_from_log(file_path):

    # Open the input file for reading
    with open(file_path, 'r') as f:
        # Read the lines of the file into a list
        lines = f.readlines()

    # Remove every other line from the list
    lines = lines[1::2]

    # remove every line except the last 50
    lines = lines[-50:]

    # Loop through the remaining lines and extract the train and val loss values
    train_losses = []
    val_losses = []
    for line in lines:
        # Split the line into its components
        components = line.strip().split(':')
        # Extract the train and val loss values
        train_loss = float(components[4][:-9])
        val_loss = float(components[5])
        # Append the values to the appropriate lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)


    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the train and val loss curves
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')

    # Set the title and axes labels
    ax.set_title('Train and Val Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    # Save the plot
    fig.savefig('loss_curves.png')

# plot datapoints from dataset
dataset = make_dataset(cfg)

#get model
model = NeuroPose()
model.load_pretrained(cfg.SOLVER.PRETRAINED_PATH)

# setup figure with ten subplots
fig, axs = plt.subplots(3, 1, figsize=(10,10))
mse = nn.MSELoss()

for i,j in enumerate(range(10,13)):
    data, label = dataset[j]
    label = label.squeeze(0)

    # plot ground truth
    axs[i].plot(label[:,j].detach().numpy(), label='ground truth')

    #run infernece
    output = model(data.unsqueeze(0))
    output = output.squeeze(0).squeeze(0)
    # plot prediction
    axs[i].plot(output[:,j].detach().numpy(), label='prediction')

    # # add legend
    axs[i].legend()
    mse_loss = mse(output, label)
    print(f"mse loss: {mse_loss}")

# show plot
plt.show()

# save plot
# fig.savefig('predictions.png')


