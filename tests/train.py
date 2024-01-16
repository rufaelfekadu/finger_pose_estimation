from data import get_data
from model import get_model
from transformer import make_transformer_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import time
import argparse
from sklearn.model_selection import train_test_split
import os

def train_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device):

    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        # Forward pass
        output = model(input_seq)  # Exclude the last pose from the target
        # Compute the loss
        loss = criterion(output, target_seq[:,-1,:])  # Exclude the first pose from the target
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")


    return total_loss / len(train_loader)

def evaluate(epoch, model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            # Forward pass
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq)

            # Compute the loss
            loss = criterion(output, target_seq[:, -1, :])
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {average_val_loss:.4f}")


    return average_val_loss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    dataset = get_data()
    tr_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=args.seed)
    train_dataset = Subset(dataset, tr_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_transformer_model()
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(epoch, model, train_loader, criterion, optimizer,scheduler, device)
        val_loss = evaluate(epoch, model, val_loader, criterion, device)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | valid loss {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            #  print some predictions
            plot_sample(model, val_loader, device, save_path=os.path.join(args.save, f'./sample_{epoch}.gif'))

    # Save the best model
    torch.save(best_model.state_dict(), args.save)

def plot_sample(model, val_loader, device, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    # Assuming data is your 4x4x150 numpy array
    data_iter = iter(val_loader)
    data, label = next(data_iter)
    B, S, C = data.shape
    # prediction
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        pred = model(data).cpu().numpy()
    #  copy data to cpu
    fig, ax = plt.subplots(1,2, figsize=(10, 7))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    data = data.cpu().numpy()
    # Initial frame
    im = ax[0].imshow(data[0, :50, :], animated=True, cmap='hot')
    im_label, = ax[1].plot(label[0, -1, :], animated=True, )
    im_pred, = ax[1].plot(pred[0, :], animated=True,)
    ax[1].legend(['Prediction', 'Ground truth'])

    def updatefig(i):
        # Update the image for frame i
        im.set_array(data[i, :50, :].T)
        im_pred.set_ydata(pred[i, :])
        im_label.set_ydata(label[i, -1,:])
        ax[0].set_title('Original EMG data')
        ax[1].set_title('Angle Values')
        return im, im_pred, im_label

    ani = animation.FuncAnimation(fig, updatefig, frames=range(B), blit=True, interval=200, repeat=False)
    ani.save(save_path, writer='imagemagick', fps=5)
    # Display the animation
    HTML(ani.to_jshtml())

def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer for Motion Prediction')
    parser.add_argument('--data', type=str, default='./data/processed',
                        help='location of the data corpus')
    parser.add_argument('--save', type=str, default='./model/model.pt',
                        help='path to save the final model')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')

    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    train(args)

if __name__ == '__main__':
    main()