from .data import get_data
from .model import get_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import time
import argparse
from sklearn.model_selection import train_test_split
def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        # Forward pass
        output = model(input_seq, target_seq[:, :-1, :])  # Exclude the last pose from the target
        # Compute the loss
        loss = criterion(output, target_seq[:, 1:, :])  # Exclude the first pose from the target
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")


    return total_loss / len(train_loader)

def evaluate(epoch, model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            # Forward pass
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq, target_seq[:, :-1, :])

            # Compute the loss
            loss = criterion(output, target_seq[:, 1:, :])
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {average_val_loss:.4f}")


    return total_loss / len(val_loader)

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

    model = get_model()
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

    # Save the best model
    torch.save(best_model.state_dict(), args.save)

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
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--max_len', type=int, default=100,
                        help='max sequence length')
    parser.add_argument('--input_dim', type=int, default=16,
                        help='input dimension')
    parser.add_argument('--output_dim', type=int, default=20,
                        help='output dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden dimension')
    parser.add_argument('--n_classes', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--max_len', type=int, default=100,
                        help='max sequence length')
    parser.add_argument('--input_dim', type=int, default=16,
                        help='input dimension')
    
    args = parser.parse_args()
    train(args)