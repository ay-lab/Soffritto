"""
Trains a Soffritto model and saves the trained model to file.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data_intra_cell_line_train
import argparse
import numpy as np
from torch.nn.functional import kl_div
import os
import json
import random

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--features', help='Path to features dataset')
parser.add_argument('--labels', help='Path to 16-fraction RT data')
parser.add_argument('--model_path', help='Path for the trained model (saved as a .pth file)')
parser.add_argument('--train_chromosomes', nargs='+', help='Training chromosomes as space-separated list of integers')
parser.add_argument('--val_chromosomes', help='Validation chromosome as integer')
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_hiddens', type=int, help="Hidden size of LSTM module")
parser.add_argument('--num_layers', type=int, help="Number of LSTM layers")
parser.add_argument('--weight_decay', type=float, help="L2 regularization coefficient")
parser.add_argument('--hyperparameter_file', help="Path to json file containing num_hiddens and num_layers for trained model")
args = parser.parse_args()

X_train, y_train, X_val, y_val = load_data_intra_cell_line_train(args.features, 
                                             args.labels, 
                                             args.train_chromosomes, 
                                             args.val_chromosomes)

# Creates directory for saved model if it doesn't already exist
os.makedirs(f"{os.path.dirname(args.model_path)}", exist_ok=True)

class Soffritto(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Soffritto, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Recurrent layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(2*hidden_size, output_size)
        
        # Softmax layer
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Hidden state
        self.hidden = None
        
    def forward(self, x):
        # If hidden state is None, initialize it
        if self.hidden is None:
            self.hidden = self.init_hidden(x.device)
        
        # Forward propagate LSTM
        out, self.hidden = self.lstm(x, self.hidden)
        
        # Detach hidden state to prevent backprop through entire history
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        # Apply log softmax for KLDivLoss
        out = self.log_softmax(out)
        
        return out
    
    def init_hidden(self, device):
        # Initialize hidden and cell states.
        return (torch.zeros(2 * self.num_layers, self.hidden_size, device=device),
                torch.zeros(2 * self.num_layers, self.hidden_size, device=device))

    def reset_hidden(self, device):
        # Manually reset hidden state
        self.hidden = self.init_hidden(device)

# Define hyperparameters
input_size = X_train[next(iter(X_train))].size(-1)
hidden_size = args.num_hiddens
num_layers = args.num_layers
output_size = 16  # 16 fractions
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = Soffritto(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss function and optimizer
criterion = nn.KLDivLoss(reduction = 'batchmean')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay) # considered more stable and better for L2 regularization than Adam

# Create dataloaders. Data is loaded by each chromosome at a time
dataloaders = {}
for chrom in X_train.keys():
    feature_tensor = X_train[chrom] 
    label_tensor = y_train[chrom]
    dataset = TensorDataset(feature_tensor, label_tensor)
    dataloaders[chrom] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Training loop
best_val_loss = float('inf')
validation_losses = []
if torch.cuda.is_available():
    print("Training on gpu")
else:
    print("Training on cpu")
for epoch in range(num_epochs):
    chrom_order = list(X_train.keys())  # Shuffle chromosome order
    random.shuffle(chrom_order)
    for chrom in chrom_order:
        model.reset_hidden(device=device)
        model.train()
        for batch in dataloaders[chrom]:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        val_loss = criterion(val_outputs, y_val.to(device).float())
    
    # Saving best model
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        # Save best model
        torch.save(model.state_dict(), args.model_path)

# Save hyperparameter configuration file
hyper_dict = dict()
hyper_dict['hidden_size'] = hidden_size
hyper_dict['num_layers'] = num_layers
with open(args.hyperparameter_file, 'w') as json_file:
    json.dump(hyper_dict, json_file, indent=4)