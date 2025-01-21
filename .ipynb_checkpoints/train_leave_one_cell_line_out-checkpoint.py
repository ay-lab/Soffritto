import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data_cross_cell_line
import argparse
import numpy as np
from torch.nn.functional import kl_div
import os

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--features', nargs='+', help='Paths to features datasets (space-separated)')
parser.add_argument('--labels', nargs='+', help='Paths to 16-fraction RT data (space-separated and in the some order as features files)')
parser.add_argument('--model_path', help='Path for the trained model (saved as a .pth file)')
parser.add_argument('--train_chromosomes', nargs='+', help='Training chromosomes as space-separated list of integers')
parser.add_argument('--val_chromosomes', help='Validation chromosome as integer')
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_hiddens', type=int, help="Hidden size of LSTM module")
parser.add_argument('--num_layers', type=int, help="Number of LSTM layers")
parser.add_argument('--weight_decay', type=float, help="L2 regularization coefficient")
args = parser.parse_args()

X_train, y_train, X_val, y_val = load_data_cross_cell_line(args.features, 
                                             args.labels, 
                                             args.train_chromosomes, 
                                             args.val_chromosomes)

# Creates directory for saved model if it doesn't already exist
os.makedirs(f"{os.path.dirname(args.model_path)}", exist_ok=True)

# Model definition
class Soffritto(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Soffritto, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(2*hidden_size, output_size)
        
        # LogSoftmax layer
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        # Initializes hidden state with zeros
        h0 = torch.zeros(2*self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(2*self.num_layers, self.hidden_size).to(x.device)
        
        # LSTM forward propagation
        out, _ = self.lstm(x, (h0, c0)) 
        
        # Applies fully connected layer to hidden state of last step
        out = self.fc(out)
        
        # Applies log softmax for KLDivLoss
        out = self.log_softmax(out)
        
        return out

# Define hyperparameters
input_size = X_train.size(1)
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

# Convert data to DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

test_data = TensorDataset(X_val, y_val)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Training loop
best_val_loss = float('inf')
validation_losses = []
if torch.cuda.is_available():
    print("Training on gpu")
else:
    print("Training on cpu")
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
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