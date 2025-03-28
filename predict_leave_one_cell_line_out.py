import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data_leave_one_cell_line_out_prediction
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--train_features_files', nargs='+', help='Space-separated list of feature file paths for training')
parser.add_argument('--test_features_file', help='Path to left out cell line features file')
parser.add_argument('--test_labels_file', help='Path to left out cell line 16-fraction RT data')
parser.add_argument('--model_path', help='Path to trained model')
parser.add_argument('--pred_file', help="Path to predicted 16-fraction RT data")
parser.add_argument('--train_chromosomes', nargs='+', help='Training chromosomes as space-separated list of integers')
parser.add_argument('--test_chromosomes', help='Test chromosome as integer')
parser.add_argument('--hyperparameter_file', help="Path to json file containing num_hiddens and num_layers for trained model")
args = parser.parse_args()

X_test, y_test = load_data_leave_one_cell_line_out_prediction(args.train_features_files, args.test_features_file, args.test_labels_file, args.train_chromosomes, args.test_chromosomes)

# Creates results directory if it doesn't already exist
os.makedirs(f"{os.path.dirname(args.pred_file)}", exist_ok=True)

# Model definition
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
        """Initialize hidden and cell states."""
        return (torch.zeros(2 * self.num_layers, self.hidden_size, device=device),
                torch.zeros(2 * self.num_layers, self.hidden_size, device=device))

    def reset_hidden(self, device):
        """Manually reset hidden state (e.g., at epoch start or chromosome change)."""
        self.hidden = self.init_hidden(device)
    
# Define model parameters
with open(args.hyperparameter_file, "r") as json_file:
    hyperparameters = json.load(json_file)
input_size = X_test.size(1)
hidden_size = hyperparameters['hidden_size']
num_layers = hyperparameters['num_layers']
output_size = 16 # 16 fractions

# Load trained model and predict on test data
model = Soffritto(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
model.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
with torch.no_grad():
    test_outputs = model(X_test)
    
# Converts prediction to probabilities by taking the exponential (log was taken for KL loss)
pred = torch.exp(test_outputs).cpu().detach().numpy()
np.save(args.pred_file, pred)