import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_data_intra_cell_line_train(features_file, labels_file, train_chromosomes, test_chromosomes):
    X = np.load(features_file)
    y = np.load(labels_file)
    
    X_train_dict = {}
    y_train_dict = {}
    X_train_arrays = [X[chrom] for chrom in train_chromosomes]

    X_train = np.concatenate(X_train_arrays, axis=0)
    X_test = X[f"{test_chromosomes}"]
    y_test = y[f"{test_chromosomes}"]
    print("Raw data is loaded.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data is scaled.")
    
    for chrom in train_chromosomes:
        X_train_dict[chrom] = scaler.transform(X[chrom])
        y_train_dict[chrom] = y[chrom]

    # Convert data to PyTorch tensors
    for key in X_train_dict:
        X_train_dict[key] = torch.tensor(X_train_dict[key], dtype=torch.float32)
        y_train_dict[key] = torch.tensor(y_train_dict[key], dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("Data is converted to PyTorch tensors.")
    
    return X_train_dict, y_train_dict, X_test, y_test

def load_data_leave_one_cell_line_out_prediction(train_features_file, test_features_file, test_labels_file, train_chromosomes, test_chromosomes):
    chroms_not_in_mouse = ["20", "21", "22"]
    X_test_cl = np.load(test_features_file)
    y = np.load(test_labels_file)
    X_train_arrays = []

    for features_file in train_features_file:
        X = np.load(features_file)
        
        if ('mESC' in features_file) or ('mNPC' in features_file):
            X_train_arrays.extend([X[chrom] for chrom in train_chromosomes if chrom not in chroms_not_in_mouse])

        else:
            X_train_arrays.extend([X[chrom] for chrom in train_chromosomes])
    
    X_train = np.concatenate(X_train_arrays, axis=0)
    X_test = X_test_cl[f"{test_chromosomes}"]
    y_test = y[f"{test_chromosomes}"]
    print("Raw data is loaded.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data is scaled.")

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("Data is converted to PyTorch tensors.")
    
    return X_test, y_test

def load_data_prediction(train_features_file, test_features_file, test_labels_file, train_chromosomes, test_chromosomes):
    X = np.load(train_features_file)
    X_test_cl = np.load(test_features_file)
    y = np.load(test_labels_file)
    
    X_train_arrays = [X[chrom] for chrom in train_chromosomes]

    X_train = np.concatenate(X_train_arrays, axis=0)
    X_test = X_test_cl[f"{test_chromosomes}"]
    y_test = y[f"{test_chromosomes}"]
    print("Raw data is loaded.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data is scaled.")

    # Convert data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("Data is converted to PyTorch tensors.")
    
    return X_test, y_test

def load_data_leave_one_cell_line_out_train_val_dict(features_files, labels_files, train_chromosomes, test_chromosomes):
    chroms_not_in_mouse = ["20", "21", "22"]
    
    X_train_list = []
    X_train_dict = {}
    y_train_arrays = {}

    # Load and store data from all files
    for features_file, labels_file in zip(features_files, labels_files):
        cell_line = features_file.split('/')[-1].split('_')[0]  # Extract cell line from filename
        X = np.load(features_file)
        y = np.load(labels_file)
        
        if (cell_line == 'mESC') or (cell_line == 'mNPC'):
            for chrom in train_chromosomes:
                if chrom not in chroms_not_in_mouse:
                    X_train_dict[(cell_line, chrom)] = X[chrom]
                    X_train_list.append(X[chrom])
                    y_train_arrays[(cell_line, chrom)] = y[chrom]
        else:
            for chrom in train_chromosomes:
                X_train_dict[(cell_line, chrom)] = X[chrom]
                X_train_list.append(X[chrom])
                y_train_arrays[(cell_line, chrom)] = y[chrom]
    
    # Concatenate training data and labels
    X_train = np.concatenate(X_train_list, axis=0)
    
    # Prepare test data
    X_test, y_test = {}, {}
    for features_file, labels_file in zip(features_files, labels_files):
        cell_line = features_file.split('/')[-1].split('_')[0]
        X = np.load(features_file)
        y = np.load(labels_file)
        
        if test_chromosomes in X and test_chromosomes in y:
            X_test[cell_line] = X[test_chromosomes]
            y_test[cell_line] = y[test_chromosomes]
    print("Raw data is loaded.")

    # Global scaling across all training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print("Data is scaled.")
    
    # Reassign scaled data back to the dictionary
    for key in X_train_dict:
        X_train_dict[key] = scaler.transform(X_train_dict[key])
    # Scale test data
    for key in X_test:
        X_test[key] = scaler.transform(X_test[key])
    
    # Convert data to PyTorch tensors
    for key in X_train_dict:
        X_train_dict[key] = torch.tensor(X_train_dict[key], dtype=torch.float32)
        y_train_arrays[key] = torch.tensor(y_train_arrays[key], dtype=torch.float32)
    for key in X_test:
        X_test[key] = torch.tensor(X_test[key], dtype=torch.float32)
        y_test[key] = torch.tensor(y_test[key], dtype=torch.float32)
    print("Data is converted to PyTorch tensors.")
    
    return X_train_dict, y_train_arrays, X_test, y_test