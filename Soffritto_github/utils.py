import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_data(features_file, labels_file, train_chromosomes, test_chromosomes):
    X = np.load(features_file)
    y = np.load(labels_file)
    
    X_train_arrays = [X[chrom] for chrom in train_chromosomes]
    y_train_arrays = [y[chrom] for chrom in train_chromosomes]

    X_train = np.concatenate(X_train_arrays, axis=0)
    y_train = np.concatenate(y_train_arrays, axis=0)
    X_test = X[f"{test_chromosomes}"]
    y_test = y[f"{test_chromosomes}"]
    print("Raw data is loaded.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data is scaled.")

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("Data is converted to PyTorch tensors.")
    
    return X_train, y_train, X_test, y_test

def load_data_cross_cell_line(features_files, labels_files, train_chromosomes, test_chromosomes):
    chroms_not_in_mouse = ["20", "21", "22"]
    X_train_arrays, y_train_arrays = [], []

    for features_file, labels_file in zip(features_files, labels_files):
        X = np.load(features_file)
        y = np.load(labels_file)
        
        if ('mESC' in features_file) or ('mNPC' in features_file):
            X_train_arrays.extend([X[chrom] for chrom in train_chromosomes if chrom not in chroms_not_in_mouse])
            y_train_arrays.extend([y[chrom] for chrom in train_chromosomes if chrom not in chroms_not_in_mouse])

        else:
            X_train_arrays.extend([X[chrom] for chrom in train_chromosomes])
            y_train_arrays.extend([y[chrom] for chrom in train_chromosomes])
    
    X_train = np.concatenate(X_train_arrays, axis=0)
    y_train = np.concatenate(y_train_arrays, axis=0)
    X_test, y_test = [], []
    for features_file, labels_file in zip(features_files, labels_files):
        X = np.load(features_file)
        y = np.load(labels_file)
        if test_chromosomes in X and test_chromosomes in y:
            X_test.append(X[test_chromosomes])
            y_test.append(y[test_chromosomes])

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    print("Raw data is loaded.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data is scaled.")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("Data is converted to PyTorch tensors.")
    
    return X_train, y_train, X_test, y_test

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
