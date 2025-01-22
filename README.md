# Soffritto
Soffritto is a deep learning model that predicts 16-fraction replication timing (represented as a high-resolution Repli-Seq heatmap) using six histone ChIP-Seq signals, GC content, gene density, and 2-fraction Repli-Seq data as input. Soffritto is composed of an Long Short Term Memory (LSTM) module and a fully connected prediction module. Scripts are provided to both train the model from scratch and predict using trained models. Soffritto was evaluated on five cell lines: H1, H9, HCT116, mESC, mNPC using an intra-cell line and Leave-One-Cell-Line-Out (LOCLO) strategy. In the intra-cell line framework, chromosome 6 was left out for validation and chromosome 9 was used as the test set. In the LOCLO evaluation, Soffritto was iteratively trained on four cell lines, leaving chromosome 6 out for validation in the four cells and chromosome 9 out entirely to ensure an unseen chromosome in the left out cell line.

## Requirements
To set up the conda environment to run Soffritto, run:
```
cd Soffritto
conda env create -f environment.yml
```

## Directories 
### data
This directory contains feature, label, and their corresponding genomic coordinate files for each cell line. The feature files are the input for Soffritto while the label files are the 16 fraction replication timing signals. All files in this directory are in numpy npz format. Each file's data is stored chromosome-wise and can be accessed in a similar fashion to a dictionary. For example, to access chromosome 1's data from H1_features.npz, one would run 
```python
import numpy as np
features = np.load("H1_features.npz")
data = features["1"]
```
The feature and label files are in the formats {cell_line}_features.npz and {cell_line}_labels.npz respectively. Each chromosome's data for the features and labels is a 2D numpy array where the rows correspond to genomic bins of size 50kb. The coordinates for these bins are provided in the same order in the {cell_line}_coordinates.bedgraph files. The columns in the feature files correspond to the following features in this order: H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, GC content, gene density, and 2-stage replication timing. The columns in the label files correspond to the 16 S phase fractions ordered from earliest (S1) to latest (S16). The labels are normalized such that each row sums to 1.

### trained_models
This directory contains trained model files for both intra-cell line evaluation and LOCLO evaluation. The json files contain the hidden size and number of LSTM layers for each trained model. The trained models files are in the PyTorch state dictionary format (.pth). The LOCLO model files are formatted as {cell_line}_left_out_model.pth to indicate that the model was trained on all cell lines except for {cell_line}. 

## Prediction
To reproduce the intra-cell line predictions in the paper, run the following command from the command line inside the Soffritto directory: 
```
python predict_intra_cell_line_all.py
```
predict_intra_cell_line_all.py predicts for all five cell lines. It creates a directory called predictions and outputs a predicted 16-fraction replication timing heatmap for chromosome 9 for each cell line with path ./predictions/{cell_line}_chr9_pred_intra_cell_line.npy. The predicted .npy file is a (bins x S phase fraction) matrix with rows normalized to sum to 1 just like the label files. 

To reproduce the LOCLO predictions, run:
```
python predict_LOCLO_all_iterations.py
```
predict_LOCLO_all_iterations.py iterates over all cell lines and predicts chromosome 9 for the left out cell line. The predictions are saved to file as ./predictions/{test_cell_line}_chr9_pred_leave_one_cell_line_out.npy

## Training
Soffritto may also be trained from scratch. Training within cell lines and in LOCLO fashion are implemented by train_intra_cell_line.py and train_leave_one_cell_line_out.py respectively. Both files save the trained model to file.
### train_intra_cell_line.py
This table provides a list of command line flags:
| Flag  | Type | Description |
| ------------- | --- | ------------- |
| --features | str  |  Path to features dataset |
| --labels | str |    Path to 16-fraction RT data  |
| --model_path  | str  |  Path for the trained model (saved as a .pth file)  |
| --train_chromosomes  | int  | Training chromosomes as space-separated list of integers |
| --val_chromosomes  | int  | Validation chromosome as integer |
| --learning_rate  | float  | Learning rate for Adam optimizer |
| --num_epochs | int  | Number of epochs |
| --batch_size | int  | Batch size |
| --num_hiddens  | int  | Hidden size of LSTM module |
| --num_layers  | int  | Number of LSTM layers |
| --weight_decay | float  | L2 regularization coefficient |

### train_leave_one_cell_line_out.py
| Flag  | Type | Description |
| ------------- | --- | ------------- |
| --features | str  |  Paths to features datasets (space-separated) |
| --labels | str |  Paths to 16-fraction RT data (space-separated and in the some order as features files)  |
| --model_path  | str  |  Path for the trained model (saved as a .pth file)  |
| --train_chromosomes | int  | Training chromosomes as space-separated list of integers |
| --val_chromosomes  | int  | Validation chromosome as integer |
| --learning_rate  | float  | Learning rate for Adam optimizer |
| --num_epochs | int  | Number of epochs |
| --batch_size | int  | Batch size |
| --num_hiddens  | int  | Hidden size of LSTM module |
| --num_layers  | int  | Number of LSTM layers |
| --weight_decay | float  | L2 regularization coefficient |
