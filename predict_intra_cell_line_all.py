"""
Runs predict_intra_cell_line.py for each cell line.
"""
import os

cell_lines = ['H1', 'H9', 'HCT116', 'mESC', 'mNPC']

for cell_line in cell_lines:
    if 'm' in cell_line:
        train_chroms = (' ').join([str(i) for i in range(1,20) if i != 6])
    else:
        train_chroms = (' ').join([str(i) for i in range(1,23) if i != 6])
    command = f"python -u predict_intra_cell_line.py " \
              f"--train_features_file ./data/{cell_line}_features.npz " \
              f"--test_features_file ./data/{cell_line}_features.npz " \
              f"--test_labels_file ./data/{cell_line}_labels.npz " \
              f"--model_path ./trained_models/{cell_line}_intra_cell_line_model.pth " \
              f"--hyperparameter_file ./trained_models/{cell_line}_intra_cell_line_model_hyperparameters.json " \
              f"--train_chromosomes {train_chroms} " \
              f"--test_chromosomes 9 " \
              f"--pred_file ./predictions/{cell_line}_chr9_pred_intra_cell_line.npy"
    os.system(command)
