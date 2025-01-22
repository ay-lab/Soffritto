"""
Runs predict_leave_one_cell_line_out.py for all left out cell line iterations.
"""
import os

cell_lines = ['H1', 'H9', 'HCT116', 'mESC', 'mNPC']

train_features_files_dict = {'H1': "./data/mNPC_features.npz ./data/H9_features.npz ./data/HCT116_features.npz ./data/mESC_features.npz",
                             'H9': "./data/H1_features.npz ./data/mNPC_features.npz ./data/HCT116_features.npz ./data/mESC_features.npz",
                             'HCT116': "./data/H1_features.npz ./data/H9_features.npz ./data/mNPC_features.npz ./data/mESC_features.npz",
                             'mESC': "./data/H1_features.npz ./data/H9_features.npz ./data/HCT116_features.npz ./data/mNPC_features.npz",
                             'mNPC': "./data/H1_features.npz ./data/H9_features.npz ./data/HCT116_features.npz ./data/mESC_features.npz"
                            }
for test_cell_line in cell_lines:
    train_features_files = train_features_files_dict[test_cell_line]
    command_template = (
    "python -u predict_leave_one_cell_line_out.py "
    "--train_features_files {train_features_files} "
    "--test_features_file ./data/{test_cell_line}_features.npz "
    "--test_labels_file ./data/{test_cell_line}_labels.npz "
    "--model_path ./trained_models/{test_cell_line}_left_out_model.pth "
    "--hyperparameter_file ./trained_models/{test_cell_line}_left_out_model_hyperparameters.json "
    "--train_chromosomes 1 2 3 4 5 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 "
    "--test_chromosomes 9 "
    "--pred_file ./predictions/{test_cell_line}_chr9_pred_leave_one_cell_line_out.npy"
    )
    command = command_template.format(train_features_files=train_features_files, test_cell_line=test_cell_line)
    os.system(command)
