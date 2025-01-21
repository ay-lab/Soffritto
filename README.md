# Soffritto
## Requirements
To set up the conda environment to run Soffritto, run:
```
conda env create -f environment.yml
```
Then, 
```
cd path_to_Soffritto_directory
conda activate soffritto
```

## Prediction
To reproduce the intra-cell line predictions in the paper, run the following command from the command line inside the Soffritto directory: 
```
python -u predict_intra_cell_line.py \
--train_features_file ./data/${CELL_LINE}_features.npz \
--test_features_file ./data/${CELL_LINE}_features.npz \
--test_labels_file ./data/${CELL_LINE}_labels.npz \
--model_path ./trained_models/${CELL_LINE}_intra_cell_line_model.pth \
--hyperparameter_file ./trained_models/${CELL_LINE}_left_out_model_hyperparameters.json \
--train_chromosomes 1 2 3 4 5 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 \
--test_chromosomes 9 \
--pred_file ./predictions/${CELL_LINE}_chr9_pred_intra_cell_line.npy
```
where ${CELL_LINE} is one of H1, H9, HCT116, mESC, and mNPC. Omit chromosomes 20, 21, and 22 for the mouse cell lines (mESC and mNPC) in the --train_chromosomes flag.

To reproduce the leave-one-cell-line-out predictions, run:
```
python -u predict_leave_one_cell_line_out.py \
--train_features_files ./data/H1_features.npz ./data/H9_features.npz ./data/HCT116_features.npz ./data/mESC_features.npz \
--test_features_file ./data/mNPC_features.npz \
--test_labels_file ./data/mNPC_labels.npz \
--model_path ./trained_models/mNPC_left_out_model.pth \
--hyperparameter_file ./trained_models/mNPC_left_out_model_hyperparameters.json \
--train_chromosomes 1 2 3 4 5 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 \
--test_chromosomes 9 \
--pred_file ./predictions/mNPC_chr9_pred_leave_one_cell_line_out.npy
```
