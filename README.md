# Dinkle
This project is NCCU cooperation with Dinkle, which used LSTM and SHAP in product quality prediction. 
This GitHub focus on reasoning product quality prediction with DeepSHAP.

# Process
## Model 
data_preprocessing -> training model -> data_testing -> analysis -> SHAP
> training model code is not in this github.

## Prediction
(pred_cleaning) -> pred_preprocessing -> prediction -> analysis
> pred_cleaning is not necessary, only for prediction can show scatter plot.

# Description
## data_preprocessing
- Input: Data after cleaning

- Output: X_train, Y_train, X_test, Y_test

## data_testing
- Input: model class and setting, model.pt, X_test, Y_test
> If you didn't run testing right after training model, then you need to load model.

- Output: test.csv, accuracy, X_test_correct, X_test_wrong, Confusion matrix

## analysis
### Testing
- Input: test.csv

- Output: each detail accuracy, pictures of confusion matrix, pictures of error distribution histogram

### Prediction
- Input: pred.npy

- Output: Pictures of error distribution histogram

## SHAP
### SHAP 
- Input: model class and setting, model.pt, X_train.npy, X_test.npy

- Output: shap.npy
### SHAP Heatmap
- Input: shap.npy

- Output: Pictures of the heatmap, and the percentage of each feature

## pred_cleaning
- Input: Raw data

- Output: true.npy

## pred_preprocessing
- Input: Data after cleaning

- Output: test.npy, label.npy

## prediction
- Input: model class and setting, model.pt, true.npy, test.npy, label.npy

- Output: pred.npy, test.csv, accuracy, pictures

# Example Folder
You can follow the `model_example` for model testing, analysis and reasoning product quality prediction with DeepSHAP; `prediction_example` for prediction and analysis.

Some sample data and model are also provided in this folder.

## Using Google Colaboratory
1. Download the `model_example.ipynb` and `prediction_example.ipynb`, upload it to your google drive, then use google colaboratory to run all. (The easiest way!)

2. Download whole folder and upload it to your google drive, then use google colaboratory to open the `model_example.ipynb` and `prediction_example.ipynb`. Remind that the `folder_path` need to change to the path of your folder.

## Using local environment
Check your environment has installed the necessnary packages. Download whole folder, then open the `model_example.py` and `prediction_example.py`. Remind that the `folder_path` need to change to the path of your folder.