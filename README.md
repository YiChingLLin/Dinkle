# Dinkle
This project is NCCU cooperation with Dinkle, used AI in product quality prediction. 

# Process
## Model 
data_preprocessing -> training model -> data_testing -> analysis -> SHAP
* training model code is not in this github.

## Prediction
pred_preprocessing -> prediction -> analysis

# Description
## data_preprocessing
- Input: Data after cleaning

- Output: X_train, Y_train, X_test, Y_test

## data_testing
- Input: model class and setting, model.pt, X_test, Y_test
> If you didn't run testing right after training model, then you need to load model.

- Output: test.csv, Accuracy, X_test_correct, X_test_wrong, Confusion matrix

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

- Output: Picture of the heatmap, and the percentage of each feature

## pred_preprocessing
- Input: Data after cleaning

- Output: test, label

## prediction
- Input: model class and setting, model.pt, true.npy, test.npy, label.npy

- Output: pred.npy, test.csv, accuracy, pictures