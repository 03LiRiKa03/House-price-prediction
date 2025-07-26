# House-price-prediction
## This project predicts house sale prices using the Kaggle Housing Prices dataset.

###  Data Preprocessing

1. Drop columns with many missing values.

1. Fill missing values with appropriate methods (None, 0, mode, median).

1. Map quality-related categorical features to numeric scores.

1. Encode categorical variables using _one-hot encoding_.

1. Align train and test datasets to have matching columns.

### Model Training and Evaluation

1. Split data into training and validation sets.

1. Train a _Random Forest Regressor_model.

1. Evaluate model performance using _RMSE_ metric.

1. Generate predictions on the test dataset.

1. Save results in **submission.csv** for _Kaggle_ submission.
