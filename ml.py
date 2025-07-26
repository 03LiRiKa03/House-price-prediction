import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data_preprocessing import preprocess_data

train_encoded, test_encoded = preprocess_data()

test_original = pd.read_csv('test.csv')

X = train_encoded.drop('SalePrice', axis=1)
y = train_encoded['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=52
)

model = RandomForestRegressor(n_estimators=125, random_state=52)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse:.2f}")

if 'SalePrice' in test_encoded.columns:
    test_encoded = test_encoded.drop('SalePrice', axis=1)

y_test_pred = model.predict(test_encoded)

submission = pd.DataFrame({
    'Id': test_original['Id'],  
    'SalePrice': y_test_pred
})

submission.to_csv('submission.csv', index=False)
