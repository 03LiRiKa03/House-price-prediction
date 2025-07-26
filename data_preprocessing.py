import pandas as pd

def preprocess_data(train_path='train.csv', test_path='test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    train.drop(columns=drop_cols, inplace=True)
    test.drop(columns=drop_cols, inplace=True)

    none_fill_cols = [
        'MasVnrType', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
    ]
    for col in none_fill_cols:
        train[col] = train[col].fillna('None')
        test[col] = test[col].fillna('None')

    zero_fill_cols = [
         'MasVnrArea', 'GarageYrBlt', 'GarageCars', 'GarageArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath'
    ]
    for col in zero_fill_cols:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    mode_fill_cols = [
         'MSZoning', 'KitchenQual', 'Exterior1st', 'Exterior2nd',
        'SaleType', 'Electrical', 'Functional', 'Utilities'
    ]
    for col in mode_fill_cols:
        train[col] = train[col].fillna(train[col].mode()[0])
        test[col] = test[col].fillna(test[col].mode()[0])

    median_fill_cols = [
          'LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea',
        'GarageCars', 'BsmtFullBath', 'BsmtHalfBath'
    ]
    for col in median_fill_cols:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(test[col].median())

    quality_map = {
        'None': 0,
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5
    }
    quality_cols = [
         'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageQual', 'GarageCond'
    ]
    for col in quality_cols:
        train[col] = train[col].map(quality_map)
        test[col] = test[col].map(quality_map)

    # Кодуємо всі інші текстові категоріальні змінні в даммі-перемінні
    cat_cols = train.select_dtypes(include=['object']).columns
    train = pd.get_dummies(train, columns=cat_cols)
    test = pd.get_dummies(test, columns=cat_cols)

    # Вирівнюємо колонки train і test
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    return train, test