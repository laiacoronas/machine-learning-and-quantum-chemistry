import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

def test(file):
    test_data = pd.read_csv(file).fillna(0)
    columns_to_drop = ['id', 'mf', 'energy']  
    target_column = 'energy'  

    if "pubchem" in file:
        model = XGBRegressor(random_state=42)
        param_grid = {'learning_rate': [0.05], 'max_depth': [5], 'min_child_weight': [5], 'n_estimators': [500], 'subsample': [0.6]}
        train_data = pd.read_csv("datasets_corrected/training/pubchem.csv").fillna(0)
    elif "coulomb" in file:
        model = RandomForestRegressor(random_state=42)
        param_grid = {'max_depth': [10], 'max_features': ['sqrt'], 'min_samples_leaf': [1], 'min_samples_split': [2], 'n_estimators': [500]}
        train_data = pd.read_csv("datasets_corrected/training/coulomb.csv").fillna(0)
    elif "combined" in file:
        model = LGBMRegressor(random_state=42)
        param_grid = {'learning_rate': [0.1], 'max_depth': [10], 'min_child_samples': [10], 'n_estimators': [500], 'subsample': [0.6]}
        train_data = pd.read_csv("datasets_corrected/training/combined.csv").fillna(0)
 
    y_train = train_data[target_column]
    X_train = train_data.drop(columns=columns_to_drop)
    y_test = test_data[target_column]
    X_test = test_data.drop(columns=columns_to_drop)

    for col in X_train.columns.difference(X_test.columns):
        X_test[col] = 0
    for col in X_test.columns.difference(X_train.columns):
        X_train[col] = 0
    X_test = X_test[X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    predictions = grid_search.best_estimator_.predict(X_test_scaled)

    absolute_percentage_errors = np.abs((predictions - y_test) / y_test) * 100
    re_percentage = np.mean(absolute_percentage_errors)
    re_std = np.std(absolute_percentage_errors)

    print("Mean RE%:", re_percentage)
    print("Std RE:", re_std)

    return grid_search.best_params_, re_percentage, re_std
