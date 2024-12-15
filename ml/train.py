import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

def training(file, model):
    if model == "xgb":
        model = XGBRegressor(random_state=42)

        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [5, 10, 20],
            'min_child_weight': [3, 5],
            'learning_rate': [0.01, 0.05, 0.2],
            'subsample': [0.6, 0.8]
        }

    if model == "rf":
        model = RandomForestRegressor(random_state=42)

        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

    if model == "lgbm":
        model = LGBMRegressor(random_state=42)

        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [10, 20, 30],
            'min_child_samples': [5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8],
            'num_leaves':[500,1000]
            
        }
    
    target_column = 'energy'
    columns_to_drop = ['id', 'mf','energy']  
    data = pd.read_csv(file)
    y = data[target_column]
    X = data.drop(columns=columns_to_drop).fillna(0)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    print("Best hyperparameters:", grid_search.best_params_)
    
    predictions = grid_search.best_estimator_.predict(X_test_scaled)

    absolute_percentage_errors = np.abs((predictions - y_test) / y_test) * 100
    re_percentage = np.mean(absolute_percentage_errors)
    re_std = np.std(absolute_percentage_errors)
    print("Mean RE%:", re_percentage)
    print("Std RE:", re_std)
    
    plt.figure(figsize=(12, 6))
    sample_indices = range(len(y_test))
    plt.plot(sample_indices, y_test, 'o-', color='blue', label='Ground Truth', markersize=4)
    plt.plot(sample_indices, predictions, 'o-', color='orange', label='Predictions', markersize=4, alpha=0.7)
    
    plt.xlabel("Sample Index")
    plt.ylabel("Energy")
    plt.title("Predictions vs Ground Truth for Energy")
    plt.legend()
    plt.show()

    explainer = shap.Explainer(grid_search.best_estimator_, feature_names=X_train.columns)
    shap_values = explainer(X_test_scaled)
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, max_display=5)
    
    return grid_search.best_params_, re_percentage, re_std