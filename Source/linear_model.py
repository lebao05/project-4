import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error

# Training function for linear-family models (no env/time features)
def training(model, X_train, y_train, model_name, target_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    print(f"{model_name} - {target_name}: MAE (Train) = {mae:.4f}")
    coeffs = model.coef_  # [no2op1, no2op2, o3op1, o3op2]
    intercept = model.intercept_
    print(f"p ({target_name.lower()}) (o3op1)     = {coeffs[2]:.4f}")
    print(f"q ({target_name.lower()}) (o3op2)     = {coeffs[3]:.4f}")
    print(f"r ({target_name.lower()}) (no2op1)    = {coeffs[0]:.4f}")
    print(f"s ({target_name.lower()}) (no2op2)    = {coeffs[1]:.4f}")
    print(f"t ({target_name.lower()}) (intercept) = {intercept:.4f}\n")
    return mae