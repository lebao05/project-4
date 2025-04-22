import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error

# Load training and test data
train_data = pd.read_csv("./Dataset/train.csv", parse_dates=["Time"])
test_data = pd.read_csv("./Dataset/dummy_test.csv", parse_dates=["Time"])
columns_to_drop = ["temperature", "humidity", "Time"]
train_data = train_data.drop(columns=columns_to_drop, errors="ignore")
test_data = test_data.drop(columns=columns_to_drop, errors="ignore")

print("5 FIRST ROWS OF TRAINING DATASET")
print(train_data.head())
print("./")
print("5 FIRST ROWS OF TEST DATASET")
print(test_data.head())

# Features and targets for training
X_train = train_data[["no2op1", "no2op2", "o3op1", "o3op2"]]
y_train_ozone = train_data["OZONE"]
y_train_no2 = train_data["NO2"]

# Features and targets for test
X_test = test_data[["no2op1", "no2op2", "o3op1", "o3op2"]]
y_test_ozone = test_data["OZONE"]
y_test_no2 = test_data["NO2"]


def training(model, X_train, y_train, model_name, target_name):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    print(f"{model_name} - {target_name}: MAE (Train) = {mae_train:.4f}")
    print(f"{model_name} - {target_name} Coefficients:")
    print(f"p ({target_name.lower()}) (o3op1) = {model.coef_[2]:.4f}")
    print(f"q ({target_name.lower()}) (o3op2) = {model.coef_[3]:.4f}")
    print(f"r ({target_name.lower()}) (no2op1) = {model.coef_[0]:.4f}")
    print(f"s ({target_name.lower()}) (no2op2) = {model.coef_[1]:.4f}")
    print(f"t ({target_name.lower()}) (intercept) = {model.intercept_:.4f}\n")
    return mae_train


print("LEAST SQUARE METHOD WITHOUT REGULARIZATION (OZONE)")
BEST_OZONE_MODEL = LinearRegression()
training(
    BEST_OZONE_MODEL,
    X_train,
    y_train_ozone,
    "least square",
    "OZONE",
)

print("LEAST SQUARE METHOD WITH RIDGE (OZONE)")
ridge_OZONE_Model = Ridge(alpha=1.0)
training(
    ridge_OZONE_Model,
    X_train,
    y_train_ozone,
    "Ridge regression",
    "OZONE",
)

print("LEAST SQUARE METHOD WITH LASSO (OZONE)")
lasso_OZONE_Model = Lasso(alpha=1.0)
training(
    lasso_OZONE_Model,
    X_train,
    y_train_ozone,
    "Lasso regression",
    "OZONE",
)

print("LEAST SQUARE METHOD WITHOUT REGULARIZATION (NO2)")
BEST_NO2_MODEL = LinearRegression()
training(BEST_NO2_MODEL, X_train, y_train_no2, "least square", "NO2")

print("LEAST SQUARE METHOD WITH RIDGE (NO2)")
ridge_NO2_Model = Ridge(alpha=1.0)
training(ridge_NO2_Model, X_train, y_train_no2, "Ridge regression", "NO2")

print("LEAST SQUARE METHOD WITH LASSO (NO2)")
lasso_NO2_Model = Lasso(alpha=1.0)
training(lasso_NO2_Model, X_train, y_train_no2, "Lasso regression", "NO2")
