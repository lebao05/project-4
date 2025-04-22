import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Create models directory
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Read data
df = pd.read_csv("./Dataset/train.csv")
df['Time'] = pd.to_datetime(df['Time'])
df['hour'] = df['Time'].dt.hour
df['dayofweek'] = df['Time'].dt.dayofweek

features = ['o3op1', 'o3op2', 'no2op1', 'no2op2', 'temp', 'humidity', 'hour', 'dayofweek']
X = df[features]
y_o3 = df['OZONE']
y_no2 = df['NO2']

X_train, X_val, y_o3_train, y_o3_val = train_test_split(X, y_o3, test_size=0.2, random_state=42)
_, _, y_no2_train, y_no2_val = train_test_split(X, y_no2, test_size=0.2, random_state=42)

# === Model 1: Decision Tree ===
print("\n Training Decision Tree models...")
tree_o3 = DecisionTreeRegressor(random_state=42)
tree_o3.fit(X_train, y_o3_train)
mae_tree_o3 = mean_absolute_error(y_o3_val, tree_o3.predict(X_val))
joblib.dump(tree_o3, model_dir/"decision_tree_o3.pkl")
print("- O3 model trained and saved")

tree_no2 = DecisionTreeRegressor(random_state=42)
tree_no2.fit(X_train, y_no2_train)
mae_tree_no2 = mean_absolute_error(y_no2_val, tree_no2.predict(X_val))
joblib.dump(tree_no2, model_dir/"decision_tree_no2.pkl")
print("- NO2 model trained and saved")

# === Model 2: SVR with RBF kernel ===
print("\n Training SVR models...")
svr_o3 = SVR(kernel='rbf')
svr_o3.fit(X_train, y_o3_train)
mae_svr_o3 = mean_absolute_error(y_o3_val, svr_o3.predict(X_val))
joblib.dump(svr_o3, model_dir/"svr_o3.pkl")
print("- O3 model trained and saved")

svr_no2 = SVR(kernel='rbf')
svr_no2.fit(X_train, y_no2_train)
mae_svr_no2 = mean_absolute_error(y_no2_val, svr_no2.predict(X_val))
joblib.dump(svr_no2, model_dir/"svr_no2.pkl")
print("- NO2 model trained and saved")

# === Model 3: K-Nearest Neighbors ===
print("\n Training KNN models...")
knn_o3 = KNeighborsRegressor(n_neighbors=5)
knn_o3.fit(X_train, y_o3_train)
mae_knn_o3 = mean_absolute_error(y_o3_val, knn_o3.predict(X_val))
joblib.dump(knn_o3, model_dir/"knn_o3.pkl")
print("- O3 model trained and saved")

knn_no2 = KNeighborsRegressor(n_neighbors=5)
knn_no2.fit(X_train, y_no2_train)
mae_knn_no2 = mean_absolute_error(y_no2_val, knn_no2.predict(X_val))
joblib.dump(knn_no2, model_dir/"knn_no2.pkl")
print("- NO2 model trained and saved")

# === Model 4: Neural Network (MLP) ===
print("\n Training Neural Network models...")
nn_o3 = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
nn_o3.fit(X_train, y_o3_train)
mae_nn_o3 = mean_absolute_error(y_o3_val, nn_o3.predict(X_val))
joblib.dump(nn_o3, model_dir/"mlp_o3.pkl")
print("- O3 model trained and saved")

nn_no2 = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
nn_no2.fit(X_train, y_no2_train)
mae_nn_no2 = mean_absolute_error(y_no2_val, nn_no2.predict(X_val))
joblib.dump(nn_no2, model_dir/"mlp_no2.pkl")
print("- NO2 model trained and saved")

# === Print MAE results ===
print("\n🔍 MAE for OZONE:")
print(f"Decision Tree: {mae_tree_o3:.2f}")
print(f"SVR (RBF):     {mae_svr_o3:.2f}")
print(f"KNN (k=5):     {mae_knn_o3:.2f}")
print(f"MLP Regressor: {mae_nn_o3:.2f}")

print("\n🔍 MAE for NO2:")
print(f"Decision Tree: {mae_tree_no2:.2f}")
print(f"SVR (RBF):     {mae_svr_no2:.2f}")
print(f"KNN (k=5):     {mae_knn_no2:.2f}")
print(f"MLP Regressor: {mae_nn_no2:.2f}")

print(f"\n All models saved to: {model_dir.absolute()}")