import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from advanced_model import get_model
from linear_model import training
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Create models directory
os.makedirs('models', exist_ok=True)

# --- Part 1: Linear-family baseline ---
print("\n=== Linear-Family Models ===\n")
# Load, drop environmental/time features
df_lm = pd.read_csv('data/train.csv', parse_dates=['Time'])
X_lm = df_lm[['no2op1','no2op2','o3op1','o3op2']]
Y_o3  = df_lm['OZONE']
Y_no2 = df_lm['NO2']
# Train linear models
training(LinearRegression(), X_lm, Y_o3,  "Least Square",   "OZONE")
training(Ridge(alpha=1.0),    X_lm, Y_o3,  "Ridge Regression","OZONE")
training(Lasso(alpha=1.0),    X_lm, Y_o3,  "Lasso Regression","OZONE")
training(LinearRegression(), X_lm, Y_no2,"Least Square",   "NO2")
training(Ridge(alpha=1.0),    X_lm, Y_no2,"Ridge Regression","NO2")
training(Lasso(alpha=1.0),    X_lm, Y_no2,"Lasso Regression","NO2")

# --- Part 2: Advanced Evaluation ---
# Load and engineer time features
df = pd.read_csv('data/train.csv', parse_dates=['Time'])
df['hour']    = df['Time'].dt.hour
df['hour_sin']= np.sin(2*np.pi*df['hour']/24)
df['hour_cos']= np.cos(2*np.pi*df['hour']/24)
features = ['o3op1','o3op2','no2op1','no2op2','temp','humidity','hour_sin','hour_cos']
X = df[features]
y_o3  = df['OZONE']
y_no2 = df['NO2']

# Scale & split
df_scaled = StandardScaler().fit_transform(X)
X_tr, X_val, y_tr_o3, y_val_o3, y_tr_no2, y_val_no2 = train_test_split(
    df_scaled, y_o3, y_no2, test_size=0.2, random_state=42
)

# Evaluation function with diagnostics
def evaluate(model, X_tr, y_tr, X_vl, y_vl, name):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_vl)
    infer_time = time.time() - t1

    mae = mean_absolute_error(y_vl, y_pred)
    r2  = r2_score(y_vl, y_pred)
    residuals = y_vl - y_pred

    sw_p = shapiro(residuals)[1]
    Xc = sm.add_constant(X_vl)
    bp_p = het_breuschpagan(residuals, Xc)[3]

    bic = None
    if name == 'Linear Regression':
        rss = np.sum(residuals**2)
        n   = len(residuals)
        k   = Xc.shape[1]
        bic = n * np.log(rss/n) + k * np.log(n)

    print(f"--- {name} ---")
    print(f"MAE           : {mae:.4f}")
    print(f"R2            : {r2:.4f}")
    if bic is not None:
        print(f"BIC           : {bic:.4f}")
    print(f"Train Time    : {train_time:.2f} s")
    print(f"Inference Time: {infer_time:.2f} s")
    print(f"Shapiro p-val : {sw_p:.4f}")
    print(f"BP p-val      : {bp_p:.4f}\n")
    return model

# Evaluate and save models
eval_models = [
    ('linear',         'Linear Regression'),
    ('random_forest',  'Random Forest'),
    ('mlp',            'Neural Network'),
]


print("\n=== Advanced Models O3 ===\n")
for key,name in eval_models:
    model = get_model(key)
    trained = evaluate(model, X_tr, y_tr_o3, X_val, y_val_o3, name)
    joblib.dump(trained, f"models/{key}_o3.pkl")

print("\n=== Advanced Models NO2 ===\n")
for key,name in eval_models:
    model = get_model(key)
    trained = evaluate(model, X_tr, y_tr_no2, X_val, y_val_no2, name)
    joblib.dump(trained, f"models/{key}_no2.pkl")


# Save scaler
joblib.dump(StandardScaler().fit(X), 'models/scaler.pkl')