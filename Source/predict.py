import argparse
import pandas as pd
import numpy as np
import joblib

# Đọc tham số
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['linear','random_forest','mlp'], default='random_forest')
    p.add_argument('--test', default='data/test.csv')
    return p.parse_args()

args = get_args()
# Tải scaler và model
df = pd.read_csv(args.test, parse_dates=['Time'])
scaler    = joblib.load('models/scaler.pkl')
model_o3  = joblib.load(f'models/{args.model}_o3.pkl')
model_no2 = joblib.load(f'models/{args.model}_no2.pkl')

# Tiền xử lý
df['hour']     = df['Time'].dt.hour
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
features = ['o3op1','o3op2','no2op1','no2op2','temp','humidity','hour_sin','hour_cos']
X_test = scaler.transform(df[features])

# Dự đoán
df['OZONE_PRED'] = model_o3.predict(X_test)
df['NO2_PRED']   = model_no2.predict(X_test)

# Lưu kết quả
df[['Time','OZONE_PRED','NO2_PRED']].to_csv('data/predictions.csv', index=False)
print('Saved predictions to data/predictions.csv')