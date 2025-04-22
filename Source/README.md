# Air Quality Sensor Calibration Project

## Cấu trúc dự án
```
project/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── Linear_Regression_o3.pkl
│   ├── Random_Forest_o3.pkl
│   ├── Neural_Network_o3.pkl
│   ├── Linear_Regression_no2.pkl
│   ├── Random_Forest_no2.pkl
│   ├── Neural_Network_no2.pkl
│   └── scaler.pkl
├── linear_model.py
├── advanced_model.py
├── train.py
├── predict.py
├── utils.py
├── README.md
└── requirements.txt
```

## Hướng dẫn sử dụng
1. **Cài đặt thư viện**
   ```bash
   pip install -r requirements.txt
   ```
2. **Huấn luyện & đánh giá**
   ```bash
   python train.py
   ```
3. **Dự đoán dữ liệu test**
   ```bash
   python predict.py --model name_model --test data/test.csv

   ```

## Thư viện cần thiết
```
pandas
numpy
scikit-learn
statsmodels
scipy
joblib
```