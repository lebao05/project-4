from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, target_name):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"{model_name} - {target_name}: MAE (Train) = {mae_train:.4f}, MAE (Test) = {mae_test:.4f}")
    print(f"{model_name} - {target_name} Coefficients:")
    print(f"p ({target_name.lower()}) (o3op1) = {model.coef_[2]:.4f}")
    print(f"q ({target_name.lower()}) (o3op2) = {model.coef_[3]:.4f}")
    print(f"r ({target_name.lower()}) (no2op1) = {model.coef_[0]:.4f}")
    print(f"s ({target_name.lower()}) (no2op2) = {model.coef_[1]:.4f}")
    print(f"t ({target_name.lower()}) (intercept) = {model.intercept_:.4f}\n")
    return mae_train, mae_test

# Models to evaluate
models = {
    "Least Squares": LinearRegression(),
    "Lasso (alpha=1.0)": Lasso(alpha=1.0),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0)
}

# Evaluate models for OZONE
print("Evaluating models for OZONE prediction:")
ozone_results = {}
for name, model in models.items():
    mae_train, mae_test = evaluate_model(model, X_train, y_train_ozone, X_test, y_test_ozone, name, "OZONE")
    ozone_results[name] = (mae_train, mae_test)

# Evaluate models for NO2
print("\nEvaluating models for NO2 prediction:")
no2_results = {}
for name, model in models.items():
    mae_train, mae_test = evaluate_model(model, X_train, y_train_no2, X_test, y_test_no2, name, "NO2")
    no2_results[name] = (mae_train, mae_test)

# Summary of best models (based on training MAE)
best_ozone_model = min(ozone_results, key=lambda k: ozone_results[k][0])
best_no2_model = min(no2_results, key=lambda k: no2_results[k][0])

print(f"\nBest model for OZONE (based on training MAE): {best_ozone_model}")
print(f"OZONE MAE (Train) = {ozone_results[best_ozone_model][0]:.4f}, MAE (Test) = {ozone_results[best_ozone_model][1]:.4f}")
print(f"Best model for NO2 (based on training MAE): {best_no2_model}")
print(f"NO2 MAE (Train) = {no2_results[best_no2_model][0]:.4f}, MAE (Test) = {no2_results[best_no2_model][1]:.4f}")