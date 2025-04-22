from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Returns a model instance based on name
def get_model(model_name):
    if model_name == 'linear':
        return LinearRegression()
    elif model_name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42
        )
    elif model_name == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            alpha=0.001,
            random_state=42,
            early_stopping=True
        )
    else:
        raise ValueError("Unsupported model: choose 'linear', 'random_forest', or 'mlp'")