from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def train_model(X_train, y_train):
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False)
    }

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
