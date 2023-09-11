import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import ridge_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np


def load_data(file_name):
    return pd.read_excel(file_name)


def preprocess_data(data):
    #!Input Data
    X = data.drop(['Client Name', 'Client e-mail', 'Country', 'Gender', 'Education', 'Age', 'Profession', 'Healthcare Cost'], axis=1)
    #!Output
    Y = data['Net Worth']
    
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1


def split_data(X_scaled, y_scaled):
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


def train_models(X_train, y_train):
    models = {
        'Ridge Regression' : ridge_regression(),
        'Linear Regression': LinearRegression(),
        'Support Vector Machine': SVR(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(),
        'XGBRegressor': XGBRegressor()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train.ravel())
        models[name] = model
        
    return models


def evaluate_models(models, X_test, y_test):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    return rmse_values


def plot_model_performance(rmse_values):
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def save_best_model(models, rmse_values):
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    dump(best_model, "car_model.joblib")
    

def predict_new_data(loaded_model, sc, sc1):
    X_test1 = sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))
    pred_value = loaded_model.predict(X_test1)
    print(pred_value)
    
    # Ensure pred_value is a 2D array before inverse transform
    if len(pred_value.shape) == 1:
        pred_value = pred_value.reshape(-1, 1)

    print("Predicted output: ",sc1.inverse_transform(pred_value))


if __name__ == "__main__":
    data = load_data('Net_Worth_Data.xlsx')
    X_scaled, y_scaled, sc, sc1 = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    models = train_models(X_train, y_train)
    rmse_values = evaluate_models(models, X_test, y_test)
    plot_model_performance(rmse_values)
    save_best_model(models, rmse_values)
    loaded_model = load("model.joblib")
    predict_new_data(loaded_model, sc, sc1)