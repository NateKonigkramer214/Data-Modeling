import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np

class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'Linear Regression':
            return LinearRegression()
        elif model_name == 'Ridge Regression':
            return Ridge()
        elif model_name == 'Poisson Regression':
            return PoissonRegressor()
        elif model_name == 'Support Vector Machine':
            return SVR()
        elif model_name == 'Random Forest':
            return RandomForestRegressor()
        elif model_name == 'Gradient Boosting Regressor':
            return GradientBoostingRegressor()
        elif model_name == 'XGBRegressor':
            return XGBRegressor()
        else:
            raise ValueError(f"Model '{model_name}' not recognized!")

#Loads the data from the excel defined by file_name varible this is then loaded into a pandas dataframe
def load_data(file_name):
    return pd.read_excel(file_name)

def preprocess_data(data):
# Check for missing values and raises a error if any missing values are found.
    if data.isnull().any().any():
        raise ValueError("The data contains missing values. Please ensure the data is cleaned before processing.")
#Seperates the features X and the target varible Y
    #!Input Data
    X = data.drop(['Client Name', 'Client e-mail', 'Country', 'Education', 'Age', 'Profession', 'Healthcare Cost', 'Net Worth'], axis=1)
    #!Output
    Y = data['Net Worth']
    
#Scaling the features and the target variable using Min-Max scaling.
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1

#Splits the data into training and testing data from scikit-learn
def split_data(X_scaled, y_scaled):
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

#This function trains a set of regression models specified by the model names list using the training data
def train_models(X_train, y_train):
    model_names = [
        'Ridge Regression',
        'Poisson Regression',
        'Linear Regression',
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor'
    ]
    
    models = {}
    for name in model_names:
        #Display model name
        print(f"Training model: {name}")
        model = ModelFactory.get_model(name)
        model.fit(X_train, y_train.ravel())
        models[name] = model
        #display when model trained successfully
        print(f"{name} trained successfully.")
#It returns a dictionary where keys are model names, and values are the trained model objects.       
    return models

#This function evaluates the models by calculating the root mean squared error 
# RMSE for each model using the testing data
def evaluate_models(models, X_test, y_test):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
#Returns a dictanary of the where keys are the model names, and values and RMSE scores   
    return rmse_values

#
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
    final = LinearRegression()
    final.fit(X_scaled,y_scaled)
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    dump(best_model, "sr1_4_model.joblib")

def predict_new_data(loaded_model, sc, sc1):
    X_test1 = sc.transform(np.array([[32,62822.09322,22609.389291,36221.45877,71.9824179984,63670.4407087328,32570.87429,30118.9767,51623.023410268]]))
    pred_value = loaded_model.predict(X_test1)
    print(pred_value)
    
    # Ensure pred_value is a 2D array before inverse transform
    if len(pred_value.shape) == 1:
        pred_value = pred_value.reshape(-1, 1)

    print("Predicted output: ", sc1.inverse_transform(pred_value))

if __name__ == "__main__":
    try: #add try except to handle missing value error
        data = load_data('Net_Worth_Data.xlsx')
        X_scaled, y_scaled, sc, sc1 = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
        models = train_models(X_train, y_train)
        rmse_values = evaluate_models(models, X_test, y_test)
        plot_model_performance(rmse_values)
        save_best_model(models, rmse_values)
        loaded_model = load("model.joblib")
        predict_new_data(loaded_model, sc, sc1)
    except ValueError as ve:
        print(f"Error: {ve}")