from  main import load_data, preprocess_data, split_data
import pandas as pd
import numpy as np

#Test case to ensure the correct loading of data
def test_load_data():
    data = load_data('D:\Assesment_12_09\Net_Worth_Data.xlsx')
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0

#Test case to ensure the correct range of data
def test_data_range():
    data = load_data(r'D:\Assesment_12_09\Net_Worth_Data.xlsx')
    X_scaled, y_scaled, _, _ = preprocess_data(data)
    
    assert 0 <= np.min(X_scaled) <= 1, "X data should be scaled between 0 and 1."
    assert 0 <= np.min(y_scaled) <= 1, "Y data should be scaled between 0 and 1."
    assert 0 <= np.max(X_scaled) <= 1, "X data should be scaled between 0 and 1."
    assert 0 <= np.max(y_scaled) <= 1, "Y data should be scaled between 0 and 1."

#Test case to ensure the correct splitting of data
def test_data_split():
    data = load_data(r'D:\Assesment_12_09\Net_Worth_Data.xlsx')
    X, Y, _, _ = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, Y)
    # Check proportions for train-test split
    assert X_train.shape[0] / X.shape[0] == 0.8
    assert X_test.shape[0] / X.shape[0] == 0.2
    assert y_train.shape[0] / Y.shape[0] == 0.8
    assert y_test.shape[0] / Y.shape[0] == 0.2