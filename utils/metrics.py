import numpy as np

def mean_squared_error(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    return np.mean((y_true - y_predict)**2)

def root_mean_squared_error(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    return np.sqrt(np.mean((y_true - y_predict)**2))

def mean_absolute_error(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    return np.mean(np.abs(y_true - y_predict))

def r2_score(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    RSS = np.sum((y_true - y_predict)**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)

    return 1 - (RSS/TSS)

def mean_absolute_percentage_error(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    #Epsilon needed to remove zero division error
    epsilon = 1e-10
    y_true_epsilon = np.where(y_true == 0, epsilon, y_true)

    return np.mean(np.abs(((y_true - y_predict) / y_true_epsilon) * 100))