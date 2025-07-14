import numpy as np

def divide_by_maximum(X):
    max_abs = np.max(np.abs(X), axis=0) #axis=0 used for column-wise searching in the event X is a multi dimensional array.
    max_abs[max_abs == 0] = 1 #Avoid zero division error in unlikely event the max magnitude for a column is 0.
    scaled = X / max_abs #max_abs used to scale X by the largest magnitude rather than the largest number.
    return scaled

def mean_normalisation(X):
    mean_X = np.mean(X, axis=0)
    max_X = np.max(X, axis=0)
    min_X = np.min(X, axis=0)
    range_X = max_X - min_X
    range_X[range_X == 0] = 1 #Avoid zero division error by assigning range_X = 1 where values are consistent in the feature.
    scaled = (X - mean_X) / range_X
    return scaled

def z_score_normalisation(X):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    std_X[std_X == 0] = 1 #Edge case where standard deviation of feature X = 0.
    scaled = (X - mean_X) / std_X
    return scaled

def min_max_normalisation(X):
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    range_X = max_X - min_X
    range_X[range_X == 0] = 1 #Handles edge case where range of feature X = 0.
    scaled = (X - min_X) / range_X
    return scaled

def robust_scaling(X):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1 #Handles edge case where inter quartile range is 0.
    median_X = np.median(X, axis=0)
    scaled = (X - median_X) / iqr
    return scaled