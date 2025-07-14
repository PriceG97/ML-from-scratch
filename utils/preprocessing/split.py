import numpy as np

def train_cv_test_split(X, y, train_size=0.6, cv_size=0.2, test_size=0.2, shuffle=True, random_state=None):
    total = train_size + cv_size + test_size
    if np.isclose(total, 1.0):
        raise ValueError(f'Train, CV & Test size must sum to 1. Current = {total}') #Raise error if size is not 1.
    
    samples = len(X)
    indices = arange(samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        
    train_end = int(train_size * n_samples)
    cv_end = train_end + int(cv_size * n_samples)
    
    train_idx = indices[:train_end]
    cv_idx = indices[train_end:cv_end]
    test_idx = indices[cv_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_cv, y_cv = X[cv_idx], y[cv_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return (X_train, y_train), (X_cv, y_cv), (X_test, y_test)