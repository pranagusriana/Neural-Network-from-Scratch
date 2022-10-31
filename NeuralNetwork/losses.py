import numpy as np

def mse(y_true, y_pred, prime=False):
    if prime:
        return -1 * (y_true - y_pred)
    return (0.5 * (y_true - y_pred) ** 2).mean()