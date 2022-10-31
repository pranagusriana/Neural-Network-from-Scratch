import numpy as np

def KFold(n_splits,data):
    np.split(data,n_splits)
    