import numpy as np

def relu(net, prime=False):
    if prime:
        return np.heaviside(net, 0)
    net[net < 0] = 0
    return net

def sigmoid(net, prime=False):
    out = 1 / (1 + np.exp(-net))
    if prime:
        return out * (1 - out)
    return out