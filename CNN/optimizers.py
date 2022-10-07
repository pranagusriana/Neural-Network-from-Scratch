import numpy as np

class SGD:
    def __init__(self, learning_rate=1e-1, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0

    def update(self, weight, gradient):
        print(weight.shape, gradient.shape)
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        updated_weight = weight + self.velocity
        return updated_weight