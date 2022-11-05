import pickle
import numpy as np
from tabulate import tabulate

class Sequential:
    """
    args:
        layers: Optional list of layers to add to the model.
    NOTE:
        First layers should have input_shape defined
    """
    def __init__(self,
                layers: list = []):
        self.build_stat = False
        self.compile_stat = False
        self.layers = []
        if (len(layers) != 0):
            for layer in layers:
                self.add(layer)

    def build(self):
        if (len(self.layers) > 1):
            for i in range(1, len(self.layers)):
                self.layers[i].add_input_shape(self.layers[i-1].getOutputShape())
            self.build_stat = True

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.compile_stat = True

    def add(self, layer):
        self.layers.append(layer)
        self.build_stat = False
        self.build()

    def __call__(self, batch_data):
        out = batch_data
        for layer in self.layers:
            out = layer(out)
        return out

    def fit(self, X, Y, epochs=1):
        for e in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                output = self.__call__(np.array(x))
                error += self.loss(np.array(y), output)
                grad = self.loss(np.array(y), output, prime=True)
                for layer in reversed(self.layers):
                    grad = layer.backward(self.optimizer, grad)
            print(f"{e + 1}/{epochs}, error={error}")
    
    def save(self, filename):
        pickle_out = open(filename, "wb")
        pickle.dump(self.layers,pickle_out)
        pickle_out.close()
    
    def load(self, filename):
        pickle_in = open(filename, "rb")
        self.layers = pickle.load(pickle_in)

    def summary(self):
        headers = ["Layer (type)", "Output Shape", "Param #"]
        myData = []
        trainable_param  = 0
        for layer in self.layers:
            myData.append((layer.__class__.__name__,layer.getOutputShape(),layer.getNumberofWeights()))
            trainable_param += layer.getNumberofWeights()
        print(tabulate(myData, headers=headers))
        print ("Total params : ",trainable_param)
        print ("Trainable params : ",trainable_param)
        print ("Non-trainable params : ",0)