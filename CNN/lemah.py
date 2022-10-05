import pickle
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
        self.layers = []
        if (len(layers) != 0):
            for layer in layers:
                self.add(layer)

    def build(self):
        if (len(self.layers) > 1):
            for i in range(1, len(self.layers)):
                self.layers[i].add_input_shape(self.layers[i-1].getOutputShape())
            self.build_stat = True

    def add(self, layer):
        self.layers.append(layer)
        self.build_stat = False
        self.build()

    def __call__(self, batch_data):
        out = batch_data
        for layer in self.layers:
            out = layer(out)
        return out
    
    def save(self, filename):
        pickle_out = open(filename, "wb")
        pickle.dump(self.layers,pickle_out)
        pickle_out.close()
    
    def load(self, filename):
        pickle_in = open(filename, "rb")
        self.layers = pickle.load(pickle_in)