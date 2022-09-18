import numpy as np
from PIL import Image

class Flatten:
    def __init__(self,
                input_shape: int = None):
        if (input_shape):
            self.add_input_shape(input_shape)

    def _check_input_shape(self, input_shape):
        assert isinstance(input_shape, tuple), 'Input shape should be tuple of integers with 3 dimensions (height, width, depth)'
        assert len(input_shape) == 3, 'Input shape should have 3 dimensions (height, width, depth)'
        self.height, self.width, self.depth = input_shape
        self.nWeights = 0

    def _calculate_output_shape(self):
        self.output_shape =  (self.height * self.width * self.depth,)

    def add_input_shape(self, input_shape):
        self._check_input_shape(input_shape)
        self._calculate_output_shape()
    
    def _check_input_data(self, input_data):
        if (not(isinstance(input_data, np.ndarray))):
            input_data = np.array(input_data)
        assert len(input_data.shape) == 4, 'Input shape should have 4 dimensions (batch, height, width, depth)'
        batch, height, width, depth = input_data.shape
        if (height != self.height or width != self.width or depth != self.depth):
            raise ValueError(f"Expected input shape is (_, {self.height}, {self.width}, {self.depth}) not {input_data.shape}")
        return input_data

    def getOutputShape(self):
        return self.output_shape

    def getNumberofWeights(self):
        return self.nWeights

    def __call__(self, batch_data):
        input_data = self._check_input_data(batch_data)
        batch, height, width, depth = input_data.shape
        return np.array([input_data[i].flatten() for i in range(batch)])