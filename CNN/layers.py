import numpy as np
from PIL import Image
import PIL
import math

class Conv2D:
    """
    args:
        filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
        kenel_size: An integer or tuple of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        pooling_filter_size: An integer or tuple of 2 integers, specifying the height and width of the pooling window. Can be a single integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
        padding: An integer or tuple of 2 integers, specifying additional padding of the input data along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
        input_shape: Tuple of 3 integers consist of (height, width, channels) of the input data
        pooling_strides: An integer or tuple of 2 integers, specifying the strides of the pooling stage along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
        pooling_mode: A string, specifying the mode of the pooling stage. There are two modes for the pooling stage 'average' and 'max'
    """
    def __init__(self,
                filters: int, 
                kernel_size: tuple[int, int]|int,
                pooling_filter_size: tuple[int, int]|int, 
                strides: tuple[int, int]|int = 1, 
                padding: tuple[int, int]|int = 0,
                input_shape: tuple[int, int, int] = None,
                pooling_strides: tuple[int, int]|int = 2,
                pooling_mode: str = 'average'):

        filters, kernel_size, strides, padding = self._check_params(filters, kernel_size, strides, padding)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        pooling_filter_size, pooling_strides, pooling_mode = self._check_params_pooling(pooling_filter_size, pooling_strides, pooling_mode)
        self.pooling_filter_size = pooling_filter_size
        self.pooling_strides = pooling_strides
        self.pooling_mode = pooling_mode.lower()
        if (input_shape):
            self.add_input_shape(input_shape)

    def _check_params(self, filters, kernel_size, strides, padding):
        if(not(isinstance(filters, int))):
            raise TypeError("Filters should be integers >= 1")
        if (isinstance(kernel_size, int) or isinstance(kernel_size, tuple)):
            if (isinstance(kernel_size, int)):
                kernel_size = (kernel_size, kernel_size)
        else:
            raise TypeError("Kernel size should be integers or tuple")

        if (isinstance(strides, int) or isinstance(strides, tuple)):
            if (isinstance(strides, int)):
                strides = (strides, strides)
        else:
            raise TypeError("Strides should be integers or tuple")

        if (isinstance(padding, int) or isinstance(padding, tuple)):
            if (isinstance(padding, int)):
                padding = (padding, padding)
        else:
            raise TypeError("Padding should be integers or tuple")

        params_are_correct = (isinstance(kernel_size[0], int)   and isinstance(kernel_size[1], int)   and
                            isinstance(strides[0], int)   and isinstance(strides[1], int)   and
                            isinstance(padding[0], int)  and isinstance(padding[1], int)  and
                            strides[0]   >= 1 and strides[1]   >= 1 and 
                            padding[0]  >= 0 and padding[1]  >= 0 and
                            filters >= 1)
        assert params_are_correct, 'Parameters should be integers equal or greater than default values.'
        
        return filters, kernel_size, strides, padding

    def _check_params_pooling(self, filter_size, strides, mode):
        if (isinstance(filter_size, int) or isinstance(filter_size, tuple)):
            if (isinstance(filter_size, int)):
                filter_size = (filter_size, filter_size)
        else:
            raise TypeError("Filter size pooling should be integers or tuple")

        if (isinstance(strides, int) or isinstance(strides, tuple)):
            if (isinstance(strides, int)):
                strides = (strides, strides)
        else:
            raise TypeError("Strides pooling should be integers or tuple")

        if (isinstance(mode, str)):
            if (not(mode.lower() in ['max', 'average'])):
                raise TypeError("Mode pooling should be string 'max' or 'average'")
        else:
            raise TypeError("Mode pooling should be string 'max' or 'average'")

        params_are_correct = (isinstance(filter_size[0], int)   and isinstance(filter_size[1], int)   and
                            isinstance(strides[0], int)   and isinstance(strides[1], int)   and
                            strides[0]   >= 1 and strides[1]   >= 1)
        assert params_are_correct, 'Parameters should be integers equal or greater than default values.'
        
        return filter_size, strides, mode

    def _check_input_shape(self, input_shape):
        assert isinstance(input_shape, tuple), 'Input shape should be tuple of integers with 3 dimensions (height, width, depth)'
        assert len(input_shape) == 3, 'Input shape should have 3 dimensions (height, width, depth)'
        self.height, self.width, self.depth = input_shape
        self.nWeights = self.filters * ((self.kernel_size[0] * self.kernel_size[1] * self.depth) + 1)
        if (((self.height - self.kernel_size[0]) <= 0) and ((self.width - self.kernel_size[1]) <= 0)):
            raise ValueError('Kernel size should be less or equal than input shape')

    def add_input_shape(self, input_shape):
        self._check_input_shape(input_shape)
        self._init_kernels()
        self._calculate_output_shape()

    def _init_kernels(self):
        self.kernels = np.random.randn(self.depth, self.kernel_size[0], self.kernel_size[1], self.filters)
        self.biases = np.random.randn(self.filters)

    def _calculate_output_shape(self):
        H = int((self.height - self.kernel_size[0] + 2 * self.padding[0]) / self.strides[0]) + 1
        W = int((self.width - self.kernel_size[1] + 2 * self.padding[1]) / self.strides[1]) + 1
        self.output_shape = (H, W, self.filters)
        if (((H - self.pooling_filter_size[0]) <= 0) and ((H - self.pooling_filter_size[1]) <= 0)):
            raise ValueError('Pooling filter size should be less or equal than output convolutional stage shape')
        H_pooling = int((H - self.pooling_filter_size[0]) / self.pooling_strides[0]) + 1
        W_pooling = int((W - self.pooling_filter_size[1]) / self.pooling_strides[1]) + 1
        self.output_shape_pooling = (H_pooling, W_pooling, self.filters)

    def _check_input_matrix(self, input_matrix):
        if (not(isinstance(input_matrix, np.ndarray))):
            input_matrix = np.array(input_matrix)
        assert len(input_matrix.shape) == 4, 'Input shape should have 4 dimensions (batch, height, width, depth)'
        batch, height, width, depth = input_matrix.shape
        if (height != self.height or width != self.width or depth != self.depth):
            raise ValueError(f"Expected input shape is (_, {self.height}, {self.width}, {self.depth}) not {input_matrix.shape}")
        return input_matrix

    def _convolution_operation(self, input_data_matrix, kernel):
        output_matrix = np.zeros((self.output_shape[0], self.output_shape[1]))
        input_data_matrix = np.pad(input_data_matrix, ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        h, w = self.output_shape[0], self.output_shape[1]
        for i in range(h):
            for j in range(w):
                output_matrix[i, j] = (input_data_matrix[i * self.strides[0] : i * self.strides[0] + self.kernel_size[0], j * self.strides[1] : j * self.strides[1] + self.kernel_size[1]] * kernel).sum()
        return output_matrix

    def _pooling_operation(self, input_data_matrix):
        output_matrix = np.zeros((self.output_shape_pooling[0], self.output_shape_pooling[1]))
        h, w = self.output_shape_pooling[0], self.output_shape_pooling[1]
        if (self.pooling_mode == 'max'):
            for i in range(h):
                for j in range(w):
                    output_matrix[i, j] = (input_data_matrix[i * self.pooling_strides[0] : i * self.pooling_strides[0] + self.pooling_filter_size[0], j * self.pooling_strides[1] : j * self.pooling_strides[1] + self.pooling_filter_size[1]]).max()
        elif (self.pooling_mode == 'average'):
            for i in range(h):
                for j in range(w):
                    output_matrix[i, j] = (input_data_matrix[i * self.pooling_strides[0] : i * self.pooling_strides[0] + self.pooling_filter_size[0], j * self.pooling_strides[1] : j * self.pooling_strides[1] + self.pooling_filter_size[1]]).mean()
        return output_matrix

    def _convolution_stage(self, input_data):
        input_data = self._check_input_matrix(input_data)
        batch, height, width, depth = input_data.shape
        out_batch = []
        for i in range(batch):
            cur_data = input_data[i]
            out = np.zeros(self.output_shape)
            for d in range(self.depth):
                cur_kernel = self.kernels[d]
                out += np.dstack([self._convolution_operation(cur_data[:, :, d], cur_kernel[:, :, k]) for k in range(self.filters)])
            out = out / self.depth
            out += self.biases
            out_batch.append(out)
        return np.array(out_batch)

    def _detector_stage(self, input_mat):
        input_mat[input_mat < 0] = 0
        return input_mat

    def _pooling_stage(self, input_data):
        batch, height, width, depth = input_data.shape
        out_batch = []
        for i in range(batch):
            cur_data = input_data[i]
            out = np.dstack([self._pooling_operation(cur_data[:, :, d]) for d in range(depth)])
            out_batch.append(out)
        return np.array(out_batch)

    # Method for forward phase, just call the model. Ex: model = Conv2D(...); model(input_data)
    def __call__(self, batch_image):
        out = self._convolution_stage(batch_image)
        out = self._detector_stage(out)
        out = self._pooling_stage(out)
        return out

    def getOutputShape(self):
        return self.output_shape_pooling # Because of the Conv2D included pooling, so the output shape is output shape pooling

    def getNumberofWeights(self):
        return self.nWeights