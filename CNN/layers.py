import numpy as np

class Conv2D:
    def __init__(self,
                filters: int, 
                kernel_size: tuple[int]|int, 
                strides: tuple[int]|int = 1, 
                padding: tuple[int]|int = 0,
                input_shape: tuple[int] = None):

        filters, kernel_size, strides, padding = self._check_params(filters, kernel_size, strides, padding)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        if (input_shape):
            self.add_input_shape(input_shape)
            print(self.kernels.shape)
            print(self.biases.shape)
            print(self.output_shape)

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

    def _check_input_shape(self, input_shape):
        assert isinstance(input_shape, tuple), 'Input shape should be tuple of integers with 3 dimensions (height, width, depth)'
        assert len(input_shape) == 3, 'Input shape should have 3 dimensions (height, width, depth)'
        self.height, self.width, self.depth = input_shape
        if (((self.height - self.kernel_size[0]) <= 0) and ((self.width - self.kernel_size[1]) <= 0)):
            raise ValueError('Kernel size should be less or equal than input shape')

    def add_input_shape(self, input_shape):
        self._check_input_shape(input_shape)
        self._init_kernels()
        self._calculate_output_shape()

    def _init_kernels(self):
        # NOTE: Masih ragu terkait nchanel dari kernels (Apakah kalo input chanelnya ada lebih dari 1 kernelnya beda atau sama)
        # Contoh: input (h, w, chanel) dengan filter f ukuran kxk apakah kernelnya jadi shape (f, k, k, chanel)?
        # Asumsi dulu tiap chanel input kernelnya sama
        self.kernels = np.random.rand(self.filters, self.kernel_size[0], self.kernel_size[1])
        # Bias juga perlu disesuaikan lagi
        self.biases = np.random.rand(self.filters)

    def _calculate_output_shape(self):
        H = int((self.height - self.kernel_size[0] + 2 * self.padding[0]) / self.strides[0]) + 1
        W = int((self.width - self.kernel_size[1] + 2 * self.padding[1]) / self.strides[1]) + 1
        self.output_shape = (H, W, self.filters)

model = Conv2D(32, (3, 3), (1, 1), (0, 0), (5, 5, 3))
