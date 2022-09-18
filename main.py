import numpy as np
from PIL import Image
import PIL
import math
import matplotlib.pyplot as plt
from CNN.layers import Conv2D, Flatten, Dense

img = np.array(Image.open("Data/train/cats/cat.1.jpg").resize((150, 150)), dtype=int)
img_size = 150
img = np.array(Image.open("./Data/train/cats/cat.1.jpg").resize((img_size, img_size)), dtype=int)
# Conv 1
conv1 = Conv2D(32, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max', input_shape=(img_size, img_size, 3))
out = conv1(img.reshape(-1, img_size, img_size, 3)/255)
print(conv1.getOutputShape(), conv1.getNumberofWeights())

# Conv 2
conv2 = Conv2D(3, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max', input_shape=out[0].shape)
out = conv2(out)
print(conv2.getOutputShape(), conv2.getNumberofWeights())

# Flatten
f = Flatten(input_shape=out[0].shape)
out = f(out)
print(f.getOutputShape(), f.getNumberofWeights())

# Dense
d = Dense(1, activation='sigmoid', input_shape=out[0].shape)
out = d(out)
print(d.getOutputShape(), d.getNumberofWeights())

# Output
print("output", out)