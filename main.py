import numpy as np
from PIL import Image
import PIL
import math
import matplotlib.pyplot as plt
from CNN.layers import Conv2D

img = np.array(Image.open("Data/train/cats/cat.1.jpg").resize((150, 150)), dtype=int)
model = Conv2D(3, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_mode='average', input_shape=(150, 150, 3))
out = model(img.reshape(-1, 150, 150, 3))
model2 = Conv2D(3, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_mode='average', input_shape=out[0].shape)
out = model2(out)
print(out, out.shape)
plt.imshow(out[0].astype(int))
plt.show()