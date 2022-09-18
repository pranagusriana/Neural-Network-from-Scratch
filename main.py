import numpy as np
from PIL import Image
from CNN.layers import Conv2D, Flatten, Dense
from CNN.lemah import Sequential

img_size = 150
img = np.array(Image.open("./Data/train/cats/cat.1.jpg").resize((img_size, img_size)), dtype=int)
model = Sequential(
    [
        Conv2D(32, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max', input_shape=(img_size, img_size, 3)),
        Conv2D(3, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ]
)
out = model(img.reshape(-1, img_size, img_size, 3)/255)
print("output", out)