import numpy as np
from PIL import Image
from CNN.layers import Conv2D, Flatten, Dense
from CNN.lemah import Sequential

img_size = 150
cat_img = np.array(Image.open("./Data/test/cats/cat.38.jpg").resize((img_size, img_size)), dtype=int)
dog_img = np.array(Image.open("./Data/test/dogs/dog.4.jpg").resize((img_size, img_size)), dtype=int)
# model = Sequential(
#     [
#         Conv2D(16, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max', input_shape=(img_size, img_size, 3)),
#         Conv2D(32, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max'),
#         Conv2D(64, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max'),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ]
# )
#out = model(np.array([cat_img, dog_img])/255)
#print(out)

# TEST SAVE & LOAD
#model.save('test.pickle')
test_model = Sequential()
test_model.load('test.pickle')
print(test_model.layers)
out_test = test_model(np.array([cat_img, dog_img])/255)
print(out_test)