from PIL import Image
import numpy as np
import os
import math
from sklearn.utils import shuffle

def load_image(path, img_size):
    return np.array(Image.open(path).resize((img_size, img_size)), dtype=int)

def data_gen(path, img_size, batch_size, random_state=None):
    labels = os.listdir(path)
    id = 0
    map_label = {}
    X_filenames = []
    y = []
    for label in labels:
        map_label[label] = id
        label_path = os.path.join(path, label)
        filenames = os.listdir(label_path)
        for filename in filenames:
            file_path = os.path.join(label_path, filename)
            X_filenames.append(file_path)
            y.append(id)
        id += 1
    n_step = math.ceil(len(X_filenames)/batch_size)
    X_filenames, y = shuffle(X_filenames, y, random_state=random_state)
    ib = 0
    X = []
    Y = []
    X_batch = []
    Y_batch = []
    for i in range(len(X_filenames)):
        if ib == batch_size:
            X.append(X_batch)
            Y.append(Y_batch)
            X_batch = []
            Y_batch = []
            ib = 0
        X_batch.append(load_image(X_filenames[i], img_size))
        Y_batch.append([y[i]])
        ib += 1
        
        # if (i == len(X_filenames) - 1):
        #     X.append(X_batch)
        #     Y.append(Y_batch)

    return np.array(X)/255, np.array(Y), map_label
