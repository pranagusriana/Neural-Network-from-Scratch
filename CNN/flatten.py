import numpy as np
from PIL import Image

class Flatten:
   def __init__(self,input_shape: int):
      self.input_shape = input_shape

   def flattening(self):
      return self.input_shape.flatten()

#Test
img = np.array(Image.open("../Data/train/cats/cat.1.jpg").resize((150, 150)), dtype=int)
model = Flatten(img)
print(model.flattening())