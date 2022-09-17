import numpy as np
from PIL import Image
from math import e

class Dense:
    #jumlah unit buat apa ya??
   def __init__(self,input,num_of_units:int,activation_function:str,input_size,output_size):
      self.num_of_units = num_of_units
      self.input = input
      self.activation_function = activation_function
      self.weights = np.random.randn(output_size,input_size)
      self.bias = np.random.randn(output_size,1)

   def computeY(self):
        return np.dot(self.weights,self.input) + self.bias

#    def relu():
#         return max(0,)

#    def sigmoid():
#         return 1/(1+e**())
#Test
img = np.array(Image.open("../Data/train/cats/cat.1.jpg").resize((150, 150)), dtype=int)
model = Dense(img,1,"relu",150,150)
print("weights")
print(model.weights)
print("bias")
print(model.bias)
print(model.computeY())