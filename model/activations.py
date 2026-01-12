import numpy as np

class activations_math :
    @staticmethod
    def linear(z):
        a = z
        return a
    # dervitive of it
    @staticmethod
    def linear_backward(z):
        da_dz = 1
        return da_dz
    
    @staticmethod
    def relu(z):
       a = np.maximum(0, z)
       return a
    # erivtive of it
    @staticmethod
    def relu_backward(z):
        da_dz = (z > 0).astype(float)
        return da_dz
    
    @staticmethod
    def sigmoid(z):
        a = 1 / (1 + np.exp(-z))
        return a
    # derivtive of it
    @staticmethod
    def sigmoid_backward(a):
        da_dz = a * (1 - a)
        return da_dz
    
    @staticmethod
    def tanh(z):
        a =  np.tanh(z)
        return a
    @staticmethod
    def tanh_backward(a):
        da_dz = 1 - a**2
        return da_dz