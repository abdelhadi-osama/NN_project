import numpy as np

class loss :

    @staticmethod
    def BCE(a,y):
        epsilon = 1e-15
        a = np.clip(a, epsilon, 1 - epsilon)
        l = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
        return l 
    # backward of BCE
    @staticmethod
    def BCE_backward(a,y):
        n = a.shape[0]
        eps = 1e-15
        dl_da = (1 / n) * (a - y) / (a * (1 - a) + eps)
        return dl_da
    

    @staticmethod
    def MSE(a,y):
        l = np.mean((y - a)**2)
        return l

    # bakward of MSE is here
    @staticmethod
    def MSE_backward(a,y):
        n =  a.shape[0]
        dl_da = (2 / n) * (a- y)
        return dl_da