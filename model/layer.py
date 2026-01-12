import numpy as np
from .initializer import W_Initializer
from .activations import activations_math

class DenseLayer():
    def __init__(self,n_input,n_neurons,activation='relu', l_lambda = 0.0001,regularization ='L2',dropout_rate=0.0):  #we are using l2 because in l2 foumela l = l + 2*lambda/m * sum(w**2) in make hte w bigger
        self.w = W_Initializer.auto_w_initializer(n_input,n_neurons,activation)
        self.b = np.zeros((1,n_neurons))
        self.activation =activation
        self.l_lambda = l_lambda
        self.regularization = regularization

        #dropout attributes
        self.dropout_rate=dropout_rate
        self.mask = None

    def forward(self,x,traning=True):
        self.x = x
        self.z = x @ self.w + self.b
        # activation function:
        if self.activation == 'linear':
            self.a = activations_math.linear(self.z)
        elif self.activation == 'relu':
            self.a = activations_math.relu(self.z)
        elif self.activation == 'sigmoid':
            self.a = activations_math.sigmoid(self.z)
        elif self.activation == 'tanh':
            self.a = activations_math.tanh(self.z)

        #Apply Inverted Dropout (Only during training & if rate > 0)
        if traning and self.dropout_rate > 0.0 :
            keep_prob = 1 - self.dropout_rate      # we caluculate keep_prob to keep the droping according to the 1's probebilty 
            # generate the mask
            self.mask = (np.random.rand(*self.a.shape) < keep_prob) / keep_prob # wthout * the output of shape is (number , number) with * the output will be number number and we can treat it like an argument

            #droping neurons form a 
            self.a *= self.mask
        else:
            self.mask = None

        return self.a

    def backward(self , dl_da):
        #appling the dropout on the gridients
        # If we killed a neuron in forward, it cannot transmit a gradient backward
        if self.mask is not None:
            dl_da *= self.mask

        size = self.x.shape[0] # we use as m for regularizations
        if self.activation == 'linear':
            da_dz = activations_math.linear_backward(self.z)
        elif self.activation == 'relu':
            da_dz = activations_math.relu_backward(self.z)
        elif self.activation == 'tanh':
            da_dz = activations_math.tanh_backward(self.a)
        elif self.activation == 'sigmoid':
            da_dz = activations_math.sigmoid_backward(self.a)  
        dl_dz = dl_da * da_dz
        dl_dw = (self.x.T @ dl_dz) 
        dl_db = np.sum(dl_dz, axis=0,keepdims=True) 
        dl_dx = dl_dz @ self.w.T

        # Add  regularization
        if self.regularization == 'L2' :  
            dl_dw += ((self.l_lambda * self.w )/size)
        elif self.regularization == 'L1' :
            dl_dw += (self.l_lambda / size) * np.sign(self.w)        
        return dl_dw,dl_db,dl_dx