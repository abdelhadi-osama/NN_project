from .layer import DenseLayer

class MLP():

    def __init__(self,layer_sizes,activation=None, l_lambda = 0.0001,regularization ='L2',dropout_rates=None,optimizer=None):
        self.layer_sizes =  layer_sizes
        self.activation = activation or ['relu'] * (len(layer_sizes)-2) + ['sigmoid']
        self.l_lambda = l_lambda
        self.regularization = regularization


        # default to adam if not provided 
        if optimizer is None :
            self.optimizer = optimizer(method= 'adam' ,lr=0.001)
        else :
            self.optimizer = optimizer

        #This code ensures that even if you forget to specify dropout, the network creates a "safe" list of zeros so the code doesn't crash later when it tries to access
        if dropout_rates is None:
            dropout_rates = [0.0] * (len(layer_sizes)-1)
        self.dropout_rates = dropout_rates

        self.layers = []
        for i in range(len(self.layer_sizes)-1):
            # --- SAFE DROPOUT LOGIC ---
            # 1. Default to 0.0
            rate = 0.0
            # 2. If list exists AND we haven't run out of values, use the provided rate
            if self.dropout_rates is not None and i < len(self.dropout_rates):
                rate = self.dropout_rates[i]
            layer = DenseLayer( 
                self.layer_sizes[i],
                self.layer_sizes[i+1],
                self.activation[i], 
                self.l_lambda, 
                self.regularization,
                dropout_rate=rate# <--- Pass dropout rate values one by one here 
                )
            self.layers.append(layer)
    
    def forward(self,x,traning=True):
        self.x = x 
        current_output = x

        for layer in self.layers:
            current_output = layer.forward(current_output,traning=traning)  
        
        return current_output 
    
    def backward(self , dl_doutput):     # the dl_doutput is the loss of the current_output
        gradients = []
        current_grad =  dl_doutput
        for layer in reversed(self.layers):
            dl_dw , dl_db , dl_dx =  layer.backward(current_grad)
            gradients.append((dl_dw,dl_db))
            current_grad =  dl_dx
        return list(reversed(gradients))

    def update_parameters(self,gradients,t) :# lr = learning rate
        for i , (dl_dw,dl_db) in enumerate(gradients):
            self.optimizer.update(self.layers[i], dl_dw, dl_db, t)