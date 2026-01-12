import numpy as np

class Optimizer:
    def __init__(self,method='adam',lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.method = method.lower()  #for generalization
        self.lr = lr
        self.beta1 = beta1 #momentum factor 
        self.beta2 = beta2 #RMSpropagation factor
        self.epsilon = epsilon
    

    def update(self, layer, dl_dw ,dl_db ,t):

        # ------------------------------------------------
        # 1. SGD (Standard Gradient Descent)
        # ------------------------------------------------
        if self.method == 'sgd':
            # w = w - gradian * lr
            layer.w -= self.lr * dl_dw
            layer.b -= self.lr * dl_db

        # ------------------------------------------------
        # 2. Momentum
        # ------------------------------------------------
        elif self.method == 'momentum' :
            # Initialize Velocity "v" if missing
            if not hasattr(layer, 'v_w') : # hasattr check the objet if the v_w is existing or not
                layer.v_w = np.zeros_like(layer.w)
                layer.v_b = np.zeros_like(layer.b)

                #Standard formula is v = beta*v + (1-beta)*grad
            layer.v_w = self.beta1 * layer.v_w +  (1 - self.beta1) * dl_dw   # i remove (1 - self.beta1) it make the momentum slow and the math of it if we dont remove it  numbers: beta = 0.9 and Learning Rate= 0.01  the Standard Implementation (PyTorch/TensorFlow): v = Beta * v_old + 1.0 * gradient
            layer.v_b = self.beta1 * layer.v_b +   (1 - self.beta1) * dl_db # i remove (1 - self.beta1) in the same reason

                # updates >> w = w - lr * v
            layer.w -= self.lr * layer.v_w
            layer.b -= self.lr * layer.v_b
        
        # ------------------------------------------------
        # 3. RMSProp
        # ------------------------------------------------

        elif self.method == 'rmsprop' :
            
            #init the cashe if missing
            if not hasattr(layer, 's_w'):
                layer.s_w = np.zeros_like(layer.w)
                layer.s_b = np.zeros_like(layer.b)

            # standard formula s = beta2 * s + (1-beta2) * dl_dw**2
            layer.s_w = self.beta2 * layer.s_w + (1 - self.beta2) * (dl_dw**2)
            layer.s_b = self.beta2 * layer.s_b + (1 - self.beta2) * (dl_db**2)
            
            # Update Weights
            #w = w - (lr / ( sqroot(s)+ epsilon ) * dl_dw
            layer.w -= self.lr * dl_dw / (np.sqrt(layer.s_w) + self.epsilon)
            layer.b -= self.lr * dl_db / (np.sqrt(layer.s_b) + self.epsilon)

        
        # ------------------------------------------------
        # 4. Adam (The King)
        # ------------------------------------------------
        elif self.method == 'adam':
            # Initialize both Velocity (v) and Cache (s)
            if not hasattr(layer, 'v_w'):
                layer.v_w = np.zeros_like(layer.w)
                layer.v_b = np.zeros_like(layer.b)
                layer.s_w = np.zeros_like(layer.w)
                layer.s_b = np.zeros_like(layer.b)


            # --- Weights ---
            # 1. Momentum
            layer.v_w = self.beta1 * layer.v_w + (1 - self.beta1) * dl_dw
            # 2. RMSProp
            layer.s_w = self.beta2 * layer.s_w + (1 - self.beta2) * (dl_dw**2)
            # 3. Bias Correction   At  previous steps  v and  s are initialized to 0. This makes the estimates tiny. We fix this by :
            v_w_corr = layer.v_w / (1 - self.beta1**t)
            s_w_corr = layer.s_w / (1 - self.beta2**t)
            # 4. Update
            layer.w -= self.lr * v_w_corr / (np.sqrt(s_w_corr) + self.epsilon)

            # --- Biases ---
            layer.v_b = self.beta1 * layer.v_b + (1 - self.beta1) * dl_db
            layer.s_b = self.beta2 * layer.s_b + (1 - self.beta2) * (dl_db**2)
            
            v_b_corr = layer.v_b / (1 - self.beta1**t)
            s_b_corr = layer.s_b / (1 - self.beta2**t)
            
            layer.b -= self.lr * v_b_corr / (np.sqrt(s_b_corr) + self.epsilon)