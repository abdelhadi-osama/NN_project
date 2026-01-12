import numpy as np

class W_Initializer:
    
    # --- 1. Specific Initialization Methods ---
    @staticmethod
    def random(input_n, output_n, scale=0.01):
        w = np.random.randn(input_n, output_n) * scale
        return w

    @staticmethod
    def xavier(input_n, output_n):
        scale = np.sqrt(2.0 / (input_n + output_n))
        w = np.random.randn(input_n, output_n) * scale
        return w
    
    @staticmethod
    def he(input_n, output_n):
        scale = np.sqrt(2.0 / input_n) 
        w = np.random.randn(input_n, output_n) * scale
        return w

    # --- 2. The Auto-Selector (Router) ---
    @staticmethod
    def auto_w_initializer(n_input, n_neurons, activation): 
        if activation == 'relu':
            return W_Initializer.he(n_input, n_neurons)
        elif activation in ['sigmoid', 'tanh', 'softmax']:
            return W_Initializer.xavier(n_input, n_neurons)
        else:
            return W_Initializer.he(n_input, n_neurons)