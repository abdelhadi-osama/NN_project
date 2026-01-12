import numpy as np

class LearningRateScheduler :
    """Various learning rate scheduling strategies"""
    @staticmethod
    def constant(initial_lr, epoch,**kwargs):
        
        return initial_lr

    @staticmethod
    def step_decay(initial_lr,epoch,step_size=30,gamma=0.1,**kwargs):

        return initial_lr * (gamma **(epoch//step_size))
    
    @staticmethod
    def exponential_decay(initial_lr,epoch,decey_rate=0.01,**kwargs):

        return initial_lr * np.exp(-decey_rate * epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr,epoch,total_epochs=100,**kwargs):
        
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))