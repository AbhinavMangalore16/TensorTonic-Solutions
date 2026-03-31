import numpy as np

def sigmoid(x): 
    """ Vectorized sigmoid function. """ 
    x = np.array(x)
    expp = np.exp(-x)
    return 1/(1+expp)