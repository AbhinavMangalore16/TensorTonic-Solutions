import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    G = np.array(G)
    g = np.array(g)
    w = np.array(w)
    Gt = G + g*g
    wt = w - (lr/(Gt+eps)**0.5)*g
    return (wt, Gt)