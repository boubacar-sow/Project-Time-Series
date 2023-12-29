import numpy as np

def wSine(L):
    # Symmetric sine window with length L
    w = np.sin((np.arange(L) + 0.5) / L * np.pi)
    return w
