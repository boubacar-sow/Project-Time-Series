import numpy as np

def wSine(L: int) -> np.ndarray:
    """
    Generate a symmetric sine window with length L.

    Args:
        L (int): Window length.

    Returns:
        np.ndarray: The generated symmetric sine window.
    """    
    # Symmetric sine window with length L
    w = np.sin((np.arange(L) + 0.5) / L * np.pi)
    return w
