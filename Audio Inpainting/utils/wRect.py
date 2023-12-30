import numpy as np

def wRect(L: int) -> np.ndarray:
    """
    Generate a rectangular window with length L.

    Args:
        L (int): Window length.

    Returns:
        np.ndarray: The generated rectangular window.
    """
    # Rectangular window with length L
    w = np.ones(L)
    return w
