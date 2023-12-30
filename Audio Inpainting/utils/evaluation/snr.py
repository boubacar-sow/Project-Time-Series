import numpy as np

import numpy as np

def SNR(xRef: np.ndarray, xEst: np.ndarray) -> float:
    """
    Compute the Signal-to-Noise Ratio (SNR).

    Args:
        xRef (np.ndarray): Reference signal.
        xEst (np.ndarray): Estimate signal.

    Returns:
        float: The computed SNR.
    """
    # Signal to noise ratio (SNR)
    # Add eps to avoid NaN/Inf values
    snr = 10 * np.log10((np.sum(np.abs(xRef)**2) + np.finfo(float).eps) / (np.sum(np.abs(xRef - xEst)**2) + np.finfo(float).eps))
    return snr
