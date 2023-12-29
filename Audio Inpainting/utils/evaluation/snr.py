import numpy as np

def SNR(xRef, xEst):
    # Signal to noise ratio (SNR)
    # Add eps to avoid NaN/Inf values
    snr = 10 * np.log10((np.sum(np.abs(xRef)**2) + np.finfo(float).eps) / (np.sum(np.abs(xRef - xEst)**2) + np.finfo(float).eps))
    return snr
