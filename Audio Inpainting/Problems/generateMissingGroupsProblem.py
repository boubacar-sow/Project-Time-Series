import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hann

from typing import Dict, Tuple
import numpy as np

def GenerateHoles(
    x: np.ndarray, 
    length: float, Nholes = 10 , freq = 8000 ,  
    GR: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Generate a clipping problem: normalize and clip a signal.

    Args:
        x (np.ndarray): Input signal (may be multichannel).
        clippingLevel (float): Clipping level, between 0 and 1.
        GR (bool, optional): Flag to generate an optional graphical display. Defaults to False.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple of two dictionaries.
            - The first dictionary (problemData) contains:
                - 'x' (np.ndarray): Clipped signal.
                - 'IMiss' (np.ndarray): Boolean vector (same size as problemData['x']) that indexes clipped samples.
                - 'clipSizes' (np.ndarray): Size of the clipped segments (not necessary for solving the problem).
            - The second dictionary (solutionData) contains:
                - 'xClean' (np.ndarray): Clean signal (input signal after normalization).
    """

    # Normalize and clip a signal.
    xMax = 0.9999
    solutionData = {'xClean': x / np.max(np.abs(x)) * xMax}

    missing_length = int(length*freq/1000)
    Rest_interval = int((len(x) - missing_length*Nholes)/(Nholes+1))
    Imiss_idxs = np.array([np.arange(i,i+missing_length) for i in range(Rest_interval,len(x)-missing_length,missing_length+Rest_interval)]).flatten()
    print(Imiss_idxs.max())
    Imiss = np.array([False]*len(x))
    Imiss[Imiss_idxs] = True
    problemData = {'x': solutionData['xClean'] * (1- Imiss)}
    problemData['IMiss'] = Imiss  # related indices



    return problemData, solutionData
