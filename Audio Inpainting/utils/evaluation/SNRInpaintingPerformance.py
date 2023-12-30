import numpy as np
from snr import SNR
from typing import List, Tuple

def SNRInpaintingPerformance(
    xRef: np.ndarray, xObs: np.ndarray, 
    xEst: np.ndarray, IMiss: np.ndarray, 
    DISP_FLAG: int = 0) -> Tuple[List[float], List[float]]:
    """
    Compute various SNR measures for inpainting performance.

    Args:
        xRef (np.ndarray): Reference signal.
        xObs (np.ndarray): Observed signal.
        xEst (np.ndarray): Estimate signal.
        IMiss (np.ndarray): Location of missing data.
        DISP_FLAG (int, optional): Flag to display SNR on all samples/clipped samples. Defaults to 0.

    Returns:
        Tuple[List[float], List[float]]: A tuple of two lists.
            - SNRAll (List[float]): SNRAll[0] is the original SNR, between xRef and xObs; SNRAll[1] is the obtained SNR, between xRef and xEst.
            - SNRmiss (List[float]): The same as SNRAll but computed on the missing/restored samples only.
    """
    pass
    # Various SNR measures for inpainting performance
    SNRAll = [SNR(xRef, xObs), SNR(xRef, xEst)]
    SNRmiss = [SNR(xRef[IMiss], xObs[IMiss]), SNR(xRef[IMiss], xEst[IMiss])]

    if DISP_FLAG > 0:
        print('SNR on all samples / clipped samples:')
        print(f'Original: {SNRAll[0]} dB / {SNRmiss[0]} dB')
        print(f'Estimate: {SNRAll[1]} dB / {SNRmiss[1]} dB')

    return SNRAll, SNRmiss
