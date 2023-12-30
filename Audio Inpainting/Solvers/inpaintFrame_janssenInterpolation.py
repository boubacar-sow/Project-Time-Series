import numpy as np
from scipy.linalg import hankel, cholesky
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import yule_walker

from typing import Dict, Any

def inpaintFrame_janssenInterpolation(problemData: Dict[str, np.ndarray], param: Dict[str, Any]) -> np.ndarray:
    """
    Frame-level inpainting method based on the linear prediction by Janssen.

    Args:
        problemData (dict): A dictionary containing the observed signal to be inpainted and the indices of clean samples.
            - 'x' (np.ndarray): Observed signal to be inpainted.
            - 'Imiss' (np.ndarray): Indices of clean samples.
        param (dict): A dictionary containing the order of the autoregressive model used for linear prediction.

    Returns:
        np.ndarray: Estimated frame.
    """

    s = problemData['x'].copy()
    N = len(s)
    Im = np.where(problemData['IMiss'])[0]
    IObs = np.where(~problemData['IMiss'])[0]
    M = len(Im)
    Im = np.sort(Im)

    if 'p' not in param:
        p = min(3 * M + 2, round(N / 3))
    else:
        p = param['p']
    if 'GR' not in param:
        param['GR'] = False
    if 'NIt' not in param:
        NIt = 100
    else:
        NIt = param['NIt']

    IAA = np.abs(np.outer(Im, np.ones(N)) - np.outer(np.ones(M), np.arange(1, N + 1)))
    IAA1 = IAA <= p
    AA = np.zeros_like(IAA)

    if param['GR']:
        plt.figure()
        plt.hold(True)

    for k in range(NIt):
        # Re-estimation of LPC
        aEst = yule_walker(s, order=p)[0]

        # Re-estimation of the missing samples
        b = aEst @ hankel(aEst, np.concatenate([aEst[-1:], np.zeros(p)]))
        AA[IAA1] = b[IAA[IAA1] + 1]
        try:
            R = cholesky(AA[IAA1, Im])
            xEst = -np.linalg.solve(R, np.linalg.solve(R.T, AA[IAA1, IObs] @ s[IObs]))
        except np.linalg.LinAlgError:
            xEst = -np.linalg.solve(AA[IAA1, Im], AA[IAA1, IObs] @ s[IObs])
        s[Im] = xEst
        if param['GR']:
            e = lfilter(aEst, 1, s)
            plt.plot(k, 10 * np.log10(np.mean(e[p:]**2)), 'o')

    y = s

    return y
