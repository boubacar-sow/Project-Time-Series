import numpy as np
from scipy.linalg import hankel, cholesky
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import yule_walker
from librosa.core import lpc

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

    s = problemData['x']
    N = len(s)
    Im = np.where(problemData['IMiss'])[0]
    IObs = np.where(~problemData['IMiss'])[0]
    M = len(Im)
    Im = np.sort(Im)
    s[Im] = 0

    if param is None or 'p' not in param:
        p = min(3*M+2, round(N/3))
    else:
        p = param['p']
    if param is None or 'GR' not in param:
        param = {'GR': False}
    if param is None or 'NIt' not in param:
        NIt = 100
    else:
        NIt = param['NIt']

    IAA = np.abs(np.outer(Im, np.ones(N)) - np.outer(np.ones(M), np.arange(N)))
    IAA1 = IAA <= p
    AA = np.zeros_like(IAA)

    if param['GR']:
        plt.figure()
        plt.hold(True)

    for k in range(NIt):
        aEst = np.atleast_1d(lpc(s, order=p))
        b = np.dot(aEst.T, hankel(aEst, np.concatenate((aEst[-1:], np.zeros(p)))))
        AA[IAA1] = b[IAA[IAA1].astype(int)]
        try:
            R = cholesky(AA[:, Im])
            xEst = -np.linalg.solve(R, np.linalg.solve(R.T, np.dot(AA[:, IObs], s[IObs])))
        except np.linalg.LinAlgError:
            xEst = -np.linalg.inv(AA[:, Im]).dot(AA[:, IObs]).dot(s[IObs])
        s[Im] = xEst
        if param['GR']:
            e = lfilter(aEst, 1, s)
            plt.plot(k, 10*np.log10(np.mean(e[p:]**2)), 'o')

    return s
