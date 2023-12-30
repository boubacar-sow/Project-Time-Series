import numpy as np
from utils.wSine import wSine
from typing import Dict, Any, Optional

def Gabor_Dictionary(param: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Generate a windowed Gabor dictionary. In this implementation, the dictionary matrix is the concatenation of a DCT 
    (left part of the matrix) and of a DST (right part).

    Args:
        param (dict, optional): A dictionary containing optional parameters such as frame length, redundancy factor 
        to adjust the number of frequencies, and weighting window function. Defaults to None.

    Returns:
        np.ndarray: The generated Gabor dictionary (cosine atoms followed by sine atoms).
    """
    pass

    defaultParam = {'N': 256, 'redundancyFactor': 1, 'wd': wSine}
    if param is None:
        param = defaultParam
    else:
        for k in defaultParam.keys():
            if k not in param or param[k] is None:
                param[k] = defaultParam[k]
    K = param['N'] * param['redundancyFactor']
    wd = param['wd']
    u = np.arange(param['N'])
    k = np.arange(K//2)
    D = np.diag(wd) @ np.hstack([np.cos(2*np.pi/K * (np.outer(u, k) + 0.5)), np.sin(2*np.pi/K * (np.outer(u, k) + 0.5))])
    D = D @ np.diag(1./np.sqrt(np.diag(D.T @ D)))
    return D
