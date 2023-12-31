import numpy as np
from utils.wSine import wSine
from typing import Dict, Any, Optional

def DCT_Dictionary(param: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Generate a windowed DCT dictionary.

    Args:
        param (dict, optional): A dictionary containing optional parameters such as frame length, 
                                redundancy factor to adjust the number of frequencies, and weighting window function. 
                                Defaults to None.

    Returns:
        np.ndarray: The generated DCT dictionary.
    """

    defaultParam = {'N': 256, 'redundancyFactor': 1, 'wd': wSine}
    if param is None:
        param = defaultParam
    else:
        for k in defaultParam.keys():
            if k not in param or param[k] is None:
                param[k] = defaultParam[k]
    K = param['N'] * param['redundancyFactor']
    wd = param['wd'](param['N'])
    u = np.arange(param['N'])
    k = np.arange(K)
    D = np.diag(wd) @ np.cos(np.pi/K * (np.outer(u, k) + 0.5))
    D = D @ np.diag(1./np.sqrt(np.diag(D.T @ D)))
    return D


