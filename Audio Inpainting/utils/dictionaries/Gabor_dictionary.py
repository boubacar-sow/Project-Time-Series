import numpy as np
from wSine import wSine

def Gabor_Dictionary(param=None):
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
