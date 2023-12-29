import numpy as np
from scipy.sparse import csc_matrix

def inpaintFrame_OMP(problemData, param):
    x = problemData['x']
    IObs = np.where(~problemData['IMiss'])[0]
    p_N = len(x)
    E2 = param['OMPerr']**2
    E2M = E2 * len(IObs)
    wa = param['wa'](param['N'])

    if 'D' not in param:
        param['D'] = param['D_fun'](param)

    Dict = param['D'][IObs, :]
    W = 1 / np.sqrt(np.diag(Dict.T @ Dict))
    Dict = Dict @ np.diag(W)
    xObs = x[IObs]

    residual = xObs
    maxNumCoef = param['sparsityDegree']
    indx = []
    currResNorm2 = E2M * 2
    j = 0
    while currResNorm2 > E2M and j < maxNumCoef:
        j += 1
        proj = Dict.T @ residual
        pos = np.argmax(np.abs(proj))
        indx.append(pos)
        a = np.linalg.pinv(Dict[:, indx[:j]]) @ xObs
        residual = xObs - Dict[:, indx[:j]] @ a
        currResNorm2 = np.sum(residual**2)

    indx = indx[:len(a)]

    Coeff = csc_matrix((param['D'].shape[1], 1))
    if len(indx) > 0:
        Coeff[indx] = a
        Coeff = W * Coeff

    y = param['D'] @ Coeff

    return y
