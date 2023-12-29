import numpy as np
from scipy.sparse import csc_matrix

def inpaintFrame_OMP_Gabor(problemData, param):
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
    Dict1 = Dict[:, :Dict.shape[1]//2]
    Dict2 = Dict[:, Dict.shape[1]//2:]
    Dict1Dict2 = np.sum(Dict1 * Dict2, axis=0)
    n12 = 1 / (1 - Dict1Dict2**2)
    xObs = x[IObs]

    residual = xObs
    maxNumCoef = param['sparsityDegree']
    indx = []
    currResNorm2 = E2M * 2
    j = 0
    while currResNorm2 > E2M and j < maxNumCoef:
        j += 1
        proj = residual.T @ Dict
        proj1 = proj[:len(proj)//2]
        proj2 = proj[len(proj)//2:]

        alpha_j = (proj1 - Dict1Dict2 * proj2) * n12
        beta_j = (proj2 - Dict1Dict2 * proj1) * n12

        err_j = np.sum(np.abs(np.outer(residual, np.ones(Dict1.shape[1])) - Dict1 @ np.diag(alpha_j) - Dict2 @ np.diag(beta_j))**2, axis=0)
        pos = np.argmin(err_j)

        indx.append(pos)
        indx.append(pos + Dict1.shape[1])
        a = np.linalg.pinv(Dict[:, indx[:2*j]]) @ xObs
        residual = xObs - Dict[:, indx[:2*j]] @ a
        currResNorm2 = np.sum(residual**2)

    indx = indx[:len(a)]

    Coeff = csc_matrix((param['D'].shape[1], 1))
    if len(indx) > 0:
        Coeff[indx] = a
        Coeff = W * Coeff

    y = param['D'] @ Coeff

    return y
