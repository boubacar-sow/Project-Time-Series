from typing import Dict, Any
import numpy as np
from cvxpy import Variable, Minimize, Problem, norm
from scipy.sparse import csc_matrix
import cvxpy as cp

def inpaintFrame_beamOMP_Gabor(problemData: Dict[str, np.ndarray], param: Dict[str, Any]) -> np.ndarray:
    """
    Inpainting method based on Orthogonal Matching Pursuit (OMP) using the Gabor dictionary. 
    The method jointly selects cosine and sine atoms at the same frequency.

    Args:
        problemData (dict): A dictionary containing the observed signal to be inpainted and the indices of clean samples.
            - 'x' (np.ndarray): Observed signal to be inpainted.
            - 'Imiss' (np.ndarray): Indices of clean samples.
        param (dict): A dictionary containing the dictionary matrix (optional if param.D_fun is set), a function handle 
        that generates the dictionary matrix param.D if param.D is not given, and the analysis window.

    Returns:
        np.ndarray: Estimated frame.
    """
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
    indxs = [[]]*4
    currResNorm2 = E2M * 2
    currResNorms = [currResNorm2]*4
    residuals = [residual]*4
    j = 0
    while min(currResNorms) > E2M and j < maxNumCoef:
        j += 1
        temp_indxs = []
        temp_res = []
        temp_curr = []
        for b in range(4) :
            residual = residuals[b]
            proj = residual.T @ Dict
            proj1 = proj[:len(proj)//2]
            proj2 = proj[len(proj)//2:]

            alpha_j = (proj1 - Dict1Dict2 * proj2) * n12
            beta_j = (proj2 - Dict1Dict2 * proj1) * n12

            err_j = np.sum(np.abs(np.outer(residual, np.ones(Dict1.shape[1])) - Dict1 @ np.diag(alpha_j) - Dict2 @ np.diag(beta_j))**2, axis=0)
            poss = np.argsort(err_j)[:4]
            
            for pos in poss :
                indx = indxs[b].copy()
                indx.append(pos)
                indx.append(pos + Dict1.shape[1])

                a = np.linalg.pinv(Dict[:, indx[:2*j]]) @ xObs
                residual = xObs - Dict[:, indx[:2*j]] @ a
                currResNorm2 = np.sum(residual**2)
                temp_indxs.append(indx)
                temp_res.append(residual)
                temp_curr.append(currResNorm2)
        to_add = np.argsort(np.array(temp_curr))[:4]
        indxs = list(np.array(temp_indxs)[to_add])
        indxs = [list(i) for i in indxs]
        currResNorms = list(np.array(temp_curr)[to_add])
        residuals = list(np.array(temp_res)[to_add])

    # argm = np.argmin(np.array(currResNorms))
    # print(argm)
    indx = indxs[0]
    currResNorm = currResNorms[0]
    residual = residuals[0]

        
    indx = indx[:len(a)]
    Coeff = csc_matrix((param['D'].shape[1], 1))
    if len(indx) > 0:
        Coeff[indx, 0] = a.flatten()
        W = W.reshape(-1, 1)
        Coeff = csc_matrix(W * Coeff.toarray())
    y = param['D'] @ Coeff

    return y
