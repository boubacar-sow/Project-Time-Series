from typing import Dict, Any
import numpy as np
from cvxpy import Variable, Minimize, Problem, norm
from scipy.sparse import csc_matrix
import cvxpy as cp

def inpaintFrame_consOMP(problemData: Dict[str, np.ndarray], param: Dict[str, Any]) -> np.ndarray:
    """
    Inpainting method based on Orthogonal Matching Pursuit (OMP) with a constraint on the amplitude of the 
    reconstructed samples and an optional constraint on the maximum value of the clipped samples.

    Args:
        problemData (dict): A dictionary containing the observed signal to be inpainted and the indices of clean samples.
            - 'x' (np.ndarray): Observed signal to be inpainted.
            - 'Imiss' (np.ndarray): Indices of clean samples.
        param (dict): A dictionary containing the dictionary matrix (optional if param.D_fun is set), a function handle 
        that generates the dictionary matrix param.D if param.D is not given, the analysis window, and an integer value 
        indicating that an upper limit constraint is active if present and non-empty.

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
    clippingLevelEst = np.max(np.abs(x) / wa)
    IMiss = np.ones(len(x), dtype=bool)
    IMiss[IObs] = False
    IMissPos = np.where((x >= 0) & IMiss)[0]
    IMissNeg = np.where((x < 0) & IMiss)[0]

    DictPos = param['D'][IMissPos, :]
    DictNeg = param['D'][IMissNeg, :]

    wa_pos = wa[IMissPos]
    wa_neg = wa[IMissNeg]
    b_ineq_pos = wa_pos * clippingLevelEst
    b_ineq_neg = -wa_neg * clippingLevelEst
    if 'Upper_Limit' in param and param['Upper_Limit']:
        b_ineq_pos_upper_limit = wa_pos * param['Upper_Limit'] * clippingLevelEst
        b_ineq_neg_upper_limit = -wa_neg * param['Upper_Limit'] * clippingLevelEst
    else:
        b_ineq_pos_upper_limit = np.inf
        b_ineq_neg_upper_limit = -np.inf

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

    Dict = param['D'][IObs, :]
    W = 1. / np.sqrt(np.diag(Dict.T @ Dict))
    Dict = Dict @ np.diag(W)
    xObs = x[IObs]

    residual = xObs
    maxNumCoef = param['sparsityDegree']
    indx = []
    currResNorm2 = E2M * 2  # set a value above the threshold in order to have/force at least one loop executed
    j = 0
    while currResNorm2 > E2M and j < maxNumCoef:
        j += 1
        proj = Dict.T @ residual
        pos = np.argmax(np.abs(proj))
        indx.append(pos)
        a = np.linalg.pinv(Dict[:, indx]) @ xObs
        residual = xObs - Dict[:, indx] @ a
        currResNorm2 = np.sum(residual**2)

    W_cvx = cp.Parameter(shape=(len(W[indx]), 1))
    W_cvx.value = W[indx].reshape(-1, 1)

    a = cp.Variable(j)
    a = cp.reshape(a, (j, 1))

    constraints = []
    if DictPos[:, indx].shape[0] > 0:
        constraints.append(DictPos[:, indx] @ cp.multiply(W_cvx, a) <= b_ineq_pos.reshape(-1, 1))
    if DictNeg[:, indx].shape[0] > 0:
        constraints.append(DictNeg[:, indx] @ cp.multiply(W_cvx, a) >= b_ineq_neg.reshape(-1, 1))
    


    # if np.isinf(b_ineq_pos_upper_limit):
    #     constraints += [DictPos[:, indx] @ (W[indx] * a) <= b_ineq_pos_upper_limit,
    #                     DictNeg[:, indx] @ (W[indx] * a) >= b_ineq_neg_upper_limit]
    xObs = xObs.reshape(-1, 1)
    objective = cp.Minimize(cp.norm(Dict[:, indx] @ a - xObs))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.value > 1e3:
        a = cp.Variable(j)
        objective = cp.Minimize(cp.norm(Dict[:, indx] @ a - xObs))
        prob = cp.Problem(objective)
        prob.solve()

    indx = indx[:len(a.value)]
       
    Coeff = csc_matrix((param['D'].shape[1], 1))
    if len(indx) > 0:
        Coeff[indx, 0] = a.value
        W = W[:, np.newaxis] 

        Coeff = csc_matrix(W * Coeff.toarray())
    y = param['D'] @ Coeff
    return y
