from cvxpy import Variable, Minimize, Problem, norm
import numpy as np

def inpaintFrame_consOMP_Gabor(problemData, param):
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

    j = len(indx)
    if np.isinf(b_ineq_pos_upper_limit):
        a = Variable(j)
        objective = Minimize(norm(Dict[:, indx] @ a - xObs))
        constraints = [DictPos[:, indx] @ (W[indx] * a) >= b_ineq_pos,
                       DictNeg[:, indx] @ (W[indx] * a) <= b_ineq_neg]
        prob = Problem(objective, constraints)
        prob.solve()

        if prob.value > 1e3:
            objective = Minimize(norm(Dict[:, indx] @ a - xObs))
            prob = Problem(objective)
            prob.solve()
    else:
        a = Variable(j)
        objective = Minimize(norm(Dict[:, indx] @ a - xObs))
        constraints = [DictPos[:, indx] @ (W[indx] * a) >= b_ineq_pos,
                       DictNeg[:, indx] @ (W[indx] * a) <= b_ineq_neg,
                       DictPos[:, indx] @ (W[indx] * a) <= b_ineq_pos_upper_limit,
                       DictNeg[:, indx] @ (W[indx] * a) >= b_ineq_neg_upper_limit]
        prob = Problem(objective, constraints)
        prob.solve()

        if prob.value > 1e3:
            objective = Minimize(norm(Dict[:, indx] @ a - xObs))
            prob = Problem(objective)
            prob.solve()

    indx = indx[:len(a.value)]

    Coeff = np.zeros((param['D'].shape[1], 1))
    if len(indx) > 0:
        Coeff[indx] = a.value
        Coeff = W * Coeff

    y = param['D'] @ Coeff

    return y
