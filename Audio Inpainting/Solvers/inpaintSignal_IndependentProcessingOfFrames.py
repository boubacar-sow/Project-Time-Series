from utils.wSine import wSine
import numpy as np
from utils.dictionaries.Gabor_dictionary import Gabor_Dictionary
from utils.dictionaries.DCT_Dictionary import DCT_Dictionary
from utils.wRect import wRect
from Solvers.inpaintFrame_OMP import inpaintFrame_OMP
from typing import Dict, Any, Tuple

def inpaintSignal_IndependentProcessingOfFrames(problemData: Dict[str, np.ndarray], param: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Audio De-clipping with overlapping blocks. This method uses a synthesis approach and a union of overcomplete DCT dictionary.

    Args:
        problemData (dict): A dictionary containing the observed signal to be inpainted and the indices of clean samples.
            - 'x' (np.ndarray): Clipped signal.
            - 'Imiss' (np.ndarray): Indices of clipped samples.
        param (dict): A dictionary containing optional parameters such as frame length, overlap factor between frames, weighting analysis window, weighting synthesis window, error threshold to stop OMP iterations, max number of non-zero components to stop OMP iterations, and other fields.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays.
            - ReconstSignal1 (np.ndarray): Reconstructed signal (all samples generated from the synthesis model).
            - ReconstSignal2 (np.ndarray): Reconstructed signal (only clipped samples are generated from the synthesis model).
    """

    defaultParam = {
        'N': 256,
        'OLA_frameOverlapFactor': 4,
        'wa': wSine,
        'OLA_ws': wSine,
        'OLA_par_waitingTime_mainProcess': 0.2,
        'OLA_par_waitingTime_thread': 0.2,
        'OLA_par_frameBlockSize': 1,
        'TCPIP_port': 3000,
        'COM_DISP': False,
        'STATE_DISP': False
    }

    if param is None:
        param = defaultParam
    else:
        for key in defaultParam:
            if key not in param or param[key] is None:
                param[key] = defaultParam[key]

    x = problemData['x']
    ClipMask = np.where(problemData['IMiss'])[0]

    if param['MULTITHREAD_FRAME_PROCESSING']:
        ReconstSignal1, ReconstSignal2 = singlethreadProcessing(x, ClipMask, param)
    else:
        ReconstSignal1, ReconstSignal2 = singlethreadProcessing(x, ClipMask, param)

    return ReconstSignal1, ReconstSignal2



def singlethreadProcessing(x, ClipMask, param):
    defaultParam = {
        'N': 256,
        'inpaintFrame': inpaintFrame_OMP,
        'OLA_frameOverlapFactor': 2,
        'wa': wSine,
        'ws': wSine,
        'SKIP_CLEAN_FRAMES': True
    }

    if param is None:
        param = defaultParam
    else:
        for key in defaultParam:
            if key not in param or param[key] is None:
                param[key] = defaultParam[key]

    bb = param['N']

    L = (len(x) // bb) * bb
    x = x[:L]
    ClipMask = ClipMask[ClipMask < L]

    Ibegin = np.arange(0, len(x) - bb + 1, bb // param['OLA_frameOverlapFactor'])
    if Ibegin[-1] != L - bb + 1:
        Ibegin = np.append(Ibegin, L - bb + 1)
    Iblk = np.outer(np.arange(bb), np.ones(len(Ibegin))) + np.outer(np.ones(bb), Ibegin)
    wa = param['wa'](bb)
    xFrames = np.diag(wa) @ x[Iblk.astype(int)]

    Mask = np.ones(len(x))
    Mask[ClipMask] = 0
    blkMask = Mask[Iblk.astype(int)]

    n, P = xFrames.shape

    Reconst = np.zeros((n, P))
    for k in range(P):
        if param['SKIP_CLEAN_FRAMES'] and np.all(blkMask[:, k]):
            continue
        frameProblemData = {'x': xFrames[:, k], 'IMiss': ~blkMask[:, k]}
        Reconst[:, k] = param['inpaintFrame'](frameProblemData, param)
    ReconstSignal1 = np.zeros(len(x))
    ws = param['ws'](bb)
    wNorm = np.zeros(len(x))
    for k in range(Iblk.shape[1]):
        ReconstSignal1[Iblk[:, k].astype(int)] += Reconst[:, k] * ws
        wNorm[Iblk[:, k].astype(int)] += ws * wa
    ReconstSignal1 /= wNorm

    ReconstSignal2 = x.copy()
    ReconstSignal2[ClipMask] = ReconstSignal1[ClipMask]

    return ReconstSignal1, ReconstSignal2
