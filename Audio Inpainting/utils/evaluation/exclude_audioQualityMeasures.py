import numpy as np
import os
import tempfile
import warnings
from scipy.io import wavfile
from snr import SNR

def audioQualityMeasures(xRef, xTest, fs, options=None):
    # Computes a series of audio quality measures
    defaultOptions = {'pemoQExec': '"C:/Program Files/PEMO-Q v1.2/audioqual.exe"', 
                      'ENABLE_PEMOQ': True, 
                      'PESQExec': './PESQ/pesq', 
                      'EAQUALExec': 'wine ./EAQUAL/eaqual.exe' if os.name != 'nt' else './EAQUAL/eaqual.exe'}

    if options is None:
        options = defaultOptions
    else:
        for k in defaultOptions.keys():
            if k not in options or options[k] is None:
                options[k] = defaultOptions[k]

    if not isinstance(xRef, str) and not isinstance(xTest, str) and len(xRef) != len(xTest):
        warnings.warn('Different lengths')
        L = min(len(xRef), len(xTest))
        xRef = xRef[:L]
        xTest = xTest[:L]

    if isinstance(xRef, str):
        refFile = xRef
        fs, sRef = wavfile.read(refFile)
    else:
        refFile = tempfile.mktemp('.wav')
        sRef = xRef
        wavfile.write(refFile, fs, xRef)

    if isinstance(xTest, str):
        testFile = xTest
        fs, sTest = wavfile.read(testFile)
    else:
        testFile = tempfile.mktemp('.wav')
        sTest = xTest
        wavfile.write(testFile, fs, xTest)

    SNRx = SNR(sRef, sTest)

    # The following parts depend on external programs and are not included in the Python code
    # PSM, PSMt = aux_pemoq(refFile, testFile, options)
    # PESQ_MOS = aux_pesq(refFile, testFile, options)
    # EAQUAL_ODG, EAQUAL_DIX = aux_eaqual(refFile, testFile, options)

    if not isinstance(xRef, str):
        os.remove(refFile)
    if not isinstance(xTest, str):
        os.remove(testFile)

    return SNRx
