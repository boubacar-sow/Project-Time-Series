import numpy as np
from snr import SNR


def SNRInpaintingPerformance(xRef, xObs, xEst, IMiss, DISP_FLAG=0):
    # Various SNR measures for inpainting performance
    SNRAll = [SNR(xRef, xObs), SNR(xRef, xEst)]
    SNRmiss = [SNR(xRef[IMiss], xObs[IMiss]), SNR(xRef[IMiss], xEst[IMiss])]

    if DISP_FLAG > 0:
        print('SNR on all samples / clipped samples:')
        print(f'Original: {SNRAll[0]} dB / {SNRmiss[0]} dB')
        print(f'Estimate: {SNRAll[1]} dB / {SNRmiss[1]} dB')

    return SNRAll, SNRmiss
