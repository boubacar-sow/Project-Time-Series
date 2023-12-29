import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy.signal import spectrogram, hann
from Problems.generateDeclippingProblem import generateDeclippingProblem
from utils.wRect import wRect
from utils.dictionaries.Gabor_dictionary import Gabor_Dictionary
from Solvers.inpaintSignal_IndependentProcessingOfFrames import inpaintSignal_IndependentProcessingOfFrames
from Solvers.inpaintFrame_OMP_Gabor import inpaintFrame_OMP_Gabor
from utils.evaluation.SNRInpaintingPerformance import SNRInpaintingPerformance
from utils.wSine import wSine

def declipOneSoundExperiment(expParam=None):
    # Set parameters
    if expParam is None:
        expParam = {}
    if 'filename' not in expParam:
        expParam['filename'] = '../../Data/testSpeech8kHz_from16kHz/male01_8kHz.wav'
    if 'clippingLevel' not in expParam:
        expParam['clippingLevel'] = 0.6

    # Solver
    if 'solver' not in expParam:
        print('Frame length=256 is used to have faster computations. Recommended frame length is 512 at 8kHz.')
        print('Overlap factor=2 is used to have faster computations. Recommended value: 4.')
        
        expParam['solver'] = {
            'name': 'OMP-G',
            'function': inpaintSignal_IndependentProcessingOfFrames,
            'param': {
                'N': 256,  # frame length
                'inpaintFrame': inpaintFrame_OMP_Gabor,  # solver function
                'OMPerr': 0.001,
                'sparsityDegree': 256/4,
                'D_fun': Gabor_Dictionary,  # Dictionary (function handle)
                'OLA_frameOverlapFactor': 2,
                'redundancyFactor': 2,  # Dictionary redundancy
                'wd': wRect,  # Weighting window for dictionary atoms
                'wa': wRect,  # Analysis window
                'OLA_ws': wSine,  # Synthesis window
                'SKIP_CLEAN_FRAMES': True,  # do not process frames where there is no missing samples
                'MULTITHREAD_FRAME_PROCESSING': False,  # not implemented yet
            }
        }
    if 'destDir' not in expParam:
        expParam['destDir'] = '../../tmp/declipOneSound/'
    if not os.path.exists(expParam['destDir']):
        os.makedirs(expParam['destDir'])

    # Read test signal
    fs, x = read(expParam['filename'])

    # Generate the problem
    problemData, solutionData = generateDeclippingProblem(x, expParam['clippingLevel'])

    # Declip with solver
    print('\nDeclipping\n')
    solverParam = expParam['solver']['param']
    xEst1, xEst2 = expParam['solver']['function'](problemData, solverParam)

    # Compute and display SNR performance
    L = len(xEst1)
    N = expParam['solver']['param']['N']
    SNRAll, SNRmiss = SNRInpaintingPerformance(
        solutionData['xClean'][N:L-N], problemData['x'][N:L-N],
        xEst2[N:L-N], problemData['IMiss'][N:L-N])
    print('SNR on missing samples:')
    print(f'Clipped: {SNRmiss[0]} dB')
    print(f'Estimate: {SNRmiss[1]} dB')

    # Plot results
    xClipped = problemData['x']
    xClean = solutionData['xClean']
    plt.figure()
    plt.plot(xClipped, 'r')
    plt.plot(xClean)
    plt.plot(xEst2, '--g')
    plt.plot([1, len(xClipped)], [1, 1]*[-1, 1]*np.max(np.abs(xClipped)), ':r')
    plt.legend(['Clipped', 'True solution', 'Estimate'])

    # Normalized and save sounds
    normX = 1.1*np.max(np.abs([xEst1, xEst2, xClean]))
    L = min([len(xEst2), len(xEst1), len(xClean), len(xEst1), len(xClipped)])
    xEst1 = xEst1[:L] / normX
    xEst2 = xEst2[:L] / normX
    xClipped = xClipped[:L] / normX
    xClean = xClean[:L] / normX
    write(os.path.join(expParam['destDir'], 'xEst1.wav'), fs, xEst1)
    write(os.path.join(expParam['destDir'], 'xEst2.wav'), fs, xEst2)
    write(os.path.join(expParam['destDir'], 'xClipped.wav'), fs, xClipped)
    write(os.path.join(expParam['destDir'], 'xClean.wav'), fs, xClean)
