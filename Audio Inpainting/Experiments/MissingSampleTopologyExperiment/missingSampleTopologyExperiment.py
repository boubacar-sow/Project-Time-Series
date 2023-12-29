import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
from utils.dictionaries.Gabor_dictionary import Gabor_Dictionary
from utils.wRect import wRect
from sympy import factorint
from scipy.io.wavfile import read
from Solvers.inpaintFrame_OMP_Gabor import inpaintFrame_OMP_Gabor
from Solvers.inpaintFrame_janssenInterpolation import inpaintFrame_janssenInterpolation
from Problems.generateMissingGroupsProblem import generateMissingGroupsProblem
from utils.evaluation.SNRInpaintingPerformance import SNRInpaintingPerformance

def MissingSampleTopologyExperiment(expParam=None):
    # Set parameters
    if expParam is None:
        expParam = {}
    if 'soundDir' not in expParam:
        expParam['soundDir'] = '../../Data/testSpeech8kHz_from16kHz/'
    if 'destDir' not in expParam:
        expParam['destDir'] = '../../tmp/missSampTopoExp/'
    if not os.path.exists(expParam['destDir']):
        os.makedirs(expParam['destDir'])

    # frame length
    if 'N' not in expParam:
        expParam['N'] = 256
        warnings.warn('Frame length=256 is used to have faster computations. Recommended frame length is 512 at 8kHz.')

    # Number of random frames to test
    if 'NFramesPerHoleSize' not in expParam:
        expParam['NFramesPerHoleSize'] = 20
        warnings.warn('expParam.NFramesPerHoleSize = 20 is used to have faster computations. Recommended value: several hundreds.')

    # Number of missing samples: which numbers to test?
    if 'totalMissSamplesList' not in expParam:
        expParam['totalMissSamplesList'] = [12, 36]
        warnings.warn('expParam.totalMissSamplesList = [12,36] is used to have faster computations. Recommended list: expParam.totalMissSamplesList = [12,36,60,120,180,240].')

    # Choose the solver methods you would like to test: OMP, L1, Janssen
    if 'solvers' not in expParam:
        expParam['solvers'] = [
            {
                'name': 'OMP-G',
                'inpaintFrame': inpaintFrame_OMP_Gabor,  # solver function
                'param': {
                    'N': expParam['N'],  # frame length
                    'OMPerr': 0.001,
                    'sparsityDegree': expParam['N'] / 4,
                    'D_fun': Gabor_Dictionary,  # Dictionary (function handle)
                    'redundancyFactor': 2,  # Dictionary redundancy
                    'wa': wRect,  # Analysis window
                }
            },
            {
                'name': 'Janssen',
                'inpaintFrame': inpaintFrame_janssenInterpolation,  # solver function
                'param': {
                    'N': expParam['N'],  # frame length
                }
            }
        ]

    # Draw a list of random frames
    soundDir = expParam['soundDir']
    wavFiles = glob.glob(os.path.join(soundDir, '*.wav'))

    # Choose an audio file at random
    frameParam = {'kFrameFile': np.random.randint(len(wavFiles), size=expParam['NFramesPerHoleSize'])}

    # For each audio file, find maximum mean energy among all frames
    fs, dum = read(wavFiles[0])
    Ne = round(512 / 16000 * fs)
    E2m = np.zeros(len(wavFiles))
    for kf in range(len(wavFiles)):
        fs, x = read(wavFiles[kf])
        xm = np.convolve(np.abs(x**2), np.ones(Ne) / Ne, mode='valid')
        E2m[kf] = 10 * np.log10(np.max(xm))

    # Choose the location of a frame at random, with a minimum energy
    maxDiffE2m = 10
    frameParam['kFrameBegin'] = np.full(expParam['NFramesPerHoleSize'], np.nan)
    for kf in range(expParam['NFramesPerHoleSize']):
        while True:
            fs, siz = read(wavFiles[frameParam['kFrameFile'][kf]])
            frameParam['kFrameBegin'][kf] = np.random.randint(siz[0] - expParam['N'] + 1)
            fs, x = read(wavFiles[frameParam['kFrameFile'][kf]], mmap=True)
            x = x[frameParam['kFrameBegin'][kf]:frameParam['kFrameBegin'][kf]+expParam['N']]
            E2m0 = 10 * np.log10(np.mean(np.abs(x**2)))
            if E2m[frameParam['kFrameFile'][kf]] - E2m0 <= maxDiffE2m:
                break

    # Test each number of missing samples
    PerfRes = np.zeros((len(expParam['totalMissSamplesList']), len(expParam['solvers']), expParam['NFramesPerHoleSize']))
    factorsToTest = [allFactors(NMissSamples) for NMissSamples in expParam['totalMissSamplesList']]
    outputFile = os.path.join(expParam['destDir'], 'missSampTopoExp.mat')
    for kSolver in range(len(expParam['solvers'])):
        print(f'\n ------ Solver: {expParam["solvers"][kSolver]["name"]} ------\n\n')
        for kMiss in range(len(expParam['totalMissSamplesList'])):
            NMissSamples = expParam['totalMissSamplesList'][kMiss]
            for kFactor in range(len(factorsToTest[kMiss])):
                holeSize = factorsToTest[kMiss][kFactor]
                NHoles = NMissSamples / holeSize
                print(f'{NHoles} {holeSize}-length holes ({NMissSamples} missing samples = {NMissSamples/expParam["N"]*100:.1f}%)')
                problemParameters = {'holeSize': holeSize, 'NHoles': NHoles}
                for kFrame in range(expParam['NFramesPerHoleSize']):
                    # load audio frame
                    fs, xFrame = read(wavFiles[frameParam['kFrameFile'][kFrame]], mmap=True)
                    xFrame = xFrame[frameParam['kFrameBegin'][kFrame]:frameParam['kFrameBegin'][kFrame]+expParam['N']]

                    # generate problem
                    problemData, solutionData = generateMissingGroupsProblem(xFrame, problemParameters)

                    # solve problem
                    xEst = expParam['solvers'][kSolver]['inpaintFrame'](problemData, expParam['solvers'][kSolver]['param'])

                    # compute and store performance
                    SNRAll, SNRmiss = SNRInpaintingPerformance(solutionData['xClean'], problemData['x'], xEst, problemData['IMiss'])
                    PerfRes[kMiss, kSolver, kFrame] = SNRmiss[1]

            np.save(outputFile, {'PerfRes': PerfRes, 'expParam': expParam})

    # Plot results
    Nrows = int(np.floor(np.sqrt(len(expParam['solvers']))))
    Ncols = int(np.ceil(np.sqrt(len(expParam['solvers'])) / Nrows))
    cmap = plt.get_cmap('tab10')
    fig, axs = plt.subplots(Nrows, Ncols)
    for kSolver in range(len(expParam['solvers'])):
        for kMiss in range(len(expParam['totalMissSamplesList'])):
            axs[kSolver // Ncols, kSolver % Ncols].plot(factorsToTest[kMiss], np.mean(PerfRes[kMiss, kSolver], axis=1), color=cmap(kMiss))
        axs[kSolver // Ncols, kSolver % Ncols].set_title(expParam['solvers'][kSolver]['name'])
    plt.show()

def allFactors(n):
    # Find the list of all factors (not only prime factors)

    primeFactors = factorint(n)

    degrees = [primeFactors[p] for p in primeFactors]

    D = np.array([np.arange(degrees[0] + 1)]).T
    for k in range(1, len(degrees)):
        Dk = np.ones((D.shape[0], 1)) * np.arange(degrees[k] + 1)
        D = np.concatenate([np.repeat(D, degrees[k] + 1, axis=0), Dk.flatten()[:, None]], axis=1)

    m = np.unique(np.sort(np.prod(np.power(np.ones((D.shape[0], 1)) * np.array(list(primeFactors.keys())), D), axis=1)))

    return m

