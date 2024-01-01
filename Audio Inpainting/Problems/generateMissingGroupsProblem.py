import numpy as np
from typing import Dict, Any, Tuple

def makeRandomMeasurementMatrix(N, NMissingBlocks, blockSize):
    nTry = 1
    while True:
        try:
            Im = np.zeros(N, dtype=bool)
            possibleStart = np.arange(1, N - blockSize + 2)

            for k in range(NMissingBlocks):
                if len(possibleStart) == 0:
                    raise ValueError('Too much missing segments')
                I = np.random.choice(possibleStart)
                Im[I:I+blockSize] = True
                possibleStart = possibleStart[(possibleStart < I - blockSize) | (possibleStart > I + blockSize)]
            break
        except ValueError:
            print(f'makeRandomMeasurementMatrix:retry ({nTry})')
            nTry += 1
            if nTry > 10:
                Im = np.tile(np.concatenate([np.ones(blockSize, dtype=bool), [False]]), NMissingBlocks)
                while len(Im) < N:
                    N0 = np.sum(~Im)
                    I0 = np.random.choice(np.where(~Im)[0], size=N0)
                    I0 = I0[-1]
                    Im = np.concatenate([Im[:I0], [False], Im[I0:]])
                Im = np.roll(Im, np.random.randint(N))
                break
    M = np.eye(N)
    M = M[~Im]
    return M, Im

def generateMissingGroupsProblem(xFrame, problemParameters):
    N = len(xFrame)  # frame length

    # Normalize
    xFrame = xFrame / np.max(np.abs(xFrame))
    # Build random measurement matrix with NHoles of length holeSize
    M, IMiss = makeRandomMeasurementMatrix(N, int(problemParameters['NHoles']), int(problemParameters['holeSize']))
    xFrameObs = xFrame.copy()
    xFrameObs[IMiss] = 0

    problemData = {'x': xFrameObs, 'IMiss': IMiss}
    solutionData = {'xClean': xFrame}
    return problemData, solutionData

import numpy as np

def makePeriodicMeasurementMatrix(N, interval_duration, missing_duration):
    # Calculate the number of missing blocks and their size
    NMissingBlocks = N // interval_duration
    blockSize = int(missing_duration * N / NMissingBlocks)

    # Initialize the boolean mask for missing samples
    Im = np.zeros(N, dtype=bool)

    # Set the missing samples
    for k in range(NMissingBlocks):
        start = k * interval_duration
        end = start + blockSize
        Im[start:end] = True

    # Create the measurement matrix
    M = np.eye(N)
    M = M[~Im]

    return M, Im

def generateMissingGroupsProblemPeriodic(x: np.ndarray, problemParameters: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Generate a missing groups problem with periodic missing intervals.

    Args:
        x (np.ndarray): The clean signal.
        problemParameters (dict): A dictionary containing the parameters of the problem such as the interval duration and the missing duration.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple of two dictionaries.
            - problemData (dict): A dictionary containing the observed signal and the indices of the missing samples.
            - solutionData (dict): A dictionary containing the clean signal and the indices of the clean samples.
    """

    # Set parameters
    if 'interval_duration' not in problemParameters:
        problemParameters['interval_duration'] = 100
    if 'missing_duration' not in problemParameters:
        problemParameters['missing_duration'] = 1

    # Calculate the number of missing blocks and their size
    NMissingBlocks = len(x) // problemParameters['interval_duration']
    blockSize = int(problemParameters['missing_duration'] * len(x) / NMissingBlocks)

    # Initialize the boolean mask for missing samples
    IMiss = np.zeros(len(x), dtype=bool)

    # Set the missing samples
    for k in range(NMissingBlocks):
        start = k * problemParameters['interval_duration']
        end = start + blockSize
        IMiss[start:end] = True

    # Create the observed signal
    xObs = x.copy()
    xObs[IMiss] = 0

    # Create the problem and solution data
    problemData = {'x': xObs, 'IMiss': IMiss}
    solutionData = {'xClean': x, 'IClean': ~IMiss}

    return problemData, solutionData
