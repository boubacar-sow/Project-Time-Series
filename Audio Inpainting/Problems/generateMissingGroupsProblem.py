import numpy as np

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
    M, IMiss = makeRandomMeasurementMatrix(N, problemParameters['NHoles'], problemParameters['holeSize'])
    xFrameObs = xFrame.copy()
    xFrameObs[IMiss] = 0

    problemData = {'x': xFrameObs, 'IMiss': IMiss}
    solutionData = {'xClean': xFrame}
    return problemData, solutionData
