import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hann

def generateDeclippingProblem(x, clippingLevel, GR=False):
    # Normalize and clip a signal.
    xMax = 0.9999
    solutionData = {'xClean': x / np.max(np.abs(x)) * xMax}
    clippingLevel = clippingLevel * xMax

    # Clipping (hard threshold)
    problemData = {'x': np.minimum(np.maximum(solutionData['xClean'], -clippingLevel), clippingLevel)}
    problemData['IMiss'] = np.abs(problemData['x']) >= clippingLevel  # related indices

    # Size of the clipped segments
    problemData['clipSizes'] = np.diff(problemData['IMiss'].astype(int))
    if problemData['clipSizes'][np.where(problemData['clipSizes'])[0][0]] == -1:
        problemData['clipSizes'] = np.concatenate(([1], problemData['clipSizes']))
    if problemData['clipSizes'][np.where(problemData['clipSizes'])[0][-1]] == 1:
        problemData['clipSizes'] = np.concatenate((problemData['clipSizes'], [-1]))
    problemData['clipSizes'] = np.diff(np.where(problemData['clipSizes']))
    problemData['clipSizes'] = problemData['clipSizes'][::2]

    # Optional graphical display
    if GR:
        # Plot histogram of the sizes of the clipped segments
        plt.figure()
        plt.hist(problemData['clipSizes'], bins=np.arange(1, np.max(problemData['clipSizes'])+1))
        plt.title('Size of missing segments')
        plt.xlabel('Size')
        plt.ylabel('# of segments')

        t = np.arange(len(solutionData['xClean']))  # time scale in samples

        # Plot original and clipped signals
        plt.figure()
        plt.plot(t, solutionData['xClean'], label='original')
        plt.plot(t, problemData['x'], label='clipped')
        plt.legend()

        # Scatter plot between original and clipped signals
        plt.figure()
        plt.plot(solutionData['xClean'], problemData['x'], '.')
        plt.xlabel('Original signal')
        plt.ylabel('Clipped signal')

        # Spectrograms
        N = 512
        w = hann(N)
        fs = 1
        NOverlap = round(.8 * N)
        nfft = 2 ** (int(np.log2(N)) + 2)

        fig, axs = plt.subplots(3, 3)
        axs[0, 0].specgram(solutionData['xClean'], NFFT=N, Fs=fs, window=w, noverlap=NOverlap, mode='magnitude')
        axs[0, 0].set_title('Original')
        axs[0, 1].specgram(problemData['x'], NFFT=N, Fs=fs, window=w, noverlap=NOverlap, mode='magnitude')
        axs[0, 1].set_title('Clipped')
        axs[0, 2].specgram(solutionData['xClean'] - problemData['x'], NFFT=N, Fs=fs, window=w, noverlap=NOverlap, mode='magnitude')
        axs[0, 2].set_title('Error (=original-clipped)')
        axs[1, 0].plot(t, solutionData['xClean'])
        axs[1, 1].plot(t, solutionData['xClean'], t, problemData['x'])
        axs[1, 2].plot(t, solutionData['xClean'] - problemData['x'])

    return problemData, solutionData
