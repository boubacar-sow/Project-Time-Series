import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hann
from scipy.io.wavfile import write, read

def makeClippedSignal(x, clippingLevel, GR=False):
    # Normalize and clip a signal.
    xMax = 0.9999
    xClean = x / np.max(np.abs(x)) * xMax
    clippingLevel = clippingLevel * xMax

    # Clipping (hard threshold)
    xClipped = np.minimum(np.maximum(xClean, -clippingLevel), clippingLevel)
    IClipped = np.abs(xClipped) >= clippingLevel  # related indices

    # Size of the clipped segments
    if GR:
        clipSizes = np.diff(IClipped.astype(int))
        if clipSizes[np.where(clipSizes)[0][0]] == -1:
            clipSizes = np.concatenate(([1], clipSizes))
        if clipSizes[np.where(clipSizes)[0][-1]] == 1:
            clipSizes = np.concatenate((clipSizes, [-1]))
        clipSizes = np.diff(np.where(clipSizes))
        clipSizes = clipSizes[::2]

    # Optional graphical display
    if GR:
        # Plot histogram of the sizes of the clipped segments
        plt.figure()
        plt.hist(clipSizes, bins=np.arange(1, np.max(clipSizes)+1))
        plt.title('Size of missing segments')
        plt.xlabel('Size')
        plt.ylabel('# of segments')

        t = np.arange(len(xClean))  # time scale in samples

        # Plot original and clipped signals
        plt.figure()
        plt.plot(t, xClean, label='original')
        plt.plot(t, xClipped, label='clipped')
        plt.legend()

        # Scatter plot between original and clipped signals
        plt.figure()
        plt.plot(xClean, xClipped, '.')
        plt.xlabel('Original signal')
        plt.ylabel('Clipped signal')

        # Spectrograms
        N = 512
        w = hann(N)
        fs = 1
        NOverlap = round(.8 * N)
        nfft = 2 ** (int(np.log2(N)) + 2)

        fig, axs = plt.subplots(3, 3)
        axs[0, 0].specgram(xClean, NFFT=N, Fs=fs, window=w, noverlap=NOverlap, mode='magnitude')
        axs[0, 0].set_title('Original')
        axs[0, 1].specgram(xClipped, NFFT=N, Fs=fs, window=w, noverlap=NOverlap, mode='magnitude')
        axs[0, 1].set_title('Clipped')
        axs[0, 2].specgram(xClean - xClipped, NFFT=N, Fs=fs, window=w, noverlap=NOverlap, mode='magnitude')
        axs[0, 2].set_title('Error (=original-clipped)')
        axs[1, 0].plot(t, xClean)
        axs[1, 1].plot(t, xClean, t, xClipped)
        axs[1, 2].plot(t, xClean - xClipped)

    return xClipped, IClipped, xClean
