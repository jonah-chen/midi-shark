import argparse

import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np


def spectogram_librosa(wav_file_path, show=False):
    y, sr = librosa.load(wav_file_path)
    y = librosa.feature.melspectrogram(
        y=y,  # audio time-series
        sr=16000,  # sampling rate of y
        hop_length=512,  # number of samples between successive frames.
        n_fft=2048,  # length of the FFT window
        n_mels=229,  # number of Mel bands to generate
    )
    y = librosa.power_to_db(y, ref=np.max)
    if show:
        librosa.display.specshow(y, x_axis='time', y_axis='mel')
        plt.show()
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', help='.wav file to parse', type=str)
    args = parser.parse_args()
    spectogram_librosa(args.wav_file)
