import argparse

import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np


def spectogram_librosa(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    y = librosa.feature.melspectrogram(
        y=y,
        sr=16000,
        hop_length=512,
        n_fft=2048,
        n_mels=229)
    y = librosa.power_to_db(y, ref=np.max)
    librosa.display.specshow(y, x_axis='time', y_axis='mel')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', help='.wav file to parse', type=str)
    args = parser.parse_args()
    spectogram_librosa(args.wav_file)
