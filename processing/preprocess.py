import argparse

import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil

def spectogram_librosa(wav_file_path, offset = 0, duration = 20, show=False):
    y, sr = librosa.load(wav_file_path, offset = offset, duration = duration)
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

def save_spectrogram(filename, max_duration=20, force = False):
    song_duration = librosa.get_duration(filename = filename)

    folder_name = filename.split("/",-1)[-1][:-4]
    folder_name = f"../data/{folder_name}"

    # Forcibly remove folder and contents if it exists
    if force and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    for t in np.arange(0, song_duration, max_duration):
        duration = min(max_duration, np.floor(song_duration - t))
        spectrogram = spectogram_librosa(filename, t, duration, False)
        
        # Clean up file name string
        name = f"{folder_name}/offset_{t}_duration_{duration}"
        
        # Save spectrogram to data folder
        np.save(name, spectrogram)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', help='.wav file to parse', type=str)
    args = parser.parse_args()
    spectogram_librosa(args.wav_file)