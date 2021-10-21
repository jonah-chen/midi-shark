import os
import numpy as np
# import matplotlib.pyplot as plt
import librosa
import librosa.display
from preprocess import spectogram_librosa, save_spectrogram

# Access environmental variables
from dotenv import load_dotenv
load_dotenv(verbose=True)

# Get Pathname
pathname = os.environ.get('pathname')
for file in os.listdir(pathname)[:3]:
    if file.endswith(".wav"):
        filename = pathname + file 
        save_spectrogram(filename)