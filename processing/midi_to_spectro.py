import numpy as np
import os
from preprocess import save_spectrogram
from midi2audio import FluidSynth
fs = FluidSynth()

# Access environmental variables
from dotenv import load_dotenv
load_dotenv(verbose=True)

# Get Pathname
pathname = os.environ.get('pathname')

for file in os.listdir(pathname)[:3]:
    if file.endswith(".midi"):
        filename = pathname + file
        outputname = "../data/generated/" + pathname[-5:] + file[:-5] + "_gen" + ".wav"
        fs.midi_to_audio(filename, outputname)
        save_spectrogram(outputname, "data/spectrogram_generated")