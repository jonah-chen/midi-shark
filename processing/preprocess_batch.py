import os
import numpy as np
from preprocess import save_spectrogram
from process_midi import save_midi, save_notes
from midi2audio import FluidSynth
fs = FluidSynth()

# Access environmental variables
from dotenv import load_dotenv
load_dotenv(verbose=True)

# Get Pathname
input_path = os.environ.get('pathname')
output_path = os.environ.get('dataname')
year = os.path.basename(os.path.normpath(input_path))

NOTE_GRAPHS_PATH = os.path.join(output_path, 'note_graphs', year) + "/"
NOTES_GENERATED_PATH = os.path.join(output_path, 'notes_generated', year) + "/"
SPECTROGRAM_GENERATED_PATH = os.path.join(output_path, 'spectrograms_generated', year) + "/"
SPECTROGRAM_REAL_PATH = os.path.join(output_path, 'spectrograms_real', year) + "/"

if not os.path.isdir(NOTE_GRAPHS_PATH): os.makedirs(NOTE_GRAPHS_PATH)
if not os.path.isdir(NOTES_GENERATED_PATH): os.makedirs(NOTES_GENERATED_PATH)
if not os.path.isdir(SPECTROGRAM_GENERATED_PATH): os.makedirs(SPECTROGRAM_GENERATED_PATH)
if not os.path.isdir(SPECTROGRAM_REAL_PATH): os.makedirs(SPECTROGRAM_REAL_PATH)

for file in os.listdir(input_path)[:3]:
    if file.endswith(".wav"):
        filename = input_path + file
        save_spectrogram(filename, SPECTROGRAM_REAL_PATH)
    
    if file.endswith(".midi"):
        # # Generate Notes
        filename = input_path + file
        save_midi(filename, NOTES_GENERATED_PATH, file) 

        # Generate Spectrogram
        fs.midi_to_audio(filename, SPECTROGRAM_REAL_PATH)

        filename_csv = NOTES_GENERATED_PATH + file[:-5] + ".csv"
        notes = np.loadtxt(filename_csv, delimiter=",", dtype=str)

        save_notes(notes, NOTE_GRAPHS_PATH, file)