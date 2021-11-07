import os
import argparse
import numpy as np
from preprocess import save_spectrogram
from process_midi import save_midi, save_notes
from midi2audio import FluidSynth
from concurrent.futures import ProcessPoolExecutor

fs = FluidSynth()

# Access environmental variables
from dotenv import load_dotenv
load_dotenv(verbose=True)

def prepare_dirs(input_path, output_path, year):
    """
    Create directories to store the preprocessed files for a given year

    Parameters
    ----------
    input_path : str
        Path to the directory containing the raw files
    output_path : str
        Path to the directory to store the preprocessed files
    year : str
        Year of the files to be preprocessed
    """
    year_dir = os.path.join(output_path, year)
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
    for i in range(1, 13):
        month_dir = os.path.join(year_dir, str(i).zfill(2))
        if not os.path.exists(month_dir):
            os.makedirs(month_dir)
    return year_dir
    NOTE_GRAPHS_PATH = os.path.join(output_path, 'note_graphs', year) + "/"
    NOTES_GENERATED_PATH = os.path.join(output_path, 'notes_generated', year) + "/"
    SPECTROGRAM_GENERATED_PATH = os.path.join(output_path, 'spectrograms_generated', year) + "/"
    SPECTROGRAM_REAL_PATH = os.path.join(output_path, 'spectrograms_real', year) + "/"
    GENERATED_AUDIO_PATH = os.path.join(output_path, 'generated_audio', year) + "/"

    if not os.path.isdir(NOTE_GRAPHS_PATH): os.makedirs(NOTE_GRAPHS_PATH)
    if not os.path.isdir(NOTES_GENERATED_PATH): os.makedirs(NOTES_GENERATED_PATH)
    if not os.path.isdir(SPECTROGRAM_GENERATED_PATH): os.makedirs(SPECTROGRAM_GENERATED_PATH)
    if not os.path.isdir(SPECTROGRAM_REAL_PATH): os.makedirs(SPECTROGRAM_REAL_PATH)
    if not os.path.isdir(GENERATED_AUDIO_PATH): os.makedirs(GENERATED_AUDIO_PATH)

def preprocess_year(input_path, output_path, year, file):
    
    NOTE_GRAPHS_PATH = os.path.join(output_path, 'note_graphs', year) + "/"
    NOTES_GENERATED_PATH = os.path.join(output_path, 'notes_generated', year) + "/"
    SPECTROGRAM_GENERATED_PATH = os.path.join(output_path, 'spectrograms_generated', year) + "/"
    SPECTROGRAM_REAL_PATH = os.path.join(output_path, 'spectrograms_real', year) + "/"
    GENERATED_AUDIO_PATH = os.path.join(output_path, 'generated_audio', year) + "/"

    
    filename = os.path.join(input_path, year, file)
    if file.endswith(".wav"):
        # Generate Spectrogram from Raw Audio
        save_spectrogram(filename, SPECTROGRAM_REAL_PATH, file)
    
    if file.endswith(".midi"):
        # Generate Notes
        save_midi(filename, NOTES_GENERATED_PATH, file) 

        # Generate Spectrogram from Midi File
        fs.midi_to_audio(filename, GENERATED_AUDIO_PATH + file.replace(".midi", ".wav"))
        generated_filename = GENERATED_AUDIO_PATH + file.replace(".midi", ".wav")
        generated_file = file.replace(".midi", ".wav")
        save_spectrogram(generated_filename, SPECTROGRAM_GENERATED_PATH, generated_file)

        # Generate Note Graph
        filename_csv = NOTES_GENERATED_PATH + file[:-5] + ".csv"
        notes = np.loadtxt(filename_csv, delimiter=",", dtype=str)

        save_notes(notes, NOTE_GRAPHS_PATH, file)

if __name__ == '__main__':
    # Parse the year
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2004, help='Year to preprocess')
    args = parser.parse_args()
    year = args.year
    YEARS = [2004,2006,2008,2009,2011,2013,2014,2015,2017,2018]

    # Get Pathname
    input_path = os.environ.get('pathname')
    output_path = os.environ.get('dataname')

    if year in YEARS:
        year = str(year)
        prepare_dirs(input_path, output_path, year)
        with ProcessPoolExecutor(max_workers=24) as executor:
            for file in os.listdir(os.path.join(input_path, year)):
                executor.submit(preprocess_year, input_path, output_path, year, file)
    else:
        raise ValueError('Invaid year')
    
