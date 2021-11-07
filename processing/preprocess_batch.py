"""
Preprocess a years worth of audio and midi files, or the entire MAESTRO dataset
using parallel processing. 

Ensure the `.env` file in the root directory of `midi-shark` includes
the following entries:
    pathname: Path to the directory containing the raw files 
              of maestro-v3.0.0. NOT to a specific year
    dataname: Path to the desired output location

Ensure all the requirements specified in `requirements.txt` is installed
and also FluidSynth is installed on the system.

To preprocess only a years worth of files, execute this file with the 
following command line arguments:
    python preprocess_batch.py --year [year]

If the year is not specified, the entire MAESTRO dataset will be preprocessed.
"""
from dotenv import load_dotenv
import os
import argparse
import numpy as np
from preprocess import save_spectrogram
from process_midi import save_midi, save_notes
from midi2audio import FluidSynth
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

fs = FluidSynth()

# Access environmental variables
load_dotenv(verbose=True)


def prepare_dirs(input_path, output_path, year):
    """
    Create the necessary directories for the preprocessing.

    Parameters
    ----------
    input_path: str
        Path to the directory containing the raw files of maestro-v3.0.0. 
    output_path: str
        Path to the desired output location.
    year: str
        Year to be preprocessed.

    Returns
    -------
    None
    """
    NOTE_GRAPHS_PATH = os.path.join(output_path, 'note_graphs', year) + "/"
    NOTES_GENERATED_PATH = os.path.join(
        output_path, 'notes_generated', year) + "/"
    SPECTROGRAM_GENERATED_PATH = os.path.join(
        output_path, 'spectrograms_generated', year) + "/"
    SPECTROGRAM_REAL_PATH = os.path.join(
        output_path, 'spectrograms_real', year) + "/"
    GENERATED_AUDIO_PATH = os.path.join(
        output_path, 'generated_audio', year) + "/"

    if not os.path.isdir(NOTE_GRAPHS_PATH):
        os.makedirs(NOTE_GRAPHS_PATH)
    if not os.path.isdir(NOTES_GENERATED_PATH):
        os.makedirs(NOTES_GENERATED_PATH)
    if not os.path.isdir(SPECTROGRAM_GENERATED_PATH):
        os.makedirs(SPECTROGRAM_GENERATED_PATH)
    if not os.path.isdir(SPECTROGRAM_REAL_PATH):
        os.makedirs(SPECTROGRAM_REAL_PATH)
    if not os.path.isdir(GENERATED_AUDIO_PATH):
        os.makedirs(GENERATED_AUDIO_PATH)


def preprocess_file(input_path, output_path, year, file):
    """
    Preprocess a single file.

    Parameters
    ----------
    input_path: str
        Path to the directory containing the raw files of maestro-v3.0.0. 
    output_path: str
        Path to the desired output location.
    year: str
        Year of the file to be preprocessed.
    file: str
        Path to the file to be preprocessed.

    Returns
    -------
    None
    """
    NOTE_GRAPHS_PATH = os.path.join(output_path, 'note_graphs', year) + "/"
    NOTES_GENERATED_PATH = os.path.join(
        output_path, 'notes_generated', year) + "/"
    SPECTROGRAM_GENERATED_PATH = os.path.join(
        output_path, 'spectrograms_generated', year) + "/"
    SPECTROGRAM_REAL_PATH = os.path.join(
        output_path, 'spectrograms_real', year) + "/"
    GENERATED_AUDIO_PATH = os.path.join(
        output_path, 'generated_audio', year) + "/"

    filename = os.path.join(input_path, year, file)
    if file.endswith(".wav"):
        # Generate Spectrogram from Raw Audio
        save_spectrogram(filename, SPECTROGRAM_REAL_PATH, file)

    if file.endswith(".midi"):
        # Generate Notes
        save_midi(filename, NOTES_GENERATED_PATH, file)

        # Generate Spectrogram from Midi File
        fs.midi_to_audio(filename, GENERATED_AUDIO_PATH +
                         file.replace(".midi", ".wav"))
        generated_filename = GENERATED_AUDIO_PATH + \
            file.replace(".midi", ".wav")
        generated_file = file.replace(".midi", ".wav")
        save_spectrogram(generated_filename,
                         SPECTROGRAM_GENERATED_PATH, generated_file)

        # Generate Note Graph
        filename_csv = NOTES_GENERATED_PATH + file[:-5] + ".csv"
        notes = np.loadtxt(filename_csv, delimiter=",", dtype=str)

        save_notes(notes, NOTE_GRAPHS_PATH, file)


if __name__ == '__main__':
    # Parse the year
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=0,
                        help='Year to preprocess.' +
                             'If not specified, defaults to all years')
    args = parser.parse_args()
    year = args.year
    YEARS = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]

    # Get Pathname
    input_path = os.environ.get('pathname')
    output_path = os.environ.get('dataname')

    if not year:
        for year in YEARS:
            year = str(year)
            prepare_dirs(input_path, output_path, year)
            with ProcessPoolExecutor(max_workers=24) as executor:
                for file in os.listdir(os.path.join(input_path, year)):
                    executor.submit(preprocess_file,
                                    input_path,
                                    output_path,
                                    year,
                                    file)
    elif year in YEARS:
        year = str(year)
        prepare_dirs(input_path, output_path, year)
        with ProcessPoolExecutor(max_workers=24) as executor:
            for file in os.listdir(os.path.join(input_path, year)):
                executor.submit(preprocess_file,
                                input_path,
                                output_path,
                                year,
                                file)
    else:
        raise RuntimeError('Invaid year')
