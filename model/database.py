import torch
import os
from torch.utils.data import Dataset
import numpy as np

# Access environmental variables
from dotenv import load_dotenv
load_dotenv(verbose=True)

# Get Pathname
input_path = os.environ.get('pathname')
output_path = os.environ.get('dataname')

# Return a dictionary of file names
def get_file_names(path):
    file_names = {}
    for year in os.listdir(path):
        for song in os.listdir(path + "/" + year):
            if not song in {'MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--4', 'MIDI-Unprocessed_050_PIANO050_MID--AUDIO-split_07-06-17_Piano-e_3-01_wav--3', 'MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--3', 'MIDI-Unprocessed_081_PIANO081_MID--AUDIO-split_07-09-17_Piano-e_2_-02_wav--4', 'MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_08_WAV'}:
                for file in os.listdir(path + "/" + year + "/" + song):
                    # Ignore the end of the song (if less than 20s)
                    if file.endswith("20.npy"):
                        file_name = year + "_" + song + "_" + file
                        file_names[file_name] = path + "/" + year + "/" + song + "/" + file

    return file_names

class DeNoiseDataset(Dataset):
    """
        Dataset that contains the real and generated spectrograms.
    """

    def __init__(self, data_folder):
        """
        Args:
            data_folder (string): Path to the folder that contains everything.
        """
        self.generated = {}
        self.real = {}

        SPECTROGRAM_GENERATED_PATH = data_folder + '/spectrograms_generated/'
        SPECTROGRAM_REAL_PATH = data_folder + '/spectrograms_real/'

        self.generated = get_file_names(SPECTROGRAM_GENERATED_PATH)
        self.real = get_file_names(SPECTROGRAM_REAL_PATH)

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        song_name = list(self.real)[idx]
        real_song = self.real[song_name]
        generated_song = self.generated[song_name]
        sample = {'real': np.load(real_song), 'generated': np.load(generated_song)}

        return sample

class SpectrogramNotesDataset(Dataset):
    '''
        Dataset that contains the real spectrogram and the note graph
    '''
    def __init__(self, data_folder):
        """
        Args:
            data_folder (string): Path to the folder that contains everything.
        """
        
        self.generated = {}
        self.notes = {}

        SPECTROGRAM_REAL_PATH = data_folder + '/spectrograms_generated/'
        NOTE_GRAPH_PATH = data_folder + '/note_graphs/'

        # Get all the spectrograms in spectrograms_generated
        self.generated = get_file_names(SPECTROGRAM_REAL_PATH)
        self.notes = get_file_names(NOTE_GRAPH_PATH)

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        song_name = list(self.real)[idx]
        real_song = self.real[song_name]
        generated_song = self.generated[song_name]
        sample = {'real': np.load(real_song), 'notes': np.load(generated_song)}

        return sample

if __name__ == '__main__':
    dataset = DeNoiseDataset(output_path)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, sample in enumerate(data_loader):
        print(i, sample['real'].shape, sample['generated'].shape)