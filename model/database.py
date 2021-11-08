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

        # Get all the spectrograms in spectrograms_generated
        for year in os.listdir(SPECTROGRAM_GENERATED_PATH):
            for song in os.listdir(SPECTROGRAM_GENERATED_PATH + "/" + year):
                for file in os.listdir(SPECTROGRAM_GENERATED_PATH + "/" + year + '/' + song):
                    if file.split('.')[-1] == "npy":
                        file_name = year + "_" + song + "_" + file
                        self.generated[file_name] = np.load(SPECTROGRAM_GENERATED_PATH + "/" + year + '/' + song + '/' + file)

        # Get all the spectrograms in spectrograms_real
        for year in os.listdir(SPECTROGRAM_REAL_PATH):
            for song in os.listdir(SPECTROGRAM_REAL_PATH + "/" + year):
                for file in os.listdir(SPECTROGRAM_REAL_PATH + "/" + year + '/' + song):
                    if file.split('.')[-1] == "npy":
                        file_name = year + "_" + song + "_" + file
                        self.real[file_name] = np.load(SPECTROGRAM_REAL_PATH + "/" + year + '/' + song + '/' + file)

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        song_name = list(self.real)[idx]
        real_song = self.real[song_name]
        generated_song = self.generated[song_name]
        sample = {'real': real_song, 'generated': generated_song}

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
        self.real = {}

        SPECTROGRAM_REAL_PATH = data_folder + '/spectrograms_generated/'
        NOTE_GRAPH_PATH = data_folder + '/note_graphs/'

        # Get all the spectrograms in spectrograms_generated
        for year in os.listdir(SPECTROGRAM_REAL_PATH):
            for song in os.listdir(SPECTROGRAM_REAL_PATH + "/" + year):
                for file in os.listdir(SPECTROGRAM_REAL_PATH + "/" + year + '/' + song):
                    if file.split('.')[-1] == "npy":
                        file_name = year + "_" + song + "_" + file
                        self.generated[file_name] = np.load(SPECTROGRAM_REAL_PATH + "/" + year + '/' + song + '/' + file)

        # Get all the spectrograms in spectrograms_real
        for year in os.listdir(NOTE_GRAPH_PATH):
            for song in os.listdir(NOTE_GRAPH_PATH + "/" + year):
                for file in os.listdir(NOTE_GRAPH_PATH + "/" + year + '/' + song):
                    if file.split('.')[-1] == "npy":
                        file_name = year + "_" + song + "_" + file
                        self.real[file_name] = np.load(NOTE_GRAPH_PATH + "/" + year + '/' + song + '/' + file)

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        song_name = list(self.real)[idx]
        real_song = self.real[song_name]
        generated_song = self.generated[song_name]
        sample = {'real': real_song, 'notes': generated_song}

        return sample

if __name__ == '__main__':
    dataset = DeNoiseDataset(output_path)
    print(dataset[0])