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
def get_file_names(real_path, gen_path):
    real_file_names, gen_file_names = {}, {}
    for year in os.listdir(real_path):
        real_songs = os.listdir(os.path.join(real_path, year))
        gen_songs = os.listdir(os.path.join(gen_path, year))
        for song in real_songs:
            if song in gen_songs:
                gen_file_dir = os.listdir(os.path.join(gen_path, year, song))
                real_file_dir = os.listdir(os.path.join(real_path, year, song))
                for file in real_file_dir:
                    # Ignore the end of the song (if less than 20s)
                    if file.endswith("20.npy") and file in gen_file_dir:
                        file_name = f"{year}_{song}_{file}"
                        real_file_names[file_name] = os.path.join(real_path, year, song, file)
                        gen_file_names[file_name] = os.path.join(gen_path, year, song, file)
    return real_file_names, gen_file_names

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

        SPECTROGRAM_GENERATED_PATH = os.path.join(data_folder, 'spectrograms_generated')
        SPECTROGRAM_REAL_PATH = os.path.join(data_folder + 'spectrograms_real')

        self.real, self.generated = get_file_names(SPECTROGRAM_REAL_PATH, SPECTROGRAM_GENERATED_PATH)
        assert(len(self.real) == len(self.generated))
        assert(len(self.real))

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        song_name = list(self.real)[idx]
        real_song = self.real[song_name]
        generated_song = self.generated[song_name]
        sample = {'real': np.load(real_song), 
                  'generated': np.load(generated_song)}

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
        
        self.real = {}
        self.notes = {}

        SPECTROGRAM_REAL_PATH = os.path.join(data_folder, 'spectrograms_generated')
        NOTE_GRAPH_PATH = os.path.join(data_folder, 'note_graphs')

        # Get all the spectrograms in spectrograms_generated
        self.real, self.notes = get_file_names(SPECTROGRAM_REAL_PATH, NOTE_GRAPH_PATH)

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        song_name = list(self.real)[idx]
        real_song = self.real[song_name]
        notes = self.notes[song_name]
        sample = {'real': np.load(real_song), 
                 'notes': np.load(notes)}

        return sample

if __name__ == '__main__':
    dataset = DeNoiseDataset(output_path)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=24)
    for i, sample in enumerate(data_loader):
        print(i, sample['real'].shape, sample['generated'].shape)
        print(torch.min(sample['real']), torch.max(sample['real']))
        print(torch.min(sample['generated']), torch.max(sample['generated']))