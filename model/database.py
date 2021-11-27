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
                        real_file_names[file_name] = os.path.join(
                            real_path, year, song, file)
                        gen_file_names[file_name] = os.path.join(
                            gen_path, year, song, file)
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

        SPECTROGRAM_GENERATED_PATH = os.path.join(
            data_folder, 'spectrograms_generated')
        SPECTROGRAM_REAL_PATH = os.path.join(data_folder + 'spectrograms_real')

        self.real, self.generated = get_file_names(
            SPECTROGRAM_REAL_PATH, SPECTROGRAM_GENERATED_PATH)
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


class OnsetsFramesVelocity(Dataset):
    '''
    Dataset that contains the real spectrogram (-1,229,862,) as input;
    and onsets, offsets, frames, and velocities (-1,88,1000,) as outputs
    '''

    def __init__(self, data_folder):
        """
        Args:
            data_folder (string): Path to the folder that contains everything.
        """

        SPECTROGRAM_REAL_PATH = os.path.join(data_folder, 'spectrograms_real')
        ONSET_PATH = os.path.join(data_folder, 'onsets')
        OFFSET_PATH = os.path.join(data_folder, 'offsets')
        FRAME_PATH = os.path.join(data_folder, 'frames')
        VELOCITY_PATH = os.path.join(data_folder, 'velocities')

        # Get all the names in spectrograms_real
        self.real, self.onsets, self.offsets, self.frames, self.velocities = {}, {}, {}, {}, {}
        for year in os.listdir(SPECTROGRAM_REAL_PATH):
            real_songs = os.listdir(os.path.join(SPECTROGRAM_REAL_PATH, year))
            onsets = os.listdir(os.path.join(ONSET_PATH, year))
            offsets = os.listdir(os.path.join(OFFSET_PATH, year))
            frames = os.listdir(os.path.join(FRAME_PATH, year))
            velocities = os.listdir(os.path.join(VELOCITY_PATH, year))

            for song in real_songs:
                if song in onsets:
                    real_file_dir = os.listdir(os.path.join(
                        SPECTROGRAM_REAL_PATH, year, song))
                    onsets_file_dir = os.listdir(
                        os.path.join(ONSET_PATH, year, song))
                    offsets_file_dir = os.listdir(
                        os.path.join(OFFSET_PATH, year, song))
                    frames_file_dir = os.listdir(
                        os.path.join(FRAME_PATH, year, song))
                    velocities_file_dir = os.listdir(
                        os.path.join(VELOCITY_PATH, year, song))

                    for file in real_file_dir:
                        # Ignore the end of the song (if less than 20s)
                        if file.endswith("20.npy") and file in onsets_file_dir:
                            file_name = f"{year}_{song}_{file}"
                            self.real[file_name] = os.path.join(
                                SPECTROGRAM_REAL_PATH, year, song, file)
                            self.onsets[file_name] = os.path.join(
                                ONSET_PATH, year, song, file)
                            self.offsets[file_name] = os.path.join(
                                OFFSET_PATH, year, song, file)
                            self.frames[file_name] = os.path.join(
                                FRAME_PATH, year, song, file)
                            self.velocities[file_name] = os.path.join(
                                VELOCITY_PATH, year, song, file)

        assert(len(self.real))
        assert(len(self.real) == len(self.onsets) == len(self.offsets)
               == len(self.frames) == len(self.velocities))

        self.real_list = list(self.real)

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        song_name = self.real_list[idx]
        real_song = self.real[song_name]
        onsets_song = self.onsets[song_name]
        offsets_song = self.offsets[song_name]
        frames_song = self.frames[song_name]
        velocities_song = self.velocities[song_name]
        sample = {'real': np.load(real_song),
                  'onsets': np.load(onsets_song),
                  'offsets': np.load(offsets_song),
                  'frames': np.load(frames_song),
                  'velocities': np.load(velocities_song)}

        if (sample['onsets'].shape[1] != 1000 or
            sample['offsets'].shape[1] != 1000 or
            sample['frames'].shape[1] != 1000 or
                sample['velocities'].shape[1] != 1000):
            print(song_name)
            raise Exception("Wrong shape")

        return sample


if __name__ == '__main__':
    dataset = OnsetsFramesVelocity(output_path)
    print(len(dataset))
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=4,
                             shuffle=True, num_workers=24, drop_last=True)
    for i, sample in enumerate(data_loader):
        print(i, sample['real'].shape, sample['onsets'].shape, sample['offsets'].shape,
              sample['frames'].shape, sample['velocities'].shape)
