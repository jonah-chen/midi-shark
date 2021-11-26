import numpy as np
import os
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor


def divide_notes(notes, folder_name, name, hop_length=20):
    '''
    Divide notes into multiple 20s intervals.

    Args:
        notes (np.array): Loaded notes file in the format of
                            [note, start_time, end_time, velocity]
        folder_name (str): Output folder name (i.e. "../data/frames/2018")
        name (str): Name of the file (i.e. "frames")
        hop_length (int, optional): Hop length in milliseconds. Defaults to 20.
    '''
    for i in np.arange(0, notes.shape[1], int(20*1000/hop_length)):
        t_start = int(i)
        t_end = int(i+20*1000/hop_length)
        notes_spectrogram_final = notes[:, t_start:t_end]

        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        output_name = os.path.join(
            folder_name, f"offset_{i*hop_length/1000}_duration_20")
        np.save(output_name, notes_spectrogram_final)


def save_onsets(notes, output_folder, hop_length=20):
    """
    Generate the onset vectors

    Args:
        notes (np.array): Loaded notes file in the format of 
                          [note,start,end,velocity]
        output_folder (str): Output folder name (i.e. "../data/onsets/2018")
        hop_length (int, optional): Length of each bin in milliseconds. 
                                    Defaults to 20.

    Returns:
        np.array: Onset vectors per time
    """
    length = int(np.ceil(np.max(notes[:, 2])/hop_length))
    onsets = np.zeros((88, length), dtype=np.uint8)
    for note in notes:
        index = round(note[1]/hop_length)
        onsets[int(note[0])-21, index] = 1

    divide_notes(onsets, output_folder, hop_length)


def save_velocities(notes, output_folder, hop_length=20):
    """
    Generate the velocity vectors and saves them to the output folder.

    Args:
        notes (np.array): Loaded notes file in the format of 
                          [note,start,end,velocity]
        output_folder (str): Output folder name (i.e. "../data/velocities/2018")
        hop_length (int, optional): Length of each bin in milliseconds. 
                                    Defaults to 20.
    """
    length = int(np.ceil(np.max(notes[:, 2])/hop_length))
    velocities = np.zeros((88, length), dtype=np.float32)
    for note in notes:
        index = round(note[1]/hop_length)
        velocities[int(note[0])-21, index] = note[3]/127

    divide_notes(velocities, output_folder, hop_length)


def save_frames(notes, output_folder, hop_length=20):
    '''Generate the frame vectors

    Args:
        notes (np.array): Loaded notes file in the format of
                          [note, start_time, end_time, velocity]
        output_folder (str): Output folder name (i.e. "../data/frames/2018")
        hop_length (int, optional): Hop length in milliseconds. Defaults to 20.
    '''

    length = int(np.ceil(np.max(notes[:, 2])/hop_length))
    frames = np.zeros((88, length))

    # Set a 1 in notes_spectrogram for each note between start and end time
    for i in notes:
        # get start and end time
        start = int(i[1])
        end = int(i[2])
        # get note
        note = int(i[0])-21

        # Update notes_spectrogram
        frames[note, int(start):int(end)] = 1

    divide_notes(frames, output_folder, hop_length)


if __name__ == '__main__':
    load_dotenv(verbose=True)
    data_root = os.environ.get('dataname')
    notes_root = os.path.join(data_root, "notes_generated")

    for year in os.listdir(notes_root):
        for filename in os.listdir(os.path.join(notes_root, year)):
            input_path = os.path.join(notes_root, year, filename)
            notes = np.load(input_path)

            with ProcessPoolExecutor(max_workers=24) as executor:
                executor.submit(save_onsets, notes, os.path.join(
                    data_root, "onsets", year, filename))
                executor.submit(save_velocities, notes, os.path.join(
                    data_root, "velocities", year, filename))
                executor.submit(save_frames, notes, os.path.join(
                    data_root, "frames", year, filename))
