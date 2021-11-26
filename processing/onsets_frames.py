import numpy as np
import os

def save_notes(notes, output_folder, hop_length = 20):
    '''Generate the frame vectors

    Args:
        notes (np.array): Loaded notes file in the format of
                          [note, start_time, end_time, velocity]
        output_folder (str): Output folder name (i.e. "../data/frames")
        hop_length (int, optional): Hop length in milliseconds. Defaults to 20.
    '''

    max_duration = max([float(i) for i in notes[1:, 2]])
    notes_spectrogram = np.zeros((88, int(max_duration/hop_length)))
    
    # Set a 1 in notes_spectrogram for each note between start and end time
    for i in notes[1:]:
        # get start and end time
        start = int(i[1])
        end = int(i[2])

        # get note
        note = int(i[0])

        # Update notes_spectrogram
        notes_spectrogram[note, int(start):int(end)] = 1

    divide_notes(notes, output_folder, "frames", hop_length)

def divide_notes(notes, folder_name, name, hop_length = 20):
    '''
    Divide notes into multiple 20s intervals.

    Args:
        notes (np.array): Loaded notes file in the format of
                            [note, start_time, end_time, velocity]
        folder_name (str): Output folder name (i.e. "../data/frames")
        name (str): Name of the file (i.e. "frames")
        hop_length (int, optional): Hop length in milliseconds. Defaults to 20.
    '''
    for i in np.arange(0, notes.shape[1], int(20*1000/hop_length)):
        t_start = int(i)
        t_end = int(i+20*1000/hop_length)
        notes_spectrogram_final = notes[:, t_start:t_end]

        output_folder_name = folder_name + "/" + name
        if not os.path.isdir(output_folder_name):
            os.makedirs(output_folder_name)

        output_name = f"{output_folder_name}/{name}_{i*hop_length/1000}_duration_20"
        np.save(output_name, notes_spectrogram_final)