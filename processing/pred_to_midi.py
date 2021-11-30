from midiutil.MidiFile import MIDIFile
import numpy as np
import os
from dotenv import load_dotenv
import argparse
from midi2audio import FluidSynth, DEFAULT_SAMPLE_RATE, DEFAULT_SOUND_FONT

def create_frames_velocity_dict(frames, velocities = None):
    '''
        Creates two dictionaries where the keys are (note, time)
        and the values are the duration and velocity of the note.

        If velocities = None, then the velocity is set to 50.
    '''
    note_dict = {}
    velocity_dict = {}

    for i in range(len(frames)):
        # Go through every note:
        j_last = -1

        for j in range(len(frames[0])):
            
            is_played = (frames[i][j] == 1)
            if j == 0: is_onset = is_played
            else: is_onset = is_played and frames[i][j-1] == 0

            if is_played and is_onset:
                note_dict[(i+21, j)] = 1 # Note is on

                # Set Velocity
                if type(velocities) == type(None):
                    velocity_dict[(i+21, j)] = 50/127
                else:
                    velocity_dict[(i+21, j)] = velocities[i][j]
                j_last = j # Reset the onset time

            elif is_played and not is_onset:
                note_dict[(i+21, j_last)] += 1

    return note_dict, velocity_dict

def write_to_midi(output_file, note_dict, velocity_dict):
    '''
        Takes in the note duration and velocity dictionaries and writes
        them to a midi file.

        The midi file is saved to the output_file.
    '''

    mf = MIDIFile(1) # only 1 track
    track = 0   # the only track
    time = 0    # start at the beginning

    beat_f = 2 # how many times the bpm
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 60*beat_f)
    hop_length=20*1000/862
    f = hop_length/(1000)*beat_f

    # add some notes
    channel = 0

    for i in range(len(note_dict)):
        onset_list = list(note_dict.keys())
        note = int(onset_list[i][0])

        time = np.ceil(onset_list[i][1]*f*16)/16 # Round up to the nearest 16th note
        duration = np.ceil(note_dict[onset_list[i]]*f*16)/16 # Round up to the nearest 16th note


        velocity = int(list(velocity_dict.values())[i]*127)

        mf.addNote(track, channel, note, time, duration, velocity)

    with open(output_file, 'wb') as f:
        mf.writeFile(f)

def merge_songs(frame_input_folder, velocity_input_folder, output_file):
    '''
        Takes in a folder of 20s numpy files and combines them 
        to a single numpy file
    '''
    frame_list = os.listdir(frame_input_folder)
    frame_list = sorted(frame_list, key=lambda x: float(x.split('_')[1].split('.')[0]))

    frames = None
    velocities = None

    for f in frame_list:
        frame_single = np.load(os.path.join(frame_input_folder, f))
        
        if type(velocity_input_folder) != type(None):
            vel_single = np.load(os.path.join(velocity_input_folder, f))

        # Initialize at the first frame
        if frames is None:
            frames = frame_single
            if type(velocity_input_folder) != type(None):
                velocities = vel_single

        # Concatenate the frames
        else:
            frames = np.concatenate((frames, frame_single), axis=1)
            if type(velocity_input_folder) != type(None):
                velocities = np.concatenate((velocities, vel_single), axis=1)
    
    return frames, velocities

def folder_to_midi(frame_input_folder, velocity_input_folder, output_file, audio = True, sound_font=DEFAULT_SOUND_FONT, sample_rate=DEFAULT_SAMPLE_RATE):
    '''
        Takes in folders of 20s numpy files and combines them into a single
        midi file. 

        audio: Creates audio as well if true
    '''
    frames, velocities = merge_songs(frame_input_folder, velocity_input_folder, output_file)
    note_dict, velocity_dict = create_frames_velocity_dict(frames, velocities)
    write_to_midi(output_file, note_dict, velocity_dict)
    if audio == True:
        fs = FluidSynth(sound_font, sample_rate)
        fs.midi_to_audio(output_file, output_file[:-5]+'.wav')

if __name__ == '__main__':
    '''
        Takes in a folder of 20s numpy files and combines them 
        to a single numpy file
    '''
    load_dotenv(verbose=True)
    input_path = os.environ.get('pathname')
    output_path = os.environ.get('dataname')

    parser = argparse.ArgumentParser(description='Type in frame_folder, velocity_folder, output_name:')
    parser.add_argument('--f', help='path to input frames folder', type=str)
    parser.add_argument('--v', help='path to input velocities folder', type=str)
    parser.add_argument('--o', help='output midi file name', type=str)

    args = parser.parse_args()
    if args.v == 'None':
        args.v = None
    folder_to_midi(args.f, args.v, args.o)