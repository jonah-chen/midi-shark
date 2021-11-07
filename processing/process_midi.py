import argparse
import csv
from collections import namedtuple
import numpy as np
from mido import MidiFile
import os

def midi2labels(midi_file_path):
    NoteState = namedtuple('NoteState', 'start velocity')
    mid = MidiFile(midi_file_path)
    note_states = [None] * 128
    durations_by_note = [[] for _ in range(128)]
    cur_time = 0  # elapsed time in milliseconds
    ticks_per_beat = mid.ticks_per_beat
    for track in mid.tracks:
        cur_time = 0
        cur_tempo = 500000 / 1000  # default: 500000 microseconds per beat
        for msg in track:
            cur_time += msg.time * cur_tempo / ticks_per_beat
            if msg.type == 'note_on':
                msg_vel = msg.velocity
                msg_note = msg.note
                if msg_vel == 0:
                    state = note_states[msg_note]
                    assert(state is not None)
                    durations_by_note[msg_note].append(
                        (state.start, cur_time, state.velocity))
                    note_states[msg_note] = None
                elif msg_vel > 0:
                    assert(note_states[msg_note] is None)
                    note_states[msg_note] = NoteState(cur_time, msg_vel)
            elif msg.type == 'set_tempo':
                cur_tempo = msg.tempo
            elif msg.type == 'time_signature':
                # not needed for determining the time in milliseconds
                continue
            elif msg.type == 'control_change':
                # ignoring control changes for now
                continue
            elif msg.type == 'program_change':
                continue
            elif msg.type == 'end_of_track':
                # track ends here
                continue
            else:
                print(f'unhandled message type {msg.type}')
    return durations_by_note

def save_midi(filename, output_name, file):
    a = midi2labels(filename)
    output = output_name + file
    output = output[:-5] + ".csv"
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['note', 'start', 'end', 'velocity'])
        for note, durations in enumerate(a):
            for (start, end, velocity) in durations:
                writer.writerow([note, start, end, velocity])

    return a

def save_notes(notes, folder_name, file):
    max_duration = max([float(i) for i in notes[1:,2]])
    notes_spectrogram = np.zeros((229,int(max_duration)))

    for i in notes[1:]:
        # get start and end time
        start = float(i[1])
        end = float(i[2])
        # get note
        note = int(i[0])
        # get velocity
        velocity = int(i[3])

        # Update notes_spectrogram
        notes_spectrogram[f_to_mel(note), int(start):int(end)] = 1


    n = notes_spectrogram.shape[1]/(20*1000)

    for i in range(int(n)): 
        t_start = i*20*1000
        t_end = (i+1)*20*1000

        duration = (t_end - t_start)*20/892
        a = notes_spectrogram[:,t_start:t_end]
        notes_spectrogram_final = np.zeros((229,862))
        ii = 0
        for j in [int(k) for k in np.linspace(0, t_end-t_start-1, 862)]:
            notes_spectrogram_final[:,ii] = a[:,j]
            ii += 1

        output_folder_name = folder_name + "/" + file[:-5]
        if not os.path.isdir(output_folder_name):
            os.makedirs(output_folder_name)

        output_name = output_folder_name+"/offset_"+str(i*20)+".0_duration_20"
        np.save(output_name, notes_spectrogram_final)

# Convert note to frequency
def f_to_mel(n):
    frequency = 440 * 2**((n-69)/12)
    # Normalize frequency between 0 and 1
    # frequency = 7.529e-5*(frequency-8.175798916)
    
    # Convert frequency to mel scale
    mel = 175.28862145395186*np.log10(1+frequency/700)

    return int(mel)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi_file', help='.midi file to parse', type=str)
    parser.add_argument('-o', '--out_csv_path', help='name of the csv file',
                        type=str, default='out.csv')
    args = parser.parse_args()
    durations_by_note = midi2labels(args.midi_file)
    with open(args.out_csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['note', 'start', 'end', 'velocity'])
        for note, durations in enumerate(durations_by_note):
            for (start, end, velocity) in durations:
                writer.writerow([note, start, end, velocity])
