import argparse
from collections import namedtuple

import numpy as np

from mido import MidiFile


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi_file', help='.midi file to parse', type=str)
    parser.add_argument('-o', '--out_path', help='name of the npy file',
                        type=str, default='out.npy')
    return parser.parse_args()


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
                print(msg)
            elif msg.type == 'end_of_track':
                # track ends here
                continue
            else:
                print(f'unhandled message type {msg.type}')
    return durations_by_note


if __name__ == '__main__':
    args = get_args()

    durations_by_note = midi2labels(args.midi_file)

    arr = None
    for note, durations in enumerate(durations_by_note):
        for (start, end, velocity) in durations:
            row = np.array([note, start, end, velocity])
            if arr is None:
                arr = row
            else:
                arr = np.vstack((arr, row))
    np.save(args.out_path, arr)
