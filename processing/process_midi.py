import argparse
import csv
from collections import namedtuple

from mido import MidiFile


def midi2labels(midi_file_path):
    NoteState = namedtuple('NoteState', 'start velocity')
    mid = MidiFile(midi_file_path)
    note_states = [None] * 128
    durations_by_note = [[] for _ in range(128)]
    cur_time = 0
    for track in mid.tracks:
        cur_time = 0
        for msg in track:
            cur_time += msg.time
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
            else:
                print(f'unhandled message type {msg.type}')
    return durations_by_note


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
