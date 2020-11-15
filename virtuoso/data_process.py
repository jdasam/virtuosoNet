# import copy
import numpy as np
import random
import math

PITCH_VEC_IDX = 13
PITCH_SCL_IDX = 0


def key_augmentation(data_x, key_change, pitch_std):
    # key_change = 0
    if key_change == 0:
        return data_x
    pitch_start_index = PITCH_VEC_IDX
    shifted_pitch = np.zeros_like(data_x[:,:13])
    original_pitch = data_x[:,pitch_start_index:pitch_start_index+13]
    shifted_pitch[:, 0] = original_pitch[:, 0]
    if key_change > 0:
        shifted_pitch[:, 1+key_change:] = original_pitch[:, 1:-key_change]
        shifted_pitch[:, 1:1+key_change] = original_pitch[:, -key_change:]
        shifted_pitch[np.sum(original_pitch[:,-key_change:], axis=1)==1, 0] += 0.25
    else:
        shifted_pitch[:, 1:key_change] = original_pitch[:,1-key_change:]
        shifted_pitch[:, key_change:] = original_pitch[:,1:1-key_change ]
        shifted_pitch[np.sum(original_pitch[:,1:1-key_change ], axis=1)==1, 0] += 0.25
    # calculate octave shift


    data_x_aug = np.copy(data_x)
    data_x_aug[:,pitch_start_index:pitch_start_index+13] = shifted_pitch
    data_x_aug[:, PITCH_SCL_IDX] += key_change / pitch_std
    # for data in data_x_aug:
    #     octave = data[pitch_start_index]
    #     pitch_class_vec = data[pitch_start_index+1:pitch_start_index+13]
    #     pitch_class = pitch_class_vec.index(1)
    #     new_pitch = pitch_class + key_change
    #     if new_pitch < 0:
    #         octave -= 0.25
    #     elif new_pitch > 12:
    #         octave += 0.25
    #     new_pitch = new_pitch % 12

    #     new_pitch_vec = [0] * 13
    #     new_pitch_vec[0] = octave
    #     new_pitch_vec[new_pitch+1] = 1

    #     data[pitch_start_index: pitch_start_index+13] = new_pitch_vec
    #     data[PITCH_SCL_IDX] = data[PITCH_SCL_IDX] + key_change

    return data_x_aug


def make_slicing_indexes_by_measure(num_notes, measure_numbers, steps, overlap=True):
    slice_indexes = []
    if num_notes < steps:
        slice_indexes.append((0, num_notes))
    elif overlap:
        first_end_measure = measure_numbers[steps]
        last_measure = measure_numbers[-1]
        if first_end_measure < last_measure - 1:
            first_note_after_the_measure = measure_numbers.index(first_end_measure+1)
            slice_indexes.append((0, first_note_after_the_measure))
            second_end_start_measure = measure_numbers[num_notes - steps]
            first_note_of_the_measure = measure_numbers.index(second_end_start_measure)
            slice_indexes.append((first_note_of_the_measure, num_notes))

            if num_notes > steps * 2:
                first_start = random.randrange(int(steps/2), int(steps*1.5))
                start_measure = measure_numbers[first_start]
                end_measure = start_measure

                while end_measure < second_end_start_measure:
                    start_note = measure_numbers.index(start_measure)
                    if start_note+steps < num_notes:
                        end_measure = measure_numbers[start_note+steps]
                    else:
                        break
                    end_note = measure_numbers.index(end_measure-1)
                    slice_indexes.append((start_note, end_note))

                    if end_measure > start_measure + 2:
                        start_measure = end_measure - 2
                    elif end_measure > start_measure + 1:
                        start_measure = end_measure - 1
                    else:
                        start_measure = end_measure
        else:
            slice_indexes.append((0, num_notes))
    else:
        num_slice = math.ceil(num_notes / steps)
        prev_end_index = 0
        for i in range(num_slice):
            if prev_end_index + steps >= num_notes:
                slice_indexes.append((prev_end_index, num_notes))
                break
            end_measure = measure_numbers[prev_end_index + steps]
            if end_measure >= measure_numbers[-1]:
                slice_indexes.append((prev_end_index, num_notes))
                break
            first_note_after_the_measure = measure_numbers.index(end_measure + 1)
            slice_indexes.append((prev_end_index, first_note_after_the_measure))
            prev_end_index = first_note_after_the_measure
    return slice_indexes


def make_slice_with_same_measure_number(num_notes, measure_numbers, measure_steps):
    num_total_measure = measure_numbers[-1] + 1
    slice_indexes = []
    current_measure = 0
    current_note_idx = 0
    if num_total_measure < measure_steps:
        slice_indexes.append((0, num_notes))
    else:
        while current_measure + measure_steps < num_total_measure:
            current_slice_end_measure = current_measure + measure_steps
            if current_slice_end_measure == num_total_measure - 1:
                next_measure_end_note_idx = num_notes - 1
            else:
                next_measure_end_note_idx = measure_numbers.index(current_slice_end_measure+1)
            slice_indexes.append((current_note_idx, next_measure_end_note_idx))
            # move with random overlap
            current_measure = current_measure + int(measure_steps * (0.25 + random.random() * 0.5))
            current_note_idx = measure_numbers.index(current_measure)
        last_measure_start = num_total_measure - measure_steps
        last_measure_start_note_idx = measure_numbers.index(last_measure_start)
        slice_indexes.append((last_measure_start_note_idx, num_notes))

    return slice_indexes


def make_slicing_indexes_by_beat(beat_numbers, beat_steps, overlap=True):
    slice_indexes = []
    num_notes = len(beat_numbers)
    num_beats = beat_numbers[-1]
    if num_beats < beat_steps:
        slice_indexes.append((0, num_notes))
    elif overlap:
        first_end_beat = beat_steps
        last_end_beat = num_beats

        if first_end_beat < last_end_beat - 1:
            first_note_after_the_beat = beat_numbers.index(first_end_beat + 1)
            slice_indexes.append((0, first_note_after_the_beat))
            second_end_start_beat = num_beats - beat_steps
            first_note_of_the_beat = beat_numbers.index(second_end_start_beat)
            slice_indexes.append((first_note_of_the_beat, num_notes))
            if num_beats > beat_steps * 2:
                first_start = random.randrange(int(beat_steps / 2), int(beat_steps * 1.5))
                start_beat = first_start
                end_beat = start_beat

                while end_beat < second_end_start_beat:
                    start_note = beat_numbers.index(start_beat)
                    if start_beat + beat_steps < num_beats:
                        end_beat = start_beat + beat_steps
                    else:
                        break
                    end_note = beat_numbers.index(end_beat)
                    slice_indexes.append((start_note, end_note))

                    if end_beat > start_beat + 2:
                        start_beat = end_beat - 2
                    elif end_beat > start_beat + 1:
                        start_beat = end_beat - 1
                    else:
                        start_beat = end_beat

        else:
            slice_indexes.append((0, num_notes))
    return slice_indexes
