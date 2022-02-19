import copy
import pretty_midi

from .xml_midi_matching import make_available_note_feature_list
from .utils import get_item_by_xml_position
from .xml_utils import get_measure_accidentals, cal_up_trill_pitch
from .feature_utils import Tempo
from . import pedal_cleaning


def apply_tempo_perform_features(score, features, start_time=0, predicted=False, return_tempo=False):
    # score: ScoreData, features: perform features in dictionary of list
    # predicted: Whether the feature is generated from model or extracted from performance. 
    #            It is for handling missing features from perform feature extraction

    beats = score.xml_obj.get_beat_positions()
    xml_notes = copy.deepcopy(score.xml_notes)
    num_beats = len(beats)
    num_notes = score.num_notes
    tempos = []
    ornaments = []
    # xml_positions = [x.note_duration.xml_position for x in xml_notes]
    previous_position = None
    current_sec = start_time
    key_signatures = score.xml_obj.get_key_signatures()
    trill_accidentals = score.xml_obj.get_accidentals

    valid_notes = make_available_note_feature_list(xml_notes, features, predicted=predicted)
    previous_tempo = 0

    # apply tempo
    for i in range(num_beats - 1):
        beat = beats[i]
        feat = get_item_by_xml_position(valid_notes, beat)
        start_position = feat['xml_position']
        if start_position == previous_position:
            continue

        # if predicted:
        #     qpm_saved = 10 ** feat['beat_tempo']
        #     num_added = 1
        #     next_beat = beats[i+1]
        #     start_index = feat['index']
        #     for j in range(1,20):
        #         if start_index-j < 0:
        #             break
        #         previous_note = xml_notes[start_index-j]
        #         previous_pos = previous_note.note_duration.xml_position
        #         if previous_pos == start_position:
        #             qpm_saved += 10 ** features['beat_tempo'][start_index-j]
        #             num_added += 1
        #         else:
        #             break

        #     for j in range(1,40):
        #         if start_index + j >= num_notes:
        #             break
        #         next_note = xml_notes[start_index+j]
        #         next_position = next_note.note_duration.xml_position
        #         if next_position < next_beat:
        #             qpm_saved += 10 ** features['beat_tempo'][start_index+j]
        #             num_added += 1
        #         else:
        #             break

        #     qpm = qpm_saved / num_added
        # else:
            # qpm = 10 ** feat['beat_tempo']
        qpm = 10 ** feat['beat_tempo']
        # qpm = 10 ** feat.qpm
        divisions = feat['divisions']

        if previous_tempo != 0:
            passed_second = (start_position - previous_position) / divisions / previous_tempo * 60
        else:
            passed_second = 0
        current_sec += passed_second
        tempo = Tempo(start_position, qpm, time_position=current_sec, end_xml=0, end_time=0)
        if len(tempos) > 0:
            tempos[-1].end_time = current_sec
            tempos[-1].end_xml = start_position

        tempos.append(tempo)

        previous_position = start_position
        previous_tempo = qpm

    # apply note 
    feature_by_note = [dict(zip(features,t)) for t in zip(*features.values())]
    xml_notes = apply_feature_to_notes(xml_notes, feature_by_note, tempos)
    # xml_notes, ornaments = apply_trills(xml_notes, feature_by_note, key_signatures, trill_accidentals)
    xml_notes = apply_duration_for_grace_note(xml_notes)

    xml_notes = xml_notes + ornaments
    xml_notes.sort(key=lambda x: (x.note_duration.xml_position, x.note_duration.time_position, -x.pitch[1]) )
    
    if return_tempo:
        return xml_notes, tempos
    else:
        return xml_notes

def cal_time_position_with_tempo(note, xml_dev, tempos):
    corresp_tempo = get_item_by_xml_position(tempos, note)
    previous_sec = corresp_tempo.time_position
    passed_duration = note.note_duration.xml_position + xml_dev - corresp_tempo.xml_position
    # passed_duration = note.note_duration.xml_position - corresp_tempo.xml_position
    passed_second = passed_duration / note.state_fixed.divisions / corresp_tempo.qpm * 60

    return previous_sec + passed_second
    
def apply_feature_to_notes(xml_notes, feature_by_note, tempos):
    # for i, note in enumerate(xml_notes):
    prev_vel = 64
    for note, feat in zip(xml_notes, feature_by_note):
        if not feat["onset_deviation"] == None:
            xml_deviation = feat["onset_deviation"] * note.state_fixed.divisions
        else:
            xml_deviation = 0

        note.note_duration.time_position = cal_time_position_with_tempo(note, xml_deviation, tempos)

        end_note = copy.copy(note)
        end_note.note_duration = copy.copy(note.note_duration)
        end_note.note_duration.xml_position = note.note_duration.xml_position + note.note_duration.duration

        end_position = cal_time_position_with_tempo(end_note, 0, tempos)

        # handle trill notes
        # if note.note_notations.is_trill:
        #     note, temp_ornaments = make_trill_notes(note, feat, end_position, prev_vel)
        # else:
        note.note_duration.seconds = end_position - note.note_duration.time_position

        note, prev_vel = apply_feat_to_a_note(note, feat, prev_vel)
    return xml_notes

def apply_trills(xml_notes, feature_by_note, key_signatures, trill_accidentals):
    ornaments = []
    for i, note in enumerate(xml_notes):
        feat = feature_by_note[i]
        if note.note_notations.is_trill:
            end_position = note.note_duration.time_position + note.note_duration.seconds
            note, temp_ornaments = make_trill_notes(note, feat, end_position, key_signatures, trill_accidentals, xml_notes, i)
            ornaments += temp_ornaments
    return xml_notes, ornaments

def apply_feat_to_a_note(note, feat, prev_vel):
    if not feat["articulation"] == None:
        note.note_duration.seconds *= 10 ** (feat['articulation'])
        # note.note_duration.seconds *= feat.articulation
    if not feat["velocity"] == None:
        note.velocity = feat["velocity"]
        prev_vel = note.velocity
    else:
        note.velocity = prev_vel
    if not feat['pedal_at_start'] == None:
        note.pedal.at_start = int(round(feat["pedal_at_start"]))
        note.pedal.at_end = int(round(feat["pedal_at_end"]))
        note.pedal.refresh = int(round(feat["pedal_refresh"]))
        note.pedal.refresh_time = feat["pedal_refresh_time"]
        note.pedal.cut = int(round(feat["pedal_cut"]))
        note.pedal.cut_time = feat["pedal_cut_time"]
        note.pedal.soft = int(round(feat["soft_pedal"]))
    return note, prev_vel

def apply_duration_for_grace_note(xml_notes):
    for i, note in enumerate(xml_notes):
        if note.note_duration.is_grace_note and note.note_duration.duration == 0:
            for j in range(i+1, len(xml_notes)):
                next_note = xml_notes[j]
                if not next_note.note_duration.duration == 0 \
                    and next_note.note_duration.xml_position == note.note_duration.xml_position \
                    and next_note.voice == note.voice:
                    next_second = next_note.note_duration.time_position
                    note.note_duration.seconds = (next_second - note.note_duration.time_position) / note.note_duration.num_grace
                    break
    return xml_notes

def make_trill_notes(note, feat, end_position, key_signatures, trill_accidentals, xml_notes,i):
    ornaments = []
    if note.note_notations.is_trill:
        note, _ = apply_feat_to_a_note(note, feat, note.velocity)
        trill_vec = feat["trill_param"]
        trill_density = trill_vec[0]
        last_velocity = trill_vec[1] * note.velocity
        first_note_ratio = trill_vec[2]
        last_note_ratio = trill_vec[3]
        up_trill = trill_vec[4]
        total_second = end_position - note.note_duration.time_position
        num_trills = int(trill_density * total_second)
        first_velocity = note.velocity

        key = get_item_by_xml_position(key_signatures, note)
        key = key.key
        final_key = None
        for acc in trill_accidentals:
            if acc.xml_position == note.note_duration.xml_position:
                if acc.type['content'] == '#':
                    final_key = 7
                elif acc.type['content'] == '♭':
                    final_key = -7
                elif acc.type['content'] == '♮':
                    final_key = 0
        measure_accidentals = get_measure_accidentals(xml_notes, i)
        trill_pitch = note.pitch
        up_pitch, up_pitch_string = cal_up_trill_pitch(note.pitch, key, final_key, measure_accidentals)

        if up_trill:
            up = True
        else:
            up = False

        if num_trills > 2:
            mean_second = total_second / num_trills
            normal_second = (total_second - mean_second * (first_note_ratio + last_note_ratio)) / (num_trills -2)
            prev_end = note.note_duration.time_position
            for j in range(num_trills):
                if up:
                    pitch = (up_pitch_string, up_pitch)
                    up = False
                else:
                    pitch = trill_pitch
                    up = True
                if j == 0:
                    note.pitch = pitch
                    note.note_duration.seconds = mean_second * first_note_ratio
                    prev_end += mean_second * first_note_ratio
                else:
                    new_note = copy.copy(note)
                    new_note.pedals = None
                    new_note.pitch = copy.copy(note.pitch)
                    new_note.pitch = pitch
                    new_note.note_duration = copy.copy(note.note_duration)
                    new_note.note_duration.time_position = prev_end
                    if j == num_trills -1:
                        new_note.note_duration.seconds = mean_second * last_note_ratio
                    else:
                        new_note.note_duration.seconds = normal_second
                    new_note.velocity = copy.copy(note.velocity)
                    new_note.velocity = first_velocity + (last_velocity - first_velocity) * (j / num_trills)
                    prev_end += new_note.note_duration.seconds
                    ornaments.append(new_note)
        elif num_trills == 2:
            mean_second = total_second / num_trills
            prev_end = note.note_duration.time_position
            for j in range(2):
                if up:
                    pitch = (up_pitch_string, up_pitch)
                    up = False
                else:
                    pitch = trill_pitch
                    up = True
                if j == 0:
                    note.pitch = pitch
                    note.note_duration.seconds = mean_second * first_note_ratio
                    prev_end += mean_second * first_note_ratio
                else:
                    new_note = copy.copy(note)
                    new_note.pedals = None
                    new_note.pitch = copy.copy(note.pitch)
                    new_note.pitch = pitch
                    new_note.note_duration = copy.copy(note.note_duration)
                    new_note.note_duration.time_position = prev_end
                    new_note.note_duration.seconds = mean_second * last_note_ratio
                    new_note.velocity = copy.copy(note.velocity)
                    new_note.velocity = last_velocity
                    prev_end += mean_second * last_note_ratio
                    ornaments.append(new_note)
        else:
            note.note_duration.seconds = total_second
    return note, ornaments
