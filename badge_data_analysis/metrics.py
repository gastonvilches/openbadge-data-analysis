
# To do: comment and document all
#
#        Make functions that runs the typical pipeline

import numpy as np

def speaking_time(data):
    window = data.window_length
    for member in data.members:
        data[member]['speaking_time'] = window*np.sum(data[member]['real_speak'])
        data[member]['speaking_time'] /= data.meeting_duration
    return data

def overlap_time(data):
    for member in data.members:
        data[member]['overlap_time'] = 0.0
    for win in range(data.number_of_windows):
        speaking = [m for m in data.members if bool(data[m]['real_speak'][win])]
        if len(speaking) >= 2:
            for member in speaking:
                data[member]['overlap_time'] += data.window_length
    for member in data.members:
        data[member]['overlap_time'] /= data.meeting_duration
    return data

def _fill_speech_gaps(data, max_gap=1):
    key = 'real_speak_filled_' + str(max_gap)
    for member in data.members:
        data[member][key] = data[member]['real_speak'].copy()
    for m in data.members:    
        for w in range(data.number_of_windows-2):
            if data[m]['real_speak'][w] and not data[m]['real_speak'][w+1]:
                gap = 1
                limit = np.clip(max_gap+1, None, data.number_of_windows-1-w)
                for d in range(2, limit+1):
                    if data[m]['real_speak'][w+d]:
                        break
                    gap += 1
                if gap <= max_gap:
                    data[m][key][w+1:w+gap+1] = 1
    return data

def overlap_count(data, fill_gaps=False, max_gap=1):
    for member in data.members:
        data[member]['overlap_count'] = 0
    prev_overlap = np.zeros((len(data.members),))
    key = 'real_speak_filled_' + str(max_gap) if fill_gaps else 'real_speak'
    if fill_gaps and not key in data[data.members[0]].keys():
        _fill_speech_gaps(data, max_gap)
    for w in range(data.number_of_windows):
        speaking = np.array([data[m][key][w] for m in data.members])
        if np.sum(speaking) >= 2:
            for idx, m in enumerate(data.members):
                if speaking[idx] and not prev_overlap[idx]:
                    data[m]['overlap_count'] += 1
                    prev_overlap[idx] = 1
        prev_overlap[np.where(speaking==0)[0]] = 0
    return data

def turn_taking(data, min_succesive_non_overlap=2, fill_gaps=False, max_gap=1):
    for member in data.members:
        data[member]['turn_taking_count'] = 0
    key = 'real_speak_filled_' + str(max_gap) if fill_gaps else 'real_speak'
    if fill_gaps and not key in data[data.members[0]].keys():
        _fill_speech_gaps(data, max_gap)
    non_overlap_accum = np.zeros((len(data.members),))
    active_speaker_idx = -1
    for w in range(data.number_of_windows):
        speaking = np.array([data[m][key][w] for m in data.members])
        if np.sum(speaking) == 1:
            speaker_idx = np.where(speaking)[0][0]
            if speaker_idx != active_speaker_idx:
                non_overlap_accum[speaker_idx] += 1
                if non_overlap_accum[speaker_idx] >= min_succesive_non_overlap:
                    active_speaker_idx = speaker_idx
                    data[data.members[speaker_idx]]['turn_taking_count'] += 1
                    non_overlap_accum = np.zeros((len(data.members),))
    return data



































