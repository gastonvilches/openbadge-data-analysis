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

def calculate_indicators(data, print_results=True, round_decimals=2):
    p_values = np.array([data[m]['speaking_time'] for m in data.members])
    o_values = np.array([data[m]['overlap_time'] for m in data.members])
    oc_values = np.array([data[m]['overlap_count'] for m in data.members])
    tt_values = np.array([data[m]['turn_taking_count'] for m in data.members])
    
    p_cv = np.std(p_values)/np.mean(p_values)
    dominance = p_values.max()/np.mean(p_values[p_values != p_values.max()])
    
    total_p = np.sum(p_values)
    total_o = np.sum(o_values)
    total_cp = total_p - total_o
    
    total_oc = np.sum(oc_values)
    total_tt = np.sum(tt_values)
    
    avg_s_segm = total_p/total_tt*data.meeting_duration
    avg_o_segm = total_o/total_oc*data.meeting_duration
    ttf = total_tt/(data.meeting_duration/60)
        
    if print_results:
        d = round_decimals
        print('Team vocalization distribution:')
        print('    Coefficient of variation:  ', np.round(p_cv, d))
        print('    Dominance:                 ', np.round(dominance, d))
        print('    Total team participation:  ', str(np.round(100*total_p, d)) + '%')
        print('    Clean participation:       ', str(np.round(100*total_cp, d)) + '%')
        print('Turn taking:')
        print('    Team turn taking frequency:', np.round(ttf, d), 'Turns/min')
        print('    Avgerage speech segment:   ', np.round(avg_s_segm, d), 'sec')
        print('Overlapping speech:')
        print('    Total overlap time:        ', str(np.round(100*total_o, d)) + '%')
        print('    Avgerage overlap duration: ', np.round(avg_o_segm, d), 'sec')
        
    return (p_values, p_cv, dominance, total_p, total_cp, tt_values, ttf,
            avg_s_segm, o_values, oc_values, total_o, avg_o_segm)
    
    
    
    
    






























