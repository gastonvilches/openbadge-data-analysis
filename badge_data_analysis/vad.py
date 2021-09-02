import numpy as np
from scipy.stats import gaussian_kde

def xcorr(x, y, max_lag=None, normalize=True, eps=1e-5):
    if len(x) > len(y):
        s, l = y, x
    else:
        s, l = x, y
    if max_lag == None:
        init_pad = len(s) - 1
        end_pad = len(s) - 1
        out_len = len(s) + len(l) - 1
    else:
        init_pad = max_lag
        end_pad = np.clip(max_lag - (len(l) - len(s)), 0, None)
        out_len = 2*max_lag + 1
    l = np.pad(l, (init_pad, end_pad))
    corr = np.zeros((out_len,))
    s_energy = np.sum(s**2)
    for lag in range(out_len):
        corr[lag] = np.sum(l[lag:lag+len(s)]*s)
        if normalize:
            corr[lag] /= np.sqrt(np.sum(l[lag:lag+len(s)]**2)*s_energy)
    return corr

def genuine_speak(data, window=1.0, max_temp_shift=0.15, corr_thr=0.85, 
                  silence_thr_mean=1.0, silence_thr_std=0.0): #min_num_samples=0.8
    
    # From now on the sample period it is assumed constant
    sample_period = np.diff(data[data.members[0]]['time'][:2])[0]
    max_corr_lag = int(np.round(max_temp_shift/sample_period))
    
    # Get meeting start and number of windows
    start_time = data.meeting_start
    end_time = data.meeting_end
    num_win = int(np.ceil((end_time - start_time)/window))
    
    # Memory alloc
    for member in data.members:
        data[member]['gen_speak'] = np.zeros((num_win,), dtype=np.int16)
        data[member]['win_mean'] = np.zeros((num_win,))
        data[member]['win_std'] = np.zeros((num_win,))
        data[member]['win_time'] = np.linspace(start_time, start_time
                                               + (num_win-1)*window, num_win)
        data[member]['global_mean'] = np.mean(data[member]['signal'])
        data[member]['global_std'] = np.std(data[member]['signal'])
        data[member]['is_beacon'] = False
    
    # Genuine Speak
    for w_i in range(num_win-1):
        genuine = True
        win_start, win_end = (start_time+w_i*window, start_time+(w_i+1)*window)
        for member in data.members:
            idx = np.logical_and(data[member]['time'] >= win_start,
                                   data[member]['time'] < win_end)
            data[member]['win_mean'][w_i] = np.mean(data[member]['signal'][idx])
            data[member]['win_std'][w_i] = np.std(data[member]['signal'][idx])
        
        means = np.array([data[mem]['win_mean'][w_i] for mem in data.members])
        max_vol = np.max(means)
        max_vol_mem = data.members[np.where(means == max_vol)[0][0]]
        
        # This is to avoid false detections when there is silence
        if max_vol < (silence_thr_mean*data[max_vol_mem]['global_mean'] 
                      + silence_thr_std*data[max_vol_mem]['global_std']):
            continue
        
        w_idx_max = np.logical_and(data[max_vol_mem]['time'] >= win_start,
                                   data[max_vol_mem]['time'] < win_end)
        for member in data.members:
            if member == max_vol_mem:
                continue
            idx = np.logical_and(data[member]['time'] >= win_start,
                                   data[member]['time'] < win_end)
            corr = np.max(xcorr(data[max_vol_mem]['signal'][w_idx_max], 
                                data[member]['signal'][idx], max_corr_lag))
            if corr < corr_thr:
                genuine = False
                break
        if genuine:
            for member in data.members:
                if member == max_vol_mem:
                    data[member]['gen_speak'][w_i] = 1
                else:
                    data[member]['gen_speak'][w_i] = -1
    return data

def _positive_kde(member_data, key, bandwidth=None, xi_max=60, step=0.1):
    xi = np.arange(0, xi_max, step)
    if bandwidth == None:
        kde_speak = gaussian_kde(member_data[key][member_data['gen_speak'] > 0])
        kde_silen = gaussian_kde(member_data[key][member_data['gen_speak'] < 0])
    else:
        kde_speak = gaussian_kde(member_data[key][member_data['gen_speak'] > 0],
                                 bw_method=bandwidth)
        kde_silen = gaussian_kde(member_data[key][member_data['gen_speak'] < 0],
                                 bw_method=bandwidth)
    f_speak = kde_speak(xi) + kde_speak(-xi)
    f_silen = kde_silen(xi) + kde_silen(-xi)
    return xi, f_speak, f_silen

def _detect_thr(member_data, key, bandwidth=None):
    xi, f_speak, f_silen = _positive_kde(member_data, key, bandwidth)
    
    intersection = 0
    intersection_found = None
    for idx in range(len(xi)):
        if intersection_found == None: # Esto es porque a veces empieza siendo más grande f_speak
            if f_silen[idx] > f_speak[idx]:
                intersection_found = False
            continue
        if f_speak[idx] > f_silen[idx]:
            intersection = xi[idx]
            intersection_found = True
            break
        
    if not intersection_found:
        print('Intersection between silent and speaking histograms not found')
        
    return intersection

def calculate_thresholds(data, bandwidth=None):
    for member in data.members:
        if np.any(data[member]['gen_speak'] > 0):
            data[member]['thr_mean'] = _detect_thr(data[member], 'win_mean', bandwidth)
            data[member]['thr_std'] = _detect_thr(data[member], 'win_std', bandwidth)
            print('Member', member, 'mean thr:', np.round(data[member]['thr_mean'],2))
            print('Member', member, 'std  thr:', np.round(data[member]['thr_std'],2))
        else:
            data[member]['is_beacon'] = True
    return data

def all_speak(data, threshold_by_mean=True, threshold_by_std=True):
    for member in data.members:
        data[member]['all_speak'] = np.zeros((data.number_of_windows,), dtype=np.int16)
        if threshold_by_mean:
            idx = data[member]['win_mean'] > data[member]['thr_mean']
            data[member]['all_speak'][idx] = 1
        if threshold_by_std:
            idx = data[member]['win_std'] > data[member]['thr_std']
            data[member]['all_speak'][idx] = 1
    return data

def real_speak(data, corr_thr=0.85, max_temp_shift=0.15):
    for member in data.members:
        data[member]['real_speak'] = np.logical_and(data[member]['all_speak'],
                                                    data[member]['gen_speak'] >= 0)
            
    sample_period = np.diff(data[data.members[0]]['time'][:2])[0]
    max_corr_lag = int(np.round(max_temp_shift/sample_period))
    num_win = len(data[data.members[0]]['real_speak'])
    for w in range(num_win-1):
        speaking = [bool(data[m]['real_speak'][w]) for m in data.members]
        num_speaking = np.sum(speaking)
        if num_speaking >= 2:
            speaking_members = np.array(data.members)[speaking]
            win_start = data[speaking_members[0]]['win_time'][w]
            win_end = data[speaking_members[0]]['win_time'][w+1]
            for i in range(num_speaking-1):
                for j in range(i+1, num_speaking):
                    data_i = data[speaking_members[i]]
                    data_j = data[speaking_members[j]]
                    idx_i = np.logical_and(data_i['time'] >= win_start,
                                           data_i['time'] < win_end)
                    idx_j = np.logical_and(data_j['time'] >= win_start,
                                           data_j['time'] < win_end)
                    corr = np.max(xcorr(data_i['signal'][idx_i], 
                                        data_j['signal'][idx_j], max_corr_lag))
                    if corr > corr_thr:
                        if data_i['win_mean'][w] > data_j['win_mean'][w]:
                            data_j['real_speak'][w] = 0
                        else:
                            data_i['real_speak'][w] = 0
    return data




























