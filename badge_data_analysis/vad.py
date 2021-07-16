import numpy as np
from scipy.stats import gaussian_kde

# To do: Comment and document all
#
#        Reduce code length

def xcorr(x, y, max_lag=None, normalize=True):
    y_len = len(y)
    y_energy = np.sum(y**2)
    if max_lag == None:
        max_lag = y_len - 1 
    x = np.pad(x, (max_lag, max_lag + y_len - len(x)))
    corr = np.zeros((2*max_lag+1,))
    for lag in range(2*max_lag+1):
        corr[lag] = np.sum(x[lag:lag+y_len]*y)
        if normalize:
            corr[lag] /= np.sqrt(np.sum(x[lag:lag+y_len]**2)*y_energy)
    return corr

def genuine_speak(data, window=1.0, max_temp_shift=0.15, corr_thr=0.86, 
                  silence_thr_mean=0.5, silence_thr_std=0.0): #min_num_samples=0.8
    
    # From now on the sample period it is assumed constant
    sample_period = np.diff(data[list(data.keys())[0]]['time'][:2])[0]
    max_corr_lag = int(np.round(max_temp_shift/sample_period))
    # min_num_samples = int(np.round(min_num_samples*window/sample_period))
    
    # Calculate the number of windows
    start_time = np.max([data[membr]['time'][0] for membr in data.keys()])
    end_time = np.min([data[membr]['time'][-1] for membr in data.keys()])
    num_win = int(np.ceil((end_time-start_time)/window))
    
    # Memory alloc
    for member in data.keys():
        data[member]['gen_speak'] = np.zeros((num_win,), dtype=np.int16)
        data[member]['all_speak'] = np.zeros((num_win,), dtype=np.int16)
        data[member]['real_speak'] = np.zeros((num_win,), dtype=np.int16)
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
        for member in data.keys():
            idx = np.logical_and(data[member]['time'] >= win_start,
                                   data[member]['time'] < win_end)
            data[member]['win_mean'][w_i] = np.mean(data[member]['signal'][idx])
            data[member]['win_std'][w_i] = np.std(data[member]['signal'][idx])
        
        means = np.array([data[mem]['win_mean'][w_i] for mem in data.keys()])
        max_vol = np.max(means)
        max_vol_mem = list(data.keys())[np.where(means == max_vol)[0][0]]
        
        # This is to avoid false detections when there is silence
        if max_vol < (silence_thr_mean*data[max_vol_mem]['global_mean'] 
                      + silence_thr_std*data[max_vol_mem]['global_std']):
            continue
        
        w_idx_max = np.logical_and(data[max_vol_mem]['time'] >= win_start,
                                   data[max_vol_mem]['time'] < win_end)
        for member in data.keys():
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
            for member in data.keys():
                if member == max_vol_mem:
                    data[member]['gen_speak'][w_i] = 1
                else:
                    data[member]['gen_speak'][w_i] = -1
    return data

def _positive_kde(member_data, key, xi_max=40, step=0.1):
    xi = np.arange(0, xi_max, step)
    kde_speak = gaussian_kde(member_data[key][member_data['gen_speak'] > 0])
    kde_silen = gaussian_kde(member_data[key][member_data['gen_speak'] < 0])
    f_speak = kde_speak(xi) + kde_speak(-xi)
    f_silen = kde_silen(xi) + kde_silen(-xi)
    return xi, f_speak, f_silen

def _detect_thr(member_data, key):
    xi, f_speak, f_silen = _positive_kde(member_data, key)
    
    intersection = 0
    for idx in range(len(xi)):
        if f_speak[idx] > f_silen[idx]:
            intersection = xi[idx]
            break
    if intersection == 0:
        print('Intersection between silent and speaking histograms not found')
        
    return intersection

def calculate_thresholds(data):
    for member in data.keys():
        if np.any(data[member]['gen_speak'] > 0):
            data[member]['thr_mean'] = _detect_thr(data[member], 'win_mean')
            data[member]['thr_std'] = _detect_thr(data[member], 'win_std')
        else:
            data[member]['is_beacon'] = True
    return data

def all_speak(data, threshold_by_mean=True, threshold_by_std=True):
    for member in data.keys():
        if not data[member]['is_beacon']:
            if threshold_by_mean:
                idx = data[member]['win_mean'] > data[member]['thr_mean']
                data[member]['all_speak'][idx] = 1
            if threshold_by_std:
                idx = data[member]['win_std'] > data[member]['thr_std']
                data[member]['all_speak'][idx] = 1
    return data

def real_speak(data, corr_thr=0.88, max_temp_shift=0.15):
    for member in data.keys():
        if not data[member]['is_beacon']:
            data[member]['real_speak'] = np.logical_and(data[member]['all_speak'],
                                                        data[member]['gen_speak'] >= 0)
            
    sample_period = np.diff(data[list(data.keys())[0]]['time'][:2])[0]
    max_corr_lag = int(np.round(max_temp_shift/sample_period))
    num_win = len(data[list(data.keys())[0]]['real_speak'])
    for w in range(num_win-1):
        speaking = [bool(data[m]['real_speak'][w]) for m in data.keys()]
        num_speaking = np.sum(speaking)
        if num_speaking >= 2:
            speaking_members = np.array(list(data.keys()))[speaking]
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




























