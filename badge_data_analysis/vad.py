import numpy as np
from scipy.stats import gaussian_kde

# To do: Finish all_speak and real_speak
#
#        To analyze the effect of low correlation due to large distance 
#        between participants
#
#        Comment and document all
#


# vol_rule = [1 1]               # (1 & mean_thr) || (2 & str_thr)
# bandwidth = 2.5
# plot_histograms = 1

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

def genuine_speak(data, window=1.0, max_temp_shift=0.15, min_num_samples=0.8,
                  corr_thr=0.85, silence_thr_mean=1.0, silence_thr_std=0.0):
    
    # From now on the sample period it is assumed constant
    sample_period = np.diff(data[list(data.keys())[0]]['time'][:2])[0]
    max_corr_lag = int(np.round(max_temp_shift/sample_period))
    min_num_samples = int(np.round(min_num_samples*window/sample_period))
    
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
        data[member]['thr_mean'] = None
        data[member]['thr_std'] = None
        data[member]['is_beacon'] = False
    
    # Genuine Speak
    for w_i in range(num_win):
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







