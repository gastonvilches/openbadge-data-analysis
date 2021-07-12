import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from .vad import _positive_kde

# To do: comment and document all

def signals(data, title='Voice signals from each participant'):
    fig, axes = plt.subplots(nrows=len(data), ncols=1, sharex=True, 
                             sharey=True)
    for i, member in enumerate(data.keys()):
        axes[i].plot([datetime.fromtimestamp(t) for t in data[member]['time']], 
                       data[member]['signal'])
        axes[i].set_title('Participant ' + str(member))
        axes[i].grid(alpha=0.3)
    fig.suptitle(title)
    return fig, axes
        
def vad(data, gen_speak=False, all_speak=False, real_speak=True):
    # To do: plot optionally gen, all and real
    #        plot legend
    fig, axes = signals(data, title='Voice activity detection')
    for i, member in enumerate(data.keys()):
        time_vec = np.repeat(data[member]['win_time'], 2)
        time_vec[1::2] += np.diff(data[member]['win_time'][:2])[0]
        signal = ((data[member]['global_mean'] + 2*data[member]['global_std'])
                  *np.repeat(np.clip(data[member]['gen_speak'], 0, None), 2))
        axes[i].plot([datetime.fromtimestamp(t) for t in time_vec], signal)
    return fig, axes

def histograms(data, num_bins=30, plot_thresholds=True, plot_kde=True):
    # To do: plot legend
    non_beacons = [key for key in data.keys() if not data[key]['is_beacon']]
    fig, axes = plt.subplots(nrows=len(non_beacons), ncols=2)
    for i, member in enumerate(non_beacons):
        m_speak = data[member]['win_mean'][data[member]['gen_speak'] > 0]
        m_silen = data[member]['win_mean'][data[member]['gen_speak'] < 0]
        bins = np.linspace(0, np.max(np.hstack((m_speak, m_silen))), num_bins+1)
        c1, bi, ba = axes[i,0].hist(m_speak, bins=bins, alpha=0.5, color='tab:blue', density=True)
        c2, bi, ba = axes[i,0].hist(m_silen, bins=bins, alpha=0.5, color='tab:orange', density=True)
        axes[i,0].set_title('Participant ' + str(member) + ' - means histogram')
        s_speak = data[member]['win_std'][data[member]['gen_speak'] > 0]
        s_silen = data[member]['win_std'][data[member]['gen_speak'] < 0]
        bins = np.linspace(0, np.max(np.hstack((s_speak, s_silen))), num_bins+1)
        c3, bi, ba = axes[i,1].hist(s_speak, bins=bins, alpha=0.5, color='tab:blue', density=True)
        c4, bi, ba = axes[i,1].hist(s_silen, bins=bins, alpha=0.5, color='tab:orange', density=True)
        axes[i,1].set_title('Participant ' + str(member) + ' - stds histogram')
        if plot_thresholds:
            axes[i,0].plot(np.repeat(data[member]['thr_mean'], 2), 
                           [0, np.max(np.hstack((c1, c2)))], c='k')
            axes[i,1].plot(np.repeat(data[member]['thr_std'], 2), 
                           [0, np.max(np.hstack((c3, c4)))], c='k')
        if plot_kde:
            xi, f_speak, f_silen = _positive_kde(data[member], 'win_mean')
            axes[i,0].plot(xi, f_speak, color='tab:blue')
            axes[i,0].plot(xi, f_silen, color='tab:orange')
            xi, f_speak, f_silen = _positive_kde(data[member], 'win_std')
            axes[i,1].plot(xi, f_speak, color='tab:blue')
            axes[i,1].plot(xi, f_silen, color='tab:orange')
    return fig, axes

