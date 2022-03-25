import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from .vad import _positive_kde

def signals(data, title=None, fig=None, axes=None):
    if fig == None:
        fig, axes = plt.subplots(nrows=len(data), ncols=1, sharex=True, sharey=True)
    for i, member in enumerate(data.members):
        axes[i].plot([datetime.fromtimestamp(t) for t in data[member]['time']], 
                     data[member]['signal'], c=plt.get_cmap('tab10').colors[i])
        axes[i].set_title('Participant ' + str(member))
        axes[i].grid(alpha=0.3)
    fig.suptitle('Voice signals from each participant' if title==None else title)
    return fig, axes

def histograms(data, num_bins=30, plot_thresholds=True, plot_kde=True, bandwidth=None):
    non_beacons = [key for key in data.members if not data[key]['is_beacon']]
    fig, axes = plt.subplots(nrows=len(non_beacons), ncols=2, sharex=True)
    plt.xlim([0,80])
    for i, member in enumerate(non_beacons):
        m_speak = data[member]['win_mean'][data[member]['gen_speak'] > 0]
        m_silen = data[member]['win_mean'][data[member]['gen_speak'] < 0]
        bins = np.linspace(0, np.max(np.hstack((m_speak, m_silen))), num_bins+1)
        c1, bi, ba = axes[i,0].hist(m_speak, bins=bins, alpha=0.5, color='tab:blue', density=True)
        c2, bi, ba = axes[i,0].hist(m_silen, bins=bins, alpha=0.5, color='tab:orange', density=True)
        axes[i,0].set_title('Participant ' + str(member) + ' - Mean')
        s_speak = data[member]['win_std'][data[member]['gen_speak'] > 0]
        s_silen = data[member]['win_std'][data[member]['gen_speak'] < 0]
        bins = np.linspace(0, np.max(np.hstack((s_speak, s_silen))), num_bins+1)
        c3, bi, ba = axes[i,1].hist(s_speak, bins=bins, alpha=0.5, color='tab:blue', density=True)
        c4, bi, ba = axes[i,1].hist(s_silen, bins=bins, alpha=0.5, color='tab:orange', density=True)
        axes[i,1].set_title('Participant ' + str(member) + ' - Std dev')
        if plot_thresholds:
            axes[i,0].plot(np.repeat(data[member]['thr_mean'], 2), [0, np.max(np.hstack((c1, c2)))], c='k')
            axes[i,1].plot(np.repeat(data[member]['thr_std'], 2), [0, np.max(np.hstack((c3, c4)))], c='k')
        if plot_kde:
            xi, f_speak, f_silen = _positive_kde(data[member], 'win_mean', bandwidth)
            axes[i,0].plot(xi, f_speak, color='tab:blue')
            axes[i,0].plot(xi, f_silen, color='tab:orange')
            xi, f_speak, f_silen = _positive_kde(data[member], 'win_std', bandwidth)
            axes[i,1].plot(xi, f_speak, color='tab:blue')
            axes[i,1].plot(xi, f_silen, color='tab:orange')
    return fig, axes

def vad(data, gen_speak=True, all_speak=True, real_speak=True):
    if real_speak:
        fig, axes = plt.subplots(nrows=len(data)+1, ncols=1, sharex=True)
        axes[0].get_shared_y_axes().join(*axes[:-1])
        fig, axes = signals(data, 'Voice activity detection', fig, axes)
    else:
        fig, axes = signals(data, 'Voice activity detection')
    window = np.diff(data[list(data.members)[0]]['win_time'][:2])[0]
    for i, member in enumerate(data.members):
        time_vec = np.repeat(data[member]['win_time'], 2)
        time_vec[1::2] += window
        time_vec = np.array([datetime.fromtimestamp(t) for t in time_vec])
        amp = data[member]['global_mean'] + 2*data[member]['global_std']
        if gen_speak:
            gen_s = 0.8*amp*np.repeat(np.clip(data[member]['gen_speak'], 0, 1), 2)
            linestyle = ':' if all_speak and real_speak else '--' if all_speak or real_speak else '-'
            axes[i].plot(time_vec, gen_s, linestyle=linestyle, c='black')
        if all_speak:
            all_s = 1.0*amp*np.repeat(data[member]['all_speak'], 2)
            linestyle = '--' if real_speak else '-'
            axes[i].plot(time_vec, all_s, linestyle=linestyle, c='black')
        if real_speak:
            real_s = 1.2*amp*np.repeat(data[member]['real_speak'], 2)
            axes[i].plot(time_vec, real_s, c='black')
        
            time_vec = data[member]['win_time'][data[member]['real_speak'] > 0]
            time_vec = np.vstack((np.array([datetime.fromtimestamp(t) for t in time_vec]), 
                                  np.array([datetime.fromtimestamp(t) for t in time_vec+window])))
            y = (len(data.members)-1-i)*np.ones(time_vec.shape)
            axes[-1].plot(time_vec, y, c=plt.get_cmap('tab10').colors[i], linewidth=3)
    if real_speak:
        axes[-1].set_yticks(range(-1, len(data.members)+1))
        yticks = [' ']
        yticks.extend([str(member) for member in reversed(data.members)])
        yticks.append(' ')
        axes[-1].set_yticklabels(yticks)
    return fig, axes

def metrics(data):
    fig, axes = plt.subplots(2,2)
    metrics = ['speaking_time', 'overlap_time', 'overlap_count', 'turn_taking_count']
    titles = ['Vocalization distribution', 'Overlap time', 'Overlap count', 'Turn taking count']
    for i, (r,c) in enumerate(np.ndindex((2,2))):
        y = [data[member][metrics[i]] for member in data.members]
        axes[r,c].bar(range(len(data.members)), y)
        axes[r,c].set_xticks(range(len(data.members)))
        axes[r,c].set_xticklabels(data.members)
        # axes[r,c].set_title(' '.join([x.capitalize() for x in metrics[i].split('_')]))
        axes[r,c].set_title(titles[r*2+c])
        axes[r,c].grid(alpha=0.3)
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,1])
    axes[0,1].set_ylim(axes[0,0].get_ylim())
    return fig, axes





























