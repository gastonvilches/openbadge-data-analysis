import numpy as np
from . import plot
import json

def num_samples_analysis(filename):
    num_samples_list = []
    with open(filename, 'r') as fid:
        for line in fid.readlines():
            packet = json.loads(line)
            num_samples_list.append(packet['data']['num_samples'])
    numbers = np.unique(num_samples_list)
    if len(numbers) == 1:
        print('All packets have', numbers[0], 'samples')
    else:
        print('Numbers of samples per packet:', numbers)

def sampling_rate_uniformity(data):
    data2 = {}
    for member_id in data.keys():
        data2[member_id] = {'signal': np.diff(data[member_id]['time']),
                            'time': data[member_id]['time'][:-1].copy()}
    fig, axes = plot.signals(data2, title='Sampling period over time')
    return fig, axes