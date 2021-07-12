import json
import numpy as np

# To do: comment and document all

def read_file(filename, excluded_members_id=[]):
    # Read text lines and save signal and timestamps to data structure
    data = {}
    with open(filename, 'r') as fid:
        for line in fid.readlines():
            packet = json.loads(line)
            member = packet['data']['member_id']
            if member in excluded_members_id:
                continue
            ts = packet['data']['timestamp']
            sp = packet['data']['sample_period']/1000
            ns = packet['data']['num_samples']
            if not member in data.keys():
                data[member] = {'signal': packet['data']['samples'],
                                   'time': list(np.linspace(ts,ts+(ns-1)*sp,ns))}
            elif not ts in data[member]['time']:
                data[member]['signal'].extend(packet['data']['samples'])
                data[member]['time'].extend(np.linspace(ts,ts+(ns-1)*sp,ns))
                
    # Convert lists to numpy arrays
    for member in data.keys():
        data[member]['time'] = np.array(data[member]['time'])
        data[member]['signal'] = np.array(data[member]['signal'], 
                                             dtype=np.int64)
    return data

def fix_time_jumps(data, max_jump_sec=1):
    # Find and fix time jumps due to clock synchronization with hub
    time_jumps_ids = []
    for member in data.keys():
        diff = np.diff(data[member]['time'])
        idx = np.argmax(diff)
        if diff[idx] > max_jump_sec:
            time_jumps_ids.append(member)
            offset = diff[idx] - diff[idx+1]
            length, = data[member]['time'].shape
            data[member]['time'] += np.pad(offset*np.ones((idx,)), 
                                              (0,length-1-idx))

    # Print info about time jump found
    if any(time_jumps_ids):
        print('Time jumps found in ' + str(time_jumps_ids))
    
    return data

def truncate(data):
    # Truncate data to the segment where all devices are turned on
    max_start_time = np.max([data[membr]['time'][0] for membr in data.keys()])
    min_end_time = np.min([data[membr]['time'][-1] for membr in data.keys()])
    for member in data.keys():
        indices = np.logical_and(data[member]['time'] >= max_start_time,
                                 data[member]['time'] <= min_end_time)
        data[member]['signal'] = data[member]['signal'][indices]
        data[member]['time'] = data[member]['time'][indices]
        
    return data
    
    # To do: Print info about truncation
    
def remove_offset(data, percentile=1):
    # Remove specific offset due to noise floor
    for member in data.keys():
        offset = int(np.percentile(data[member]['signal'], percentile))
        data[member]['signal'] -= offset
        data[member]['signal'] = np.clip(data[member]['signal'], 0, None)
        
    return data
