import numpy as np

INIT_VALUE = -1.0

class MeetingData():
    ''' This class defines a data structure to store the audio meeting data. It
    is basically a dictionary where the keys are the participants numbers and
    the values are the data of each participant. It also have some 
    functionality to simplify code needed to process the data.
    
    Each function of the vad, processing (finish this)
    
    
    '''
    
    def __init__(self):
        self._data = {}
        self.__sp = INIT_VALUE
        self.__wl = INIT_VALUE
        self.__ms = INIT_VALUE
        self.__me = INIT_VALUE
        self.__nw = INIT_VALUE
        
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, item):
        self._data[key] = item
        
    def __len__(self):
        return len(self._data)
    
    def keys(self):
        return self._data.keys()
    
    @property
    def members(self):
        if self._data != {}:
            if not 'is_beacon' in self._data[list(self._data.keys())[0]].keys():
                return sorted(list(self._data.keys()))
            else:
                return sorted([m for m in self._data.keys() if (type(m) == int 
                                         and not self._data[m]['is_beacon'])])
        else:
            return []
    
    @property
    def beacons(self):
        return [m for m in self._data.keys() if self._data[m]['is_beacon']]
    
    @property
    def sample_period(self):
        if self.__sp == INIT_VALUE:
            self.__sp = np.diff(self._data[self.members[0]]['time'][:2])[0]
        return self.__sp
    
    @property
    def window_length(self):
        if self.__wl == INIT_VALUE:
            self.__wl = np.diff(self._data[self.members[0]]['win_time'][:2])[0]
        return self.__wl
    
    @property
    def number_of_windows(self):
        if self.__nw == INIT_VALUE:
            self.__nw = len(self._data[self.members[0]]['win_time'])
        return self.__nw
    
    @property
    def meeting_start(self):
        if self.__ms == INIT_VALUE:
            self.__ms = np.max([self._data[member]['time'][0] for member in self.members])
        return self.__ms
    
    @property
    def meeting_end(self):
        if self.__me == INIT_VALUE:
            self.__me = np.min([self._data[member]['time'][-1] for member in self.members])
        return self.__me
    
    @property
    def meeting_duration(self):
        return self.__me - self.__ms


















