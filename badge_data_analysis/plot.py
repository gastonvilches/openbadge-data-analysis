import matplotlib.pyplot as plt
from datetime import datetime

def signals(data, figsize=None, title='Voice signals from each participant'):
    if figsize == None:
        if len(data) > 6:
            _figsize = (20, 12)
        else:
            _figsize = (20, 2*len(data))
    fig, axes = plt.subplots(nrows=len(data), ncols=1, sharex=True, 
                             sharey=True, figsize=_figsize)
    for i, member in enumerate(data.keys()):
        axes[i].plot([datetime.fromtimestamp(t) for t in data[member]['time']], 
                       data[member]['signal'])
        axes[i].set_title('Participant ' + str(member))
        axes[i].grid(alpha=0.3)
    fig.suptitle(title)
    plt.show()
    
    return fig, axes
        







