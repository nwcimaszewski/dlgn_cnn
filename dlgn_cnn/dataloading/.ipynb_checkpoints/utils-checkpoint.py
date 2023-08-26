import numpy as np
import pandas as pd

# Here I have functions which operate on a pandas dataframe of electrophysiological data
# these will include computing firing rate from spike times

def create_primary_key(resp, composite=['m','s','e','u']):
    # pretty hard coded but that's fine
    resp['neuron_key'] = [ '_'.join(['m'+m[-4::], 
                                 's'+f'{s:02d}', 
                                 'e'+f'{e:02d}', 
                                 'u'+f'{u:02d}']) for _, (m,s,e,u) in resp[['m','s','e','u']].iterrows() ]


# helper functions 
def fr_from_spiketimes(resp_df, time_col_key, spike_col_key, bin_len_sec=1/30,perturbance=0):
    trange = resp_df[time_col_key]
    spike_times = resp_df[spike_col_key]
    t = np.arange(trange[0],trange[1],bin_len_sec)
    # t = np.arange(trange[0]+perturbance,trange[1]+perturbance,bin_len_ms)
    spikes_post = spike_times[:,None] > t
    spikes_pre = np.hstack([spike_times[:,None] < t, np.zeros((spike_times.shape[0],1)).astype('bool')]) # add row of 1s at end
    # ^ shape: (n_spikes,150).  index ij is whether the ith spike is before/after than the jth gridpoint.
    # we want the number of spikes within each time bin, i.e. greater than each jth gridpoint and lesser than each j+1th gridpoint
    # so this should be the conjunction of the above (minus the last time index) and the similar for < (minus the first time index)
    # then we will sum this across spikes
    spikes_in = spikes_post & spikes_pre[:,1:]
    return spikes_in.sum(0) # get number of spikes in each time bin

def make_preproc():
    pass


def binarize_opto(trange_col, opto_trange_col, dt = 1/60):
    # trange_col and opto_trange_col are both pandas series
    # bin_opto_col = pd.Series({'bin_opto' : []})
    bin_opto_col = []
    for trange, optrange in zip(trange_col, opto_trange_col): # iterate through recordings, presentation of scenes
        t = np.arange(*trange,step=dt) # trange is 2 tuple (list)
        ot = np.zeros_like(t).astype('bool')

        for x in optrange: # optrange list of 2 tuples (lists)
            bin_x = (t > x[0]) & (t < x[1]) # Boolean mask for time bins in nth optogenetic flash
            ot = ot | bin_x # take disjunction (union) with previous masks

        bin_opto_col.append(ot.astype(int))
    
    return pd.Series(bin_opto_col)

# add optogen as channel to video

def add_opto_channel(tensor, c_axis, opto):
    '''
    function to broadcast optogenetic signal to the shape one channel in a tensor representation, then append this to the representation itself
    tensor: representation to be appended to, most likely of shape (b, c, t, ...)
    opto: binary optogenetic signal, of shape (t,) or maybe (t,1)
    '''
    channel_shape = tensor.shape[:c_axis,c_axis+1:] # shape of tensor along all axes except channel = size of one channel "slice"
    opto_channel = torch.ones(channel_shape) * opto # i hope this will just figure out the broadcasting nicely by itself?

    return torch.cat((tensor,opto_channel),axis=c_axis)

                     