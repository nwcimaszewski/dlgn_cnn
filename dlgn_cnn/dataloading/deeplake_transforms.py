# file defining transforms for dataloaders which extend base transform classes of neuralpredictors.
# these are defined here mainly to maintain compatibility with neuralpredictors and not require a custom branch

import numpy as np

from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from neuralpredictors.data.transforms import MovieTransform, StaticTransform, Invertible
from collections import namedtuple, Iterable

import ipdb

class Debug():
    def __init__(self):
        pass

    def __call__(self, x):
        print(type(x))
        print(len(x))
        print(type(x[0])) 
        print(len(x[0]))
        return(x)


class Tupelize(StaticTransform):
    def __init__(self, dict_keys, tuple_type_name=None): # takes in dictionary with keys dict_keys, returns named tuple with fields dict_keys
        tuple_type_name = 'dataTuple' if tuple_type_name is None else tuple_type_name
        # ipdb.set_trace()
        self.tupelize = namedtuple(tuple_type_name, dict_keys) # dict_keys is name of data tensors, keys of dict as returned by pytorch loader
        
    def __call__(self, x):
        # NOTE: x is of type IterableOrderedDict.  If fed a normal dict, * notation only returns list of key names and this will not do what we want
        # x = {field: x[field] for field in self.tupelize._fields} # any field not stated in construction of transform ignored (e.g. 'index')
        # print(type(x), self.tupelize._fields, x.keys())]
        # y = {}
        # for k in self.tupelize._fields:
        #     print(k)
        #     print(type(x[k]))
        #     y[k] = x[k]

        # ipdb.set_trace()
        return self.tupelize(**{k: x[k] for k in self.tupelize._fields} )
        
class Detupelize(StaticTransform):
    def __init__(self): # return data array from named tuple
        pass
        
    def __call__(self, x):
        kv = [(field, getattr(x,field)) for field in x._fields] # this way will create an order, based on how the fields are ordered
        return IterableOrderedDict(kv)
    

class SubsequenceByOpto(MovieTransform):
    def __init__(self, 
                 deeplake_dataset,
                 frames, 
                 channel_first=('videos','responses'), 
                 exclude = ['index'],
                 opto_key = 'opto',
                 offset=0,
                 force = None
                 ):
        '''
        Given a sequential (movie like) data, subselect a consequent `frames` counts of frames, starting with
        `offset` frames skipped. If `offset`< 0, then the subsequence is taken with a random (but valid) offset each iteration.

        Args:
            frames (int): Length of subsequence to be selected from each sample
            channel_first (tuple, optional): A list of data key names where the channel (and thus not time) dimension occurs on the first dimension (dim=0). Otherwise, it's assumed
            that the time dimesion occurs on the first dimension. Defaults to ('inputs',).
            offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        '''
        self.frames = frames
        # for special group, the time slicing is applied on
        self.channel_first = channel_first
        self.offset = offset
        self.ds = deeplake_dataset
        self.opto_key = opto_key
        self.exclude = exclude
        self.force = force # values

    def __call__(self, x):
        # get the length of the full sequence. Note that which axis to pick depends on if it's
        # channel first (i.e. time second) group
        t = getattr(x, self.opto_key).shape[-1] # time axis is last, IF NOT 

        opto_signal = getattr(x,self.opto_key) # (t,), scalar sequence, as this transform is applied to individual items in a batch

        switches = np.concatenate(([-1], np.argwhere(np.diff(opto_signal)!=0).flatten(),[t-1]))
        # compute possible startpoints
        viable_starts = []
        for i, j in zip(switches, switches[1:]): # inspect the consecutive pairs
            if j-i >= self.frames: # if there are at least 60 time bins between i and j
        #  j+1-frames is the value of the last possible startpoint of a constant sequence, if j should be the last included element, given end indices are not included
                viable_starts = np.append(viable_starts, np.arange(i + 1 , j + 1 - self.frames + 1)) # we can start at earliest the bin after i, and at latest j+1-frames (we want this to be included, so endpoint of arange is j+1-frames+1

        if self.force is not None: # if we only want 1s or 0s
            # ipdb.set_trace()
            # print(getattr(x,self.opto_key).shape)
            viable_starts = [i for i in viable_starts if getattr(x,self.opto_key)[int(i)] == self.force] # is only (300,) for each sample x  

        i = int(np.random.choice(viable_starts)) # choose random start, such that optogenetic signal will be constant for entire 60 frames
        # ipdb.set_trace()

        return x.__class__(
            **{
                k: getattr(x, k) if k in self.exclude else
                getattr(x, k)[:, i : i + self.frames, ...] # (b,c,t,...) called on individual items of batch, so each item is (c,t,...)
                if k in self.channel_first
                else getattr(x, k)[i : i + self.frames, ...] # (b,t,c,...)
                for k in x._fields
            }
        )

    def id_transform(self, id_map):
        # until a better solution is reached, skipping this
        return id_map

        new_map = {}
        first_group = list(id_map.keys())[0]
        v_fg = id_map[first_group]
        t = v_fg.shape[int(first_group in self.channel_first)]
        i = np.random.randint(0, t - self.frames)
        for k, v in id_map.items():
            if k in self.channel_first:
                new_map[k] = v[:, i : i + self.frames]
            else:
                new_map[k] = v[i : i + self.frames]

        return new_map

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.frames)

class SelectVideoChannel(MovieTransform):
    """
    Given a StaticImage object that includes "images", it will select a particular input channel.
    """

    def __init__(self, grab_channel, in_name = 'videos', channel_axis=1):
        self.grab_channel = grab_channel if isinstance(grab_channel, Iterable) else [grab_channel]
        self.in_name = in_name
        self.channel_axis = channel_axis

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        input = key_vals[self.in_name]
        # if batched, channel_axis
        key_vals[self.in_name] = input[:, (self.grab_channel,)] if len(input.shape) == 5 else input[self.grab_channel, ...]
        
        return x.__class__(**key_vals)

class SubsequenceWithoutOpto(MovieTransform):
    def __init__(self, 
                 deeplake_dataset,
                 frames, 
                 channel_first=('videos','responses'), 
                 exclude = ['index'],
                 time_key = 'responses', # key to extract time from 
                 offset=0
                 ):
        '''
        Given a sequential (movie like) data, subselect a consequent `frames` counts of frames, starting with
        `offset` frames skipped. If `offset`< 0, then the subsequence is taken with a random (but valid) offset each iteration.

        Args:
            frames (int): Length of subsequence to be selected from each sample
            channel_first (tuple, optional): A list of data key names where the channel (and thus not time) dimension occurs on the first dimension (dim=0). Otherwise, it's assumed
            that the time dimesion occurs on the first dimension. Defaults to ('inputs',).
            offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        '''
        
        self.frames = frames
        # for special group, the time slicing is applied on
        self.channel_first = channel_first
        self.offset = offset
        self.ds = deeplake_dataset
        # self.opto_key = opto_key
        self.time_key = time_key
        self.exclude = exclude

    def __call__(self, x):
        # get the length of the full sequence. Note that which axis to pick depends on if it's
        # channel first (i.e. time second) group
        t = getattr(x, self.time_key).shape[-1] # time axis is last, IF NOT 

        i = int(np.random.choice(t - self.frames)) # choose random start, such that optogenetic signal will be constant for entire 60 frames

        return x.__class__(
            **{
                k: getattr(x, k) if k in self.exclude else
                getattr(x, k)[:, i : i + self.frames, ...] # (b,c,t,...) called on individual items of batch, so each item is (c,t,...)
                if k in self.channel_first
                else getattr(x, k)[i : i + self.frames, ...] # (b,t,c,...)
                for k in x._fields
            }
        )

    def id_transform(self, id_map):
        # until a better solution is reached, skipping this
        return id_map

        new_map = {}
        first_group = list(id_map.keys())[0]
        v_fg = id_map[first_group]
        t = v_fg.shape[int(first_group in self.channel_first)]
        i = np.random.randint(0, t - self.frames)
        for k, v in id_map.items():
            if k in self.channel_first:
                new_map[k] = v[:, i : i + self.frames]
            else:
                new_map[k] = v[i : i + self.frames]

        return new_map

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.frames)

class NeuroNormalizeDeeplake(MovieTransform, StaticTransform, Invertible):
    '''
    Note that this normalizer only works with datasets that provide specific attributes information
    of very specific formulation

    Normalizes a trial with fields: inputs, behavior, eye_position, and responses. The pair of
    behavior and eye_position can be missing. The following normalizations are applied:

    - inputs are scaled by the training std of the stats_source and centered on the mean of the movie
    - behavior is divided by the std if the std is greater than 1% of the mean std (to avoid division by 0)
    - eye_position is z-scored
    - reponses are divided by the per neuron std if the std is greater than
            1% of the mean std (to avoid division by 0)
    '''

    def __init__(
        self,
        dl_ds, # deeplake dataset
        exclude=None,
        inputs_mean=None,
        inputs_std=None,
        subtract_behavior_mean=False,
        in_name=None,
        out_name=None,
        eye_name=None,
        resp_std_temporal=False, # if True, then std extracted timewise is used in standardization, otherwise time-pooled
        vid_stats_spatial=False, # if True, then statistics extracted pixelwise are used in normalization, otherwise pixel-pooled,
        dtype = 'float32' #  a temporary hack for having computed statistics in float64
        # vid_channels = None # for flexibility, could allow to sub-select indices from stored statistics...alternatively, never call this after subselecting channels of video batch, only before
    ):

        self.exclude = exclude or []

        if in_name is None:
            in_name = 'videos' if 'videos' in  dl_ds.tensors.keys() else 'inputs'
        if out_name is None:
            out_name = 'responses' if 'responses' in dl_ds.tensors.keys() else 'targets'
        if eye_name is None:
            eye_name = 'pupil_center' if 'pupil_center' in  dl_ds.tensors.keys() else 'eye_position'

        if inputs_mean:
            self._inputs_mean = inputs_mean
        elif vid_stats_spatial:
            self._inputs_mean = dl_ds.info.statistics[in_name]['mean_2D'].astype(dtype) #[:,:,None,None]
        else:
            self._inputs_mean = dl_ds.info.statistics[in_name]['channel_mean'][:,None,None,None].astype(dtype) # want 4D (c, t, x, y)
        # ipdb.set_trace()

        if inputs_std:
            self._inputs_std = inputs_std
        elif vid_stats_spatial:
            self._inputs_std = dl_ds.info.statistics[in_name]['std_2D'].astype(dtype) #[:,:,None,None]
        else:
            self._inputs_std = dl_ds.info.statistics[in_name]['channel_std'][:,None,None,None].astype(dtype)
            

        if resp_std_temporal:
            s = np.array(dl_ds.info.statistics[out_name]['std_2D']).astype(dtype) # (c,t)
        else:
            s = np.array(dl_ds.info.statistics[out_name]['channel_std'])[:,None].astype(dtype) #(c,1), prevents trivial broadcasting error

        # TODO: consider other baselines
        threshold = 0.01 * np.nanmean(s)
        idx = s > threshold
        self._response_precision = np.ones_like(s) / threshold
        self._response_precision[idx] = 1 / s[idx]
        transforms, itransforms = {}, {}

        # -- inputs
        transforms[in_name] = lambda x: (x - self._inputs_mean) / self._inputs_std
        itransforms[in_name] = lambda x: x * self._inputs_std + self._inputs_mean

        # -- responses
        transforms[out_name] = lambda x: x * self._response_precision 
        itransforms[out_name] = lambda x: x / self._response_precision

        # -- optogenetic signal
        # transforms['opto'] = None
        # itransforms['opto'] = None

        # -- behavior
        transforms['behavior'] = lambda x: x

        # -- trial_idx
        trial_idx_mean = np.arange(len(dl_ds)).mean()
        trial_idx_std = np.arange(len(dl_ds)).std()
        transforms['trial_idx'] = lambda x: (x - trial_idx_mean) / trial_idx_std
        itransforms['trial_idx'] = lambda x: x * trial_idx_std + trial_idx_mean

        if eye_name in dl_ds.tensors.keys():
            self._eye_mean = np.array(dl_ds.info.statistics[eye_name]['mean'])
            self._eye_std = np.array(dl_ds.info.statistics[eye_name]['std'])
            transforms[eye_name] = lambda x: (x - self._eye_mean) / self._eye_std
            itransforms[eye_name] = lambda x: x * self._eye_std + self._eye_mean

        if 'behavior' in dl_ds.tensors.keys():
            s = np.array(dl_ds.info.statistics['behavior']['std'])

            self.behavior_mean = (
                0 if not subtract_behavior_mean else np.array(dl_ds.info.statistics['behavior']['mean'])
            )
            self._behavior_precision = 1 / s
            # -- behavior
            transforms['behavior'] = lambda x: (x - self.behavior_mean) * self._behavior_precision
            itransforms['behavior'] = lambda x: x / self._behavior_precision + self.behavior_mean

        self._transforms = transforms
        self._itransforms = itransforms

    def __call__(self, x):
        '''
        Apply transformation
        '''
        # ipdb.set_trace()

        return x.__class__(
            **{k: (self._transforms[k](v) if k not in self.exclude else v) for k, v in zip(x._fields, x)}
        )

    def inv(self, x):
        return x.__class__(
            **{k: (self._itransforms[k](v) if k not in self.exclude else v) for k, v in zip(x._fields, x)}
        )

    def __repr__(self):
        return super().__repr__() + ('(not {})'.format(', '.join(self.exclude)) if self.exclude is not None else '')


class NormalizeVideo(MovieTransform, StaticTransform, Invertible):
    '''
    Added by Nick:
    Transform which normalizes only the videos.
    For the deeplake dataloader.transforms().batch().pytorch(tensors= [] ) syntax, the ability to pass a dictionary of different transforms to .transforms() makes it easier to do this
    The other alternative would be to find a way to cast the Samples returned by dataloader to the NamedTuple format necessary for neuralpredictors transforms to work flawlessly
    Barring that, I would have to create customs versions of each one eliminating references to x._fields.
    I thought there must be a simple way to define this attribute as simply the output of x.keys() but I can't figure it out rn
    '''

    def __init__(
        self,
        deeplake_dataset, # reads statistics of deeplake dataset object
        inputs_mean=None,
        inputs_std=None,
        in_name=None
    ):

        if in_name is None:
            in_name = 'videos' if 'videos' in deeplake_dataset.info.statistics.keys() else 'inputs'

        self._inputs_mean = deeplake_dataset.info.statistics[in_name]['mean'][()] if inputs_mean is None else inputs_mean
        self._inputs_std = deeplake_dataset.info.statistics[in_name]['std'][()] if inputs_mean is None else inputs_std

        # -- inputs
        transform = lambda x: (x - self._inputs_mean) / self._inputs_std
        itransform = lambda x: x * self._inputs_std + self._inputs_mean    

        self._transform = transform
        self._itransform = itransform

    def __call__(self, x):
        '''
        Apply transformation
        This takes in a dict object ( has .keys() method ) and returns the same
        This is causing errors in the DeepLakeDataLoader.transform({k1:,k2:,...}) because I think there each tensor is passed in in a numpy format, which doesn't have keys.
        If the data were passed as a single-dict key it would be no problem
        '''
        return self._transform(x)

    def inv(self, x):
        return self.itransform(x)


class StandardizeResponse(MovieTransform, StaticTransform, Invertible):
    '''
    Added by Nick:
    Transform which normalizes only the responses.
    For the deeplake dataloader.transforms().batch().pytorch(tensors= [] ) syntax, the ability to pass a dictionary of different transforms to .transforms() makes it easier to do this
    The other alternative would be to find a way to cast the Samples returned by dataloader to the NamedTuple format necessary for neuralpredictors transforms to work flawlessly
    Barring that, I would have to create customs versions of each one eliminating references to x._fields.
    I thought there must be a simple way to define this attribute as simply the output of x.keys() but I can't figure it out rn
    '''

    def __init__(
        self,
        deeplake_dataset,
        outputs_mean=None,
        outputs_std=None,
        out_name='responses',
        pct_threshold = 0.01 # below this percentile of the mean of std values, neuron responses are scaled, as this might cause unrealistic 
    ):

        s = np.array(deeplake_dataset.info.statistics[out_name]['std'])

        # TODO: consider other baselines
        threshold = pct_threshold * np.nanmean(s) #
        idx = s > threshold
        self._response_precision = np.ones_like(s) / threshold
        self._response_precision[idx] = 1 / s[idx]

        
        # -- inputs
        transform = lambda x: x * self._response_precision
        itransform = lambda x: x / self._response_precisionn    

        self._transform = transform
        self._itransform = itransform

    def __call__(self, x):
        '''
        Apply transformation
        This takes in a dict object ( has .keys() method ) and returns the same
        This is causing errors in the DeepLakeDataLoader.transform({k1:,k2:,...}) because I think there each tensor is passed in in a numpy format, which doesn't have keys.
        If the data were passed as a single-dict key it would be no problem
        '''
        return self._transform(x)

    def inv(self, x):
        return self.itransform(x)
    
