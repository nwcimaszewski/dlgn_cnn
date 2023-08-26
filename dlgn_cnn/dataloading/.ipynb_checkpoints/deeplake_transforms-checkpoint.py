# file defining transforms for dataloaders which extend base transform classes of neuralpredictors.
# these are defined here mainly to maintain compatibility with neuralpredictors and not require a custom branch

import numpy as np

from neuralpredictors.data.transforms import MovieTransform, StaticTransform, Invertible
from collections import namedtuple


class Debug():
    def __init__(self,tensorname):
        self.tensorname = tensorname

    def __call__(self, x):
        print(f'Debugging...what does each transform do with its tensor?  how does it merge them after?  Well this batch from{self.tensorname} is of type:') 
        print(type(x))
        return(x)

class Tupelize():
    def __init__(self, tensor_name): # create named tuple from array using name of tensor array is taken from
        self.tupelize = namedtuple(f'{tensor_name}Tuple', tensor_name)
        
    def __call__(self, x):
        return self.tupelize(x)
        
class Detupelize():
    def __init__(self, field_name): # return data array from named tuple
        self.field_name = field_name
        
    def __call__(self, x):
        return getattr(x, self.field_name)

class SubsequenceByOpto(MovieTransform):
    def __init__(self, 
                 deeplake_dataset,
                 frames, 
                 channel_first=("inputs",), 
                 offset=-1
                 ):
        """
        Given a sequential (movie like) data, subselect a consequent `frames` counts of frames, starting with
        `offset` frames skipped. If `offset`< 0, then the subsequence is taken with a random (but valid) offset each iteration.

        Args:
            frames (int): Length of subsequence to be selected from each sample
            channel_first (tuple, optional): A list of data key names where the channel (and thus not time) dimension occurs on the first dimension (dim=0). Otherwise, it's assumed
            that the time dimesion occurs on the first dimension. Defaults to ("inputs",).
            offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        """
        self.frames = frames
        # for special group, the time slicing is applied on
        self.channel_first = channel_first
        self.offset = offset
        self.ds = deeplake_dataset

    def __call__(self, x):
        first_group = x._fields[0]

        # get the length of the full sequence. Note that which axis to pick depends on if it's
        # channel first (i.e. time second) group
        t = getattr(x, first_group).shape[int(first_group in self.channel_first)]

        if self.offset < 0:
            i = np.random.randint(0, t - self.frames)
        else:
            i = self.offset
        return x.__class__(
            **{
                k: getattr(x, k)[:, i : i + self.frames, ...]
                if k in self.channel_first
                else getattr(x, k)[i : i + self.frames, ...]
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
        return self.__class__.__name__ + "({})".format(self.frames)


class NormalizeVideo(MovieTransform, StaticTransform, Invertible):
    """
    Added by Nick:
    Transform which normalizes only the videos.
    For the deeplake dataloader.transforms().batch().pytorch(tensors= [] ) syntax, the ability to pass a dictionary of different transforms to .transforms() makes it easier to do this
    The other alternative would be to find a way to cast the Samples returned by dataloader to the NamedTuple format necessary for neuralpredictors transforms to work flawlessly
    Barring that, I would have to create customs versions of each one eliminating references to x._fields.
    I thought there must be a simple way to define this attribute as simply the output of x.keys() but I can't figure it out rn
    """

    def __init__(
        self,
        deeplake_dataset, # reads statistics of deeplake dataset object
        inputs_mean=None,
        inputs_std=None,
        in_name=None
    ):

        if in_name is None:
            in_name = "videos" if "videos" in deeplake_dataset.info.statistics.keys() else "inputs"

        self._inputs_mean = deeplake_dataset.info.statistics[in_name]["mean"][()] if inputs_mean is None else inputs_mean
        self._inputs_std = deeplake_dataset.info.statistics[in_name]["std"][()] if inputs_mean is None else inputs_std

        # -- inputs
        transform = lambda x: (x - self._inputs_mean) / self._inputs_std
        itransform = lambda x: x * self._inputs_std + self._inputs_mean    

        self._transform = transform
        self._itransform = itransform

    def __call__(self, x):
        """
        Apply transformation
        This takes in a dict object ( has .keys() method ) and returns the same
        This is causing errors in the DeepLakeDataLoader.transform({k1:,k2:,...}) because I think there each tensor is passed in in a numpy format, which doesn't have keys.
        If the data were passed as a single-dict key it would be no problem
        """
        return self._transform(x)

    def inv(self, x):
        return self.itransform(x)


class StandardizeResponse(MovieTransform, StaticTransform, Invertible):
    """
    Added by Nick:
    Transform which normalizes only the responses.
    For the deeplake dataloader.transforms().batch().pytorch(tensors= [] ) syntax, the ability to pass a dictionary of different transforms to .transforms() makes it easier to do this
    The other alternative would be to find a way to cast the Samples returned by dataloader to the NamedTuple format necessary for neuralpredictors transforms to work flawlessly
    Barring that, I would have to create customs versions of each one eliminating references to x._fields.
    I thought there must be a simple way to define this attribute as simply the output of x.keys() but I can't figure it out rn
    """

    def __init__(
        self,
        deeplake_dataset,
        outputs_mean=None,
        outputs_std=None,
        out_name='responses',
        pct_threshold = 0.01 # below this percentile of the mean of std values, neuron responses are scaled, as this might cause unrealistic 
    ):

        s = np.array(deeplake_dataset.info.statistics[out_name]["std"])

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
        """
        Apply transformation
        This takes in a dict object ( has .keys() method ) and returns the same
        This is causing errors in the DeepLakeDataLoader.transform({k1:,k2:,...}) because I think there each tensor is passed in in a numpy format, which doesn't have keys.
        If the data were passed as a single-dict key it would be no problem
        """
        return self._transform(x)

    def inv(self, x):
        return self.itransform(x)
    
