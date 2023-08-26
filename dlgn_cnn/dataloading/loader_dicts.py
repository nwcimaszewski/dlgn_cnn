# Rewriting some of the dataloaders functions from sensorium 2023
# Added functionality of to load deeplake dataset objects from local storage or Deeplake API

import numpy as np
from neuralpredictors.data.datasets import MovieFileTreeDataset, TransformDataset
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import (AddBehaviorAsChannels,
                                              AddPupilCenterAsChannels,
                                              ChangeChannelsOrder, CutVideos,
                                              ExpandChannels, NeuroNormalizer,
                                              ScaleInputs, SelectInputChannel,
                                              Subsample, Subsequence, ToTensor)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
# from torch.utils.data import Transform
import deeplake

from torchvision.transforms import Compose

from .deeplake_transforms import *

def deeplake_loader_dict(
    paths,
    batch_size,
    preproc,
    train_val_split = [.80,.20],
    use_api = False,
    normalize = True,
    exclude: str = None,
    cuda: bool = False,
    max_frame=None,
    frames=30, # default 30 frames, 1 sec at 30 Hz
    offset=-1, # causes random offset from beginning of each sample
    inputs_mean=None,
    inputs_std=None,
    include_behavior=True,
    include_pupil_centers=True,
    include_optogenetics=False,
    include_pupil_centers_as_channels=False,
    scale=1,
    to_cut=True,
    exclude_beh_channels=None,
    channels_last=False,
    split_suffix=False, # whether 
    user_token=None, # deeplake[enterprise] credentials, necessary for using optimized dataloaders
    org_id=None
):
    """
    Added by Nicholas: function to return video loaders directly from (list of paths to local) deeplake dataset(s)
    
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        paths (list): list of paths for the deeplake datasets
        batch_size (int): batch size.
        frames (int, optional): how many frames ot take per video
        max_frame (int, optional): which is the maximal frame that could be taken per video
        offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        include_pupil_centers (bool, optional): whether to include pupil center data
        include_pupil_centers_as_channels(bool, optional): whether to include pupil center data as channels
        scale (float, optional): scalar factor for the image resolution.
            scale = 1: full iamge resolution (144 x 256)
            scale = 0.25: resolution used for model training (36 x 64)
        float64 (bool, optional):  whether to use float64 in MovieFileTreeDataset
        split_suffix (bool, optional): whether local path for deeplake dataset has "_train" or "_val" appended or not 
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    # initialize data_keys
    data_keys = [
        "videos",
        "responses"
    ]
    if include_behavior:
        data_keys.append("behavior")
    if include_pupil_centers:
        data_keys.append("pupil_center")
    if include_optogenetics:
        data_keys.append("opto")
    # our deeplake dataset must have tensors for each data_key

    # hard coding this for now
    # naming is just for consistency with Sensorium 2023
    loaders_dict = {"oracle": {}, "train": {}}

    path_suffixes  = ['train','val'] # possible suffixes to local deeplake paths - created to accomodate downloaded V1 scans
    loader_keys = loaders_dict.keys() # keys for loader dict used by training functions
    
    # V1 datasets provided for sensorium competition are already separated by train/val/test
    for path in paths: # for each provided dataset 
        if use_api is True:
            ds = deeplake.load(f'{path}',token=user_token, org_id=org_id)
        else:
            ds = deeplake.load(f'{path}',token=user_token, org_id=org_id)
        
        train_ds, val_ds = ds.random_split(train_val_split) # HARD CODING 80/20 split
        # TODO: generalize this to more keys than just 'train' and 'val
        # n = len(loaders_dict)
        loaders_dict['train'][ path ] = train_ds.dataloader().transform(preproc(train_ds)).batch(batch_size).pytorch() 
        loaders_dict['oracle'][ path ] = val_ds.dataloader().transform(preproc(val_ds)).batch(batch_size).pytorch() 
            
        # TODO: generalize function to sinzlab formatted V1 scans
        # difference is that pathnames of deeplake datasets have "_train", "_val", "_test" as suffixes
        
        return loaders_dict
        
        # if split_suffix: # if suffix is there...
        #     ds = deeplake.load(f'{path}_{suffix}',token=user_token, org_id=org_id)
        # else:
        #     ds = deeplake.load(f'{path}',token=user_token, org_id=org_id)
        #     train_ds, val_ds = ds.random_split([0.8, 0.2])
            
        # for i, suffix in enumerate(path_suffixes):
        #     if use_api is True: # download dataset directly from the Deeplake API
                # ds = deeplake.load(f'hub://sinzlab/sensorium2023_{path}_{suffix}',read_only=True) # maybe fix this so paths can be passed in in a more forgiving format
                # this bypasses the standard FileTreeHierarchy/TransformDataset classes from neural.predictors.data.datasets
                # dat2 = TransformDatasetFromDeeplake(ds,*data_keys)
                # instead local Deeplake datasets can be created using deeplake_tutorial.ipynb
                # we want the dataloader to return a tuple with response, stimulus, pupil fixation, 
                # use built-in deeplake pytorch dataloader method instead
                # docs here: https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.pytorch
                
            
            # ds.pytorch(tensors=tuple(data_keys), pin_memory=True, # not sure whether to pin memory or not
            #         num_workers=0,
            #         batch_size=batch_size,
            #         transform=partial(norm_and_cut, 
            #                           dataset_statistic=ds.info.statistics, frames=frames, data_keys = data_keys, channels_last=channels_last), # transform is the function norm_and_cut with following params input
            #         shuffle=True)



# class Preprocess(Transform):
#     # a class that will take in transforms for each data tensor, then compile them somehow?  and will be passed to the pytorch dataloader 
#     # i can call it a CompositeTransform maybe
    
#     def __init__(self,transforms):
#         # let transforms be a dict with keys equal to data_keys
#         # e.g. transforms = {'videos': some_transform,'responses': some_transform, }

#     def __call__(self):
        
