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

import ipdb 
        

def deeplake_loader_dict(
    paths,
    batch_size,
    preproc,
    train_val_split = [.80,.20],
    use_api = False, # option to download datasets directly from API
    cuda: bool = False,
    include_behavior=True,
    include_pupil_centers=True,
    include_optogenetics=False,
    include_pupil_centers_as_channels=False,
    channels_last=False,
    tiered_datasets = False, # use this when there are separate deeplake datasets for training and validation data. otherwise each set will be split
    tiers = ['train','val'],
    tier_suffixes = None,
    user_token=None, # deeplake[enterprise] credentials, necessary for using optimized dataloaders
    org_id=None
):
    """
    Added by Nicholas: function to return video loaders directly from (list of paths to local) deeplake dataset(s)
    
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        paths (list): list of paths for the deeplake datasets
        batch_size (int): batch size.
        cuda (bool, optional): whether to place the data on gpu or not.
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

    loaders_dict = {}
    for tier in tiers:
        loaders_dict[tier] = {}

    if tiered_datasets and (tier_suffixes is None):
        tier_suffixes = {tier: f'_{tier}' for tier in tiers} # possible suffixes to local deeplake paths - created to accomodate downloaded V1 scans
    # loader_keys = loaders_dict.keys() # keys for loader dict used by training functions
    
    # V1 datasets provided for sensorium competition are already separated by train/val/test
    for path in paths: # for each provided dataset 
        # if use_api is True:
        #     ds = deeplake.load(f'{path}',token=user_token, org_id=org_id)
        # else:
        #     ds = deeplake.load(f'{path}',token=user_token, org_id=org_id)
        ds = deeplake.load(f'{path}',token=user_token, org_id=org_id)
        if tiered_datasets:
            for tier, suffix in tier_suffixes.items():
                if path.endswith(suffix):
                    readout_key = path[:-len(suffix)]
                    # loader = ds.dataloader().batch(batch_size).pytorch().transform(preproc(ds))
                    loaders_dict[tier][ readout_key ] = ds.dataloader().batch(batch_size).pytorch().transform(preproc(ds))
                    # ipdb.set_trace()
                
        else:
            ds = deeplake.load(f'{path}',token=user_token, org_id=org_id)
            train_ds, val_ds = ds.random_split(train_val_split)
            loaders_dict['train'][ path ] = train_ds.dataloader().transform(preproc(train_ds)).batch(batch_size).pytorch() 
            loaders_dict['val'][ path ] = val_ds.dataloader().transform(preproc(val_ds)).batch(batch_size).pytorch() 
            ipdb.set_trace()
        # TODO: generalize this to more keys than just 'train' and 'val
        
    return loaders_dict