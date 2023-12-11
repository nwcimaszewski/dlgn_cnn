# Rewriting some of the dataloaders functions from sensorium 2023
# Added functionality of to load deeplake dataset objects from local storage or Deeplake API

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
import deeplake

from torchvision.transforms import Compose

from .deeplake_transforms import *

from .make_preproc import make_preproc
from typing import Dict

import ipdb 
import torch
import random
from copy import deepcopy # deepcopy needed because we edit transform config dicts in the make_preproc function, so to prevent this from changing some base config outside of the funciton we deepcopy. all other values of config that are list valued will not edited



def deeplake_dataloader_fn(seed:int, **config) -> Dict:
    """
    Added by Nicholas: function to return video loaders directly from (list of paths to local) deeplake dataset(s)
    
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        seed - random seed set by Deeplake when generating random splits
        config:
            'tiers': list of strings 'train' and 'test' (and others if need be)
            'splits': list of decimals indicating percentage of samples to be allotted to each tier
            'shuffle': whether to randomly shuffle samples within each dataloader
            'batch_size': batch size
            'data_dir': directory on server where Deeplake dataset objects are kept
            'dataset_paths': list of relative paths from within data_dir enumerating specific datasets to use

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    # super hacky - SET ALL SEEDS TO seed
    deeplake.random.seed(int(seed)) # set seed with deeplake, as randomness comes from .random_split
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)  # this sets both CPU and CUDA seeds for PyTorch
    
    tiers, tier_suffixes = config.get('tiers'), config.get('tier_suffixes')
    shuffle, batch_size = config.get('shuffle'), config.get('batch_size')

    paths = [ config.get('data_dir') + path for path in config.get('dataset_paths') ]
    loader_dict = {tier: {} for tier in tiers} if tiers else {} # loaders indexed by membership in train/test tier, as well as origin (mouse and session ID)
    
    for path in paths: # for each provided dataset 
        ds = deeplake.load(path, token=config.get('user_token'), org_id=config.get('org_id'))
        preproc = make_preproc( config.get('transforms') , ds, seed = seed) # returns preprocessor from torch.Compose()
        if tiers:
            if tier_suffixes:
                for tier, suffix in tier_suffixes.items():
                    if path.endswith(suffix): # assuming path ends with one of the provided suffixes (otherwise suffixes are wrong)
                        readout_key = path[:-len(suffix)]
                        loader_dict[tier][ readout_key ] = ds.dataloader().shuffle(shuffle=shuffle).batch(batch_size).pytorch().transform(preproc)
            else: # if we want tiers but dataset pathnames have no suffixes denoting tier
                ds_split = ds.random_split(config.get('tier_splits'))[:len(tiers)] # split randomly, only keep 2 tiers (train and val)
                for subset, tier in zip(ds_split, tiers):
                    preproc = make_preproc( config.get('transforms'), # deepcopy because dicts of dicts, so edits in make_preproc would edit base config
                                           subset, seed = seed) # this is actually redundant...because info objects get transferred, so statistics are actually incorrect technically...
                    loader_dict[tier][path] = subset.dataloader().shuffle(shuffle=shuffle).batch(batch_size).pytorch().transform(preproc)
        else: # if no tiers, so we just want to return one dict of dataloaders keyed by path
            loader_dict[path] = ds.dataloader().shuffle(shuffle=shuffle).batch(batch_size).pytorch().transform(preproc)
    # print(torch.cuda.memory_allocated())

    return loader_dict

'''
def subdlgn_dataloader_fn(seed:int, **config) -> Dict:
    """
    Returns severely subsampled dataset, so as to debug training by overfitting.
    
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        seed - random seed set by Deeplake when generating random splits
        config:
            'tiers': list of strings 'train' and 'test' (and others if need be)
            'splits': list of decimals indicating percentage of samples to be allotted to each tier
            'shuffle': whether to randomly shuffle samples within each dataloader
            'batch_size': batch size
            'data_dir': directory on server where Deeplake dataset objects are kept
            'dataset_paths': list of relative paths from within data_dir enumerating specific datasets to use
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    deeplake.random.seed(int(seed)) # set seed with deeplake, as randomness comes from .random_split

    tiers, splits = config.get('tiers'), config.get('splits') # splits should be one longer than tiers, because we will "throw out" most data, to overfit to some
    shuffle, batch_size = config.get('shuffle'), config.get('batch_size')


    paths = [ config.get('data_dir') + path for path in config.get('dataset_paths') ]
    loader_dict = {tier: {} for tier in tiers} if tiers else {} #
    
    for path in paths: # for each provided dataset 
        ds = deeplake.load(path, token=config.get('user_token'), org_id=config.get('org_id'))
        preproc = make_preproc( config.get('transform_configs'), ds) # returns preprocessor from torch.Compose()
        ds_split = ds.random_split( splits ) [:-1] # take some for training, some for val, discard rest ( [2::] )
        for subset, tier in zip(ds_split, tiers):
            preproc = make_preproc( config.get('transform_configs'), subset) # this is actually redundant...because info objects get transferred, so statistics are actually incorrect technically...
            loader_dict[tier][path] = subset.dataloader().shuffle(shuffle=shuffle).batch(batch_size).pytorch().transform(preproc)
        if not tiers:
            loader_dict[path] = ds.dataloader().shuffle(shuffle=shuffle).batch(batch_size).pytorch().transform(preproc)
    
    return loader_dict
'''