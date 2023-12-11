from dlgn_cnn.dataloading.deeplake_transforms import *
from neuralpredictors.data.transforms import *
from torchvision.transforms import Compose
from collections.abc import Iterable
from typing import Dict

def make_preproc(transform_configs: Dict, dataset, seed):
    # transform_configs is dict, with keys that are transforms imported here
    transforms = []
    for name, config in transform_configs.items():
        if type(config) is dict:
            transform = globals()[name]
            # STRONG RESTRICTION: any transform that computes something from the dataset needs to take in dataset as named arg "deeplake_dataset"
            if 'deeplake_dataset' in transform.__init__.__code__.co_varnames:
                config['deeplake_dataset'] = dataset
            if 'seed' in transform.__init__.__code__.co_varnames:
                # print(transform) 
                # import ipdb
                # ipdb.set_trace()
                config['seed'] = seed
            transforms.append(transform(**config))
        elif isinstance(config, Iterable): # if config has one transform name with multiple configs, we add it multiple times
            for subconfig in config:
                transform = globals()[name]
                if 'deeplake_dataset' in transform.__init__.__code__.co_varnames:
                    config['deeplake_dataset'] = dataset
                    transforms.append(transform(**subconfig))
                else:
                    transforms.append(transform(**subconfig))

    return Compose(transforms)
