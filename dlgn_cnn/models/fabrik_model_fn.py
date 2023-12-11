from typing import Dict
import numpy as np
import torch
from dlgn_cnn.models.model_components import *

from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict, set_random_seed
from neuralpredictors.utils import get_module_output

# here import the model classes we will want to fit
# from .video_encoder import 
from .model_classes import BaseResponseModel
import ipdb 

def dlgn_base_model_fn(dataloaders: Dict, seed: int, **config) -> torch.nn.Module:
    """
    Builds model components specified in **config, with minimally one core and one readout module
    Calls model class constructor with given components
    Args:
        data_loaders: a dictionary of data loaders, (possibly tiered?)
        seed: random seed (e.g. for model initialization)
    Returns:
        Instance of torch.nn.Module
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]
    in_name, out_name = config.get('in_name'), config.get('out_name')    
    set_random_seed(seed)
    
    core = make_core(**config.get('core_config'))
    
    readout_config = impute_readout_config(core, dataloaders, config.get('readout_config'), in_name, out_name)
    readout = make_readout(dataloaders, from_deeplake = True, **readout_config) # from_deeplake passed to models.utils.prepare_grid, which inits readout locs with cortical_coords
    
    gru = make_gru(**config.get('gru_config')) if config.get('gru') else None

    shifter = make_shifter(**config.get('shifter_config')) if config.get('shifter') else None
    
    modulator = make_modulator(**config.get('modulator_config')) if config.get('modulator') else None
    
    final_nonlin = make_nonlin(**config.get('final_nonlin_config'))


    # model = VideoFiringRateEncoder(
    #     core=core,
    #     readout=readout,
    #     shifter=shifter,
    #     modulator=modulator,
    #     gru=gru,
    #     elu_offset=0.0,
    #     nonlinearity_type="elu",
    #     nonlinearity_config=None,
    #     twoD_core = '3' not in config.get('core_type')
    # )
    model = BaseResponseModel(
        core=core,
        readout=readout,
        shifter=shifter,
        modulator=modulator,
        gru=gru,
        final_nonlin = final_nonlin,
        twoD_core = '3' not in config['core_config']['type'],
        in_name = in_name,
        out_name=out_name
    )

    return model


def impute_readout_config(core, dataloaders, readout_config, in_name='videos',out_name='responses'):
    # adds fields to readout config which depend on computations from dataloader
    # for FullGaussian2d readouts, compute, in_shape_dict, n_neurons_dict, mean_actiivty_dict

    session_shape_dict = get_dims_for_loader_dict(dataloaders) # in_name: (b, c, t, x, y)
    readout_config['n_neurons_dict'] = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    # estimate mean firing rate (initial value for bias of readout) from first batch
    print('imputing mean activity')
    ipdb.set_trace()
    readout_config['mean_activity_dict'] = {k: next(iter(dataloaders[k]))[out_name].to('cuda').mean(0).mean(-1).to('cpu') for k in dataloaders.keys() } 
    
    get_cxy = itemgetter(1,3,4)
    if '3d' not in str(core.__class__):
        collapse_time = lambda b_c_t_x_y: torch.Size(b_c_t_x_y[0]*b_c_t_x_y[2], b_c_t_x_y[1]) + b_c_t_x_y[3:]
        readout_config['in_shape_dict'] = {k: get_cxy(tuple(get_module_output(core, collapse_time(v[in_name])))) for k, v in session_shape_dict.items()}
    else:
        readout_config['in_shape_dict'] = {k: get_cxy(tuple(get_module_output(core, v[in_name]))) for k, v in session_shape_dict.items()}
    
    return readout_config