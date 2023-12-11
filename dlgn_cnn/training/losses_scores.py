import warnings

import numpy as np
import torch
from neuralpredictors.measures.np_functions import corr
from neuralpredictors.training import device_state
import ipdb


# This file contains functions used to compute some metric of:
# loss (prediciton mismatch, to be backpropagated through network), or
# score (predictive performance)
# these functions are written to be as general as possible but some
# generally, these functions take in the model prediction and the target response

def call_stop_fn(model, score_name='get_tier_score', **configs):
    '''Early stopping requires objective(model), but score functions are stored as attributes of models

    Params:
    score_name (str): name of attribute    
    '''
    stop_fn = getattr(model, score_name)
    return stop_fn(**configs)

def poisson_loss(model_output, model_target, truncate, 
                 reduction = (0,1,2), device='cuda', eps: int = 1e-8, **kwargs):
    '''
    :param torch.Tensor model_output: (b, c, t), output of model
    :param torch.Tensor model_target: (b, c, t), ground truth response
    :param int truncate: number of time points to truncate from front of true response.  necessary for 3D conv models without temporal padding, which will produce output shorter than true response
    :param Tuple reduction: axes over which to sum. for loss per neuron, omit 1, for loss over time, omit 2, for loss per stimulus, omit 0
    :param aggregate: method to reduce over axes.  
        NOT IMPLEMENTED - for now, only sum.  
        from probabilistic perspective, consider each scalar component of loss statistically independent (conditioned on stimulus, other model inputs), so joint probability is sum of individual (marginal) probabilities.  For Poisson loss, this is motivated by model of neurons spike counts as inhomogeneous Poisson processes, which we aim to infer conditional intensity function.
    '''
    model_output, model_target = model_output.to(device), model_target[:,:,truncate:].to(device)
    loss_array = model_output - model_target * torch.log(model_output + eps)
    # ipdb.set_trace()
    return loss_array.sum(reduction) if reduction else loss_array

def l2_metric_loss(model_output, model_target, truncate, 
                 reduction = (0,1,2), device='cuda', eps: int = 1e-8, **kwargs):
    '''
    :param torch.Tensor model_output: (b, c, t), output of model
    :param torch.Tensor model_target: (b, c, t), ground truth response
    :param int truncate: number of time points to truncate from front of true response.  necessary for 3D conv models without temporal padding, which will produce output shorter than true response
    :param Tuple reduction: axes over which to sum. for loss per neuron, omit 1, for loss over time, omit 2, for loss per stimulus, omit 0
    :param aggregate: method to reduce over axes.  
        NOT IMPLEMENTED - for now, only sum.  
        from probabilistic perspective, consider each scalar component of loss statistically independent (conditioned on stimulus, other model inputs), so joint probability is sum of individual (marginal) probabilities.  For Poisson loss, this is motivated by model of neurons spike counts as inhomogeneous Poisson processes, which we aim to infer conditional intensity function.
    '''

    model_output, model_target = model_output.to(device), model_target[:,:,truncate:].to(device)
    loss_array = (model_output - model_target) ** 2 # SSE or MSE, depending on reduction (sum in our case)
    # ipdb.set_trace()
    return loss_array.sum(reduction) if reduction else loss_array


# note - compare logging of wandb corr_score(, reduction=(-1)).mean(0,1), corr_score(, reduction=(0,-1)).mean(1)
# won't reduce over (1), because we don't care how predictions disperse from mean over other neurons, don't want to weight by std computed from other neurons...
def corr_score(model_output, model_target, truncate, 
                 reduction = (-1), device='cuda', eps: int = 1e-8, **kwargs):
    # regardless of reduction, we take mean / std over axis -1, because we want correlation (between model and truth) of each 
    X, Y = model_output.to(device), model_target[:,:,truncate:].to(device) # (b,c,t)
    # axis = -1 -> (scaling responses differently depending on std of neuron to specific stimulus!  even if we take mean over all stimuli later i.e. we weight predictions of more static responses more highly than predictions of more dynamic responses)
    # axis = (0,-1) -> (scaling responses depending on std of neuron across time and batch!  weight video responses with high/low std equally, )

    # reduction -> output shape
    # (0) -> (c,t), (1) -> (b,t), (2) -> (b,c)
    # (0,1) ->  (t), (0,2) ->  (c), (1,2) ->  (b)
    X_normed =  (X - X.mean(reduction,keepdims=True) ) / ( X.std(reduction,keepdims=True, correction=0) + eps) # center by (b,c) mean, scale by (b,c) std 
    Y_normed = (Y - Y.mean(reduction,keepdims=True) ) / ( Y.std(reduction,keepdims=True, correction=0) + eps)
    
    score_array = (X_normed * Y_normed)
    # ipdb.set_trace()
    return score_array.mean(reduction,keepdims=True).detach().cpu() # .mean(everything but 1)?  or outside of function?  mean_ax param in loss/score getters of trainer?



'''
vestiges from Sensorium 2023

def correlation(
    model,
    dataloaders,
    tier=None,
    device="cpu",
    from_deeplake=False,
    latency=None,
    total=False,
    **kwargs,
):
    """
    Computes single-trial correlation between model prediction and true responses
    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        tier(str): the data-tier (train/test/val). If tier is None, then it is assumed that the the tier-key is not present.
        device (str, optional): device to compute on. Defaults to "cpu".
        as_dict (bool, optional): whether to return the results per data_key. Defaults to False.
        per_neuron (bool, optional): whether to return the results per neuron or averaged across neurons. Defaults to True.
    Returns:
        dict or np.ndarray: contains the correlation values.
    """
    corr_dict = {}
    dataloaders = dataloaders[tier] if tier is not None else dataloaders # if a dict of dataloaders, or a dict of dicts of (dataloaders)
    # print(device, 'score.get_correlations')
    for k, v in dataloaders.items(): # for individual loaders
        target, output = model_predictions( # list of batchwise response tensor
            dataloader=v,
            model=model,
            data_key=k, # key used to index model readouts
            device=device,
            from_deeplake=from_deeplake,
            skip=latency
        )

        # correlations = [] # list of correlations across all neurons across datasets 

        target = np.concatenate(target, axis=1).T  # flattens across time and batch, so for 57 test scenes
        output = np.concatenate(output, axis=1).T  # transpose to (frames, neurons) shape
        corr_dict[k] = corr(target, output, axis=0) # compute between responses over frames

        # ipdb.set_trace()

        if np.any(np.isnan(corr_dict[k])):
            warnings.warn(
                "{}% NaN correlations computed , NaNs will be set to Zero.".format(
                    np.isnan(corr_dict[k]).mean() * 100
                )
            )
        corr_dict[k][np.isnan(corr_dict[k])] = 0
    composite_correlations = np.hstack([val for val in corr_dict.values()]) # as single np array

    # print('Before return of scores.get_correlations')
    # print('Allocated:',torch.cuda.memory_allocated(),torch.cuda.max_memory_allocated())
    # print('Reserved:',torch.cuda.memory_reserved(),torch.cuda.max_memory_reserved())


    if scalar:
        return composite_correlations.mean()
    else:
       return composite_correlations, corr_dict # return in both array and dict format


def get_poisson_loss(
    model,
    dataloaders,
    device="cpu",
    as_dict=False,
    avg=False,
    per_neuron=True,
    eps=1e-12,
):
    poisson_loss = {}
    for k, v in dataloaders.items():
        target, output = model_predictions(
            dataloader=v, model=model, data_key=k, device=device
        )
        loss = output - target * np.log(output + eps) # ( predicted number of spikes ) minus ( true number of spikes times log of predicted number )
        poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    # ipdb.set_trace()

    nothing=[]

    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return (
                np.mean(np.hstack([v for v in poisson_loss.values()]))
                if avg
                else np.sum(np.hstack([v for v in poisson_loss.values()]))
            )
'''