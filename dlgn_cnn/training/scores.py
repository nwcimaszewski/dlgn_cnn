import warnings

import numpy as np
import torch
from neuralpredictors.measures.np_functions import corr
from neuralpredictors.training import device_state
import ipdb


# local: predict 
def model_predictions(
    model, dataloader, data_key, device="cpu", skip=0, from_deeplake=False, in_name = 'videos', out_name = 'videos'
):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: list of batches of true responses, i.e. (b, n, t) shape arrays
        output: list of batches of true responses, i.e. (b, n, t) shape arrays

    """
    
    target, output = [], []
    for batch in dataloader: # for one batch
        batch_dict = batch._asdict() if not isinstance(batch, dict) else batch
        if from_deeplake:
            images = batch_dict[in_name] # take videos from batch (b,c,t,x,y)
            responses = batch_dict[out_name] # (b,n,t)
        else:
            images, responses = (
                batch[:2]
                if not isinstance(batch, dict)
                else (batch["videos"], batch["responses"])
            )
        # print('scores.model_predictions')
        with torch.no_grad():
            resp = responses.detach().cpu()[:, :, skip:] # truncate "skip" many frames - (b, n, t2)
            target = target + list(resp) # append to list of all response batches
            with device_state(model, device): 
                out = (
                    model(images.to(device), data_key=data_key, **batch_dict) # (b,n,t)
                    .detach()
                    .cpu()[:, -resp.shape[-1] :, :] # (b,t,n,?)
                )
                assert (
                    out.shape[1] == resp.shape[-1]
                ), f"model prediction is too short ({out.shape[1]} vs {resp.shape[-1]})" 
                # ipdb.set_trace()
                output = output + list(out.permute(0, 2, 1)) # append to list of all output batches

    return target, output


def get_correlations(
    model,
    dataloaders,
    tier=None,
    device="cpu",
    from_deeplake=False,
    skip=0,
    scalar=False,
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
    correlations = {}
    dataloaders = dataloaders[tier] if tier is not None else dataloaders # if a dict of dataloaders, or a dict of dicts of (dataloaders)
    # print(device, 'score.get_correlations')
    for k, v in dataloaders.items(): # for individual loaders
        target, output = model_predictions( # list of batchwise response tensor
            dataloader=v,
            model=model,
            data_key=k, # key used to index model readouts
            device=device,
            from_deeplake=from_deeplake,
            skip=skip
        )

        # correlations = [] # list of correlations across all neurons across datasets 

        target = np.concatenate(target, axis=1).T  # flattens across time and batch, so for 57 test scenes
        output = np.concatenate(output, axis=1).T  # transpose to (frames, neurons) shape
        correlations[k] = corr(target, output, axis=0) # compute between responses over frames

        # ipdb.set_trace()

        if np.any(np.isnan(correlations[k])):
            warnings.warn(
                "{}% NaN correlations computed , NaNs will be set to Zero.".format(
                    np.isnan(correlations[k]).mean() * 100
                )
            )
        correlations[k][np.isnan(correlations[k])] = 0
    composite_correlations = np.hstack([val for val in correlations.values()]) # as single np array
    
    if scalar:
        return composite_correlations.mean()
    else:
       return composite_correlations, correlations # return in both array and dict format


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
    ipdb.set_trace()

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
