import os
from functools import partial

import numpy as np
import torch
import wandb
from neuralpredictors.measures import modules
from neuralpredictors.training import LongCycler, early_stopping
from nnfabrik.utility.nn_helpers import set_random_seed
from tqdm import tqdm

from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss

import ipydb

# the following dictionaries can be fed as arguments to the trainer.
# these parameter values will be supplied by the experimenter (me, currently).
# I can create constructors for the dicts feeding in default values, e.g. with lambda functions.  Theese can be stored in this file and imported into jupyter notebooks

# the following functions construct dicts for arguments input to trainer, which just serves to organize parameters a bit more based on their specific use in training
# they make use of locals(), which returns the namespace internal to a function as a dict, allowing for default values along with user-defined values to be easily organized before input to trainer
# all default values taken from sensorium 2023 standard_trainer inputs

def make_stopping_config(
    stop_function="get_correlations",    
    interval=1,
    patience=5,
    epoch=0,    
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True):

    return locals()

def make_loss_config(
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    loss_accum_batch_n=None,
    detach_core=False):

    return locals()

def make_optim_config(
    verbose=True,
    print_step=1000,
    save_checkpoints=True,
    checkpoint_save_path="local/",
    checkpoint_save_prefix = None,
    chpt_save_step=15):
    
    return locals()

def make_lr_sched_config(
    lr_init=0.005,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
):
    return locals()

def make_wandb_config(use_wandb=True,
    wandb_project="factorised_core_parameter_search",
    wandb_entity="movies_parameter_search",
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None):
    
    return locals()
        
## TODO - add loading from checkpoints in case of train stop
def new_standard_trainer(
    model,
    dataloaders, # dataloaders
    seed, # seed
    from_deeplake=False,
    stopping_config = make_stopping_config(),
    loss_config = make_loss_config(),
    optim_config = make_optim_config(),
    lr_sched_config = make_lr_sched_config(),
    wandb_config = make_wandb_config(),
    **kwargs,
):

    # TODO: FINISH THIS!!
    # create cycler
    #
    
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        from_deeplake: whether dataloaders passed in return from deeplake dataset object, or are pytorch dataloaders from other format, e.g. neuralpredictors filetree dataset format
        **kwargs:
    Returns:

    """

        model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss) # fetch loss function from neuralpredictors
    
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["oracle"],
        device=device,
        per_neuron=False,
        avg=True,
        from_deeplake=from_deeplake,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = make_optimizer(*optim_config)
    # make optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    # make scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )



    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f"optim_step_count = {optim_step_count}")
    # this is 1...but why would it be the number of different scans in the training set?  i think i'll get rid of this

    # construct optimizer
    # construct lr scheduler
    # construct early stopper (takes in scheduler, uses stop fn (not loss fn) on validation set)
    # iterate through epochs of stopper
    #   iterate through batches
    #       update step: pass through model, compute loss, pass gradient back through optimizer
    #       for some subsample of batches / epochs 
    #           compute validation correlation per neuron
    #           plot pred vs. obs spike count traces, for neurons with highest validation correlation
    #           plot readout locations (sampled if sampled, mean if determin.) on the super imposed STRFs
    #               [ STRFs should be stored in info construct of Deeplake Dataset so it can be accessed from dataloader ]

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )

        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    batch_no_tot = 0
    # train over epochs

    keys = [input_key, output_key, pupil_key, behavior_key, opto_key]

   
    
    lr_scheduler = make_lr_scheduler(*lr_sched_config)
    
    # stop_fn = getattr()
    # loss_fn = getattr()

    early_stopping = early_stopping(model, stop_fn, *stop_config)

    for epoch, val_obj in early_stopping:
        for batch in dataloader:
            input = batch[input_key]
            behavior = batch[behavior_key]
            pupil = batch[pupil_key]
            opto = batch[opto_key]
            output = batch[output_key]

            model_output = model(input, behavior, pupil, opto)

            loss = loss_fn(model_output, output)
        



    def full_objective(model, dataloader, data_key, *args, **kwargs):
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        # todo - think how to avoid sum in model.core.regularizer()
        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(
                not detach_core
            ) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(
                model.core.regularizer()
            ) + model.readout.regularizer(data_key)
        if from_deeplake:
            for k in kwargs.keys():
                if k not in ["id", "index"]:
                    kwargs[k] = torch.Tensor(np.asarray(kwargs[k])).to(device)
        model_output = model(args[0].to(device), data_key=data_key, **kwargs)
        time_left = model_output.shape[1]

        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

        total_loss = (loss_scale * criterion(model_output,original_data,) + regularizers)
        
        return total_loss, criterion(model_output,original_data), regularizers

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss) # fetch loss function from neuralpredictors
    
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["oracle"],
        device=device,
        per_neuron=False,
        avg=True,
        from_deeplake=from_deeplake,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))
    
    # make optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    # make scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )



    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f"optim_step_count = {optim_step_count}")
    # this is 1...but why would it be the number of different scans in the training set?  i think i'll get rid of this

    # construct optimizer
    # construct lr scheduler
    # construct early stopper (takes in scheduler, uses stop fn (not loss fn) on validation set)
    # iterate through epochs of stopper
    #   iterate through batches
    #       update step: pass through model, compute loss, pass gradient back through optimizer
    #       for some subsample of batches / epochs 
    #           compute validation correlation per neuron
    #           plot pred vs. obs spike count traces, for neurons with highest validation correlation
    #           plot readout locations (sampled if sampled, mean if determin.) on the super imposed STRFs
    #               [ STRFs should be stored in info construct of Deeplake Dataset so it can be accessed from dataloader ]

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )

        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    batch_no_tot = 0
    # train over epochs
    for epoch, val_obj in early_stopping( # early_stopping creates a generator object over epochs, and calls to the specified objective function (stop_closure, in this case get_correlations) it also contains conditionals to decrease lr as according to scheduler and objective values
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data

            loss, pred_loss, reg_loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

            loss.backward()

            epoch_loss += loss.detach()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()

                #                 optimizer.zero_grad(set_to_none=False)
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        if save_checkpoints:
            if epoch % chpt_save_step == 0:
                torch.save(
                    model.state_dict(), f"{checkpoint_save_path}epoch_{epoch}.pth"
                )

        ## after - epoch-analysis

        validation_correlation = get_correlations(
            model,
            dataloaders["oracle"],
            device=device,
            as_dict=False,
            per_neuron=True, # change for per_neuron
            from_deeplake=from_deeplake,
        )
        val_loss, _, _ = full_objective(
            model,
            dataloaders["oracle"],
            data_key,
            *batch_args,
            **batch_kwargs,
            detach_core=detach_core,
        ).detach()
        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation.mean()}")

        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "Batch": batch_no_tot,
                "Epoch": epoch,
                "validation_correlation": validation_correlation,
                "mean_validation_correlation": validation_correlation.mean(),
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
        torch.save(model.state_dict(), f"{checkpoint_save_path}{checkpoint_save_prefix}_final.pth")

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["oracle"], device=device, as_dict=False, per_neuron=True, from_deeplake=from_deeplake,
    )
    print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")

    output = {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)
    if use_wandb:
        wandb.finish()

    # removing the checkpoints except the last one
    to_clean = os.listdir(checkpoint_save_path)
    for f2c in to_clean:
        if "epoch_" in f2c and f2c[-4:] == ".pth":
            os.remove(f"{checkpoint_save_path}{f2c}")

    return score, output, model.state_dict()

def plot_obs_pred_psth(stim_ind, neur_ind, dataset, model_pred):
    # will just feed in result of forward pass on dataset, which should be of same size as dataset.responses
    # as this is, stim_ind is not the scene ID of the video, but just the literal ordinal index of a scene in the dataset
    psth_obs = dataset.responses[stim_ind,neur_ind].numpy() # (len(neur_ind),300)
    psth_pred = model_pred[stim_ind][:,:,neur_ind] # not sure if there's a more syntactically uniform way to achieve this (like, identically for the two arrays)
    neur_ID = [ dataset.info.neuron_ids[ind] for ind in neur_ind ]
    t_ax = np.linspace(0,5,300)
    
    f, ax = plt.subplots(len(neur_ind),len(stim_ind), figsize = (len(stim_ind)*4,len(neur_ind)*2) )
    for i, n in enumerate(neur_ind):
        for j, s in enumerate(stim_ind):
            ax[i,j].plot(t_ax,psth_obs[j,i],'g')
            ax[i,j].plot(t_ax,psth_pred[j,:,i])
            ax[i,j].set(title=f' Pred vs. Obs PSTH \n Unit {neur_ID[i]}, Stim {s}',
                        xlabel = 'Time (s)',
                        ylabel = 'Spikes/sec',
                        xticks = [],
                        yticks = [])
            ax[i,j].title.set(fontsize=7)
            ax[i,j].xaxis.label.set(fontsize=6)
            ax[i,j].yaxis.label.set(fontsize=6)
    plt.tight_layout()

    return f

## TODO - add loading from checkpoints in case of train stop
def standard_trainer(
    model,
    dataloaders, # dataloaders
    seed, # seed
    avg_loss=False, # loss_config
    scale_loss=True, # loss_config
    loss_function="PoissonLoss", # loss_config
    stop_function="get_correlations", # stopping_config
    loss_accum_batch_n=None, # loss_config?
    device="cuda", # device
    verbose=True, # optim_config?
    interval=1, # stopping_config
    patience=5, # stopping_config
    epoch=0, # stopping_config
    lr_init=0.005, #lr_sched_config
    max_iter=200, # stopping_config
    maximize=True, # stopping_config, call it max_stop_fn
    tolerance=1e-6, # stopping_config
    restore_best=True, # stopping_config
    lr_decay_steps=3, # lr_sched_config
    lr_decay_factor=0.3, # lr_sched_config
    min_lr=0.0001, # lr_sched_config
    cb=None, # callback function?? fuer was
    detach_core=False, # loss_config
    use_wandb=True, # wandb_config
    wandb_project="factorised_core_parameter_search", # wandb_config
    wandb_entity="movies_parameter_search", # wandb_config
    wandb_name=None, # wandb_config
    wandb_model_config=None, # wandb_config
    wandb_dataset_config=None, # wandb_config
    print_step=1000, # optim_config
    save_checkpoints=True, # optim_config
    checkpoint_save_path="local/", # optim_config
    checkpoint_save_prefix = None, # optim_config
    chpt_save_step=15, # optim_config
    from_deeplake=False,
    **kwargs,
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function (THE STOPPING FUNCTION)
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        cb: whether to execute callback function
        **kwargs:

    Returns:

    """

    keys = [input_key, output_key, pupil_key, behavior_key, opto_key]

    optimizer = make_optimizer(*optim_config)
    
    lr_scheduler = make_lr_scheduler(*lr_sched_config)
    
    early_stopping = early_stopping(*stop_config)
                                    
    for epoch in epochs:
        for batch in dataloader:
            input = batch[input_key]
            behavior = batch[behavior_key]
            pupil = batch[pupil_key]
            opto = batch[opto_key]
            output = batch[output_key]

            model_output = model(input, behavior, pupil, opto)

            loss = loss_fn(model_output, output)



    def full_objective(model, dataloader, data_key, *args, **kwargs):
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        # todo - think how to avoid sum in model.core.regularizer()
        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(
                not detach_core
            ) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(
                model.core.regularizer()
            ) + model.readout.regularizer(data_key)
        if from_deeplake:
            for k in kwargs.keys():
                if k not in ["id", "index"]:
                    kwargs[k] = torch.Tensor(np.asarray(kwargs[k])).to(device)
        model_output = model(args[0].to(device), data_key=data_key, **kwargs)
        time_left = model_output.shape[1]

        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

        total_loss = (loss_scale * criterion(model_output,original_data,) + regularizers)
        
        return total_loss, criterion(model_output,original_data), regularizers

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss) # fetch loss function from neuralpredictors
    
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["oracle"],
        device=device,
        per_neuron=False,
        avg=True,
        from_deeplake=from_deeplake,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )



    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f"optim_step_count = {optim_step_count}")
    # this is 1...but why would it be the number of different scans in the training set?  i think i'll get rid of this

    # construct optimizer
    # construct lr scheduler
    # construct early stopper (takes in scheduler, uses stop fn (not loss fn) on validation set)
    # iterate through epochs of stopper
    #   iterate through batches
    #       update step: pass through model, compute loss, pass gradient back through optimizer
    #       for some subsample of batches / epochs 
    #           compute validation correlation per neuron
    #           plot pred vs. obs spike count traces, for neurons with highest validation correlation
    #           plot readout locations (sampled if sampled, mean if determin.) on the super imposed STRFs
    #               [ STRFs should be stored in info construct of Deeplake Dataset so it can be accessed from dataloader ]

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )

        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    batch_no_tot = 0
    # train over epochs
    for epoch, val_obj in early_stopping( # early_stopping creates a generator object over epochs, and calls to the specified objective function (stop_closure, in this case get_correlations) it also contains conditionals to decrease lr as according to scheduler and objective values
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data

            loss, pred_loss, reg_loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

            loss.backward()

            epoch_loss += loss.detach()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()

                #                 optimizer.zero_grad(set_to_none=False)
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        if save_checkpoints:
            if epoch % chpt_save_step == 0:
                torch.save(
                    model.state_dict(), f"{checkpoint_save_path}epoch_{epoch}.pth"
                )

        ## after - epoch-analysis

        validation_correlation = get_correlations(
            model,
            dataloaders["oracle"],
            device=device,
            as_dict=False,
            per_neuron=True, # change for per_neuron
            from_deeplake=from_deeplake,
        )
        val_loss, _, _ = full_objective(
            model,
            dataloaders["oracle"],
            data_key,
            *batch_args,
            **batch_kwargs,
            detach_core=detach_core,
        ).detach()
        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation.mean()}")

        # TODO:
        # # plot sample prediction for wandb
        # stim_ind

        
        
        # fig = plot_obs_pred_psth(stim_ind, neur_ind, dataloaders['train'].dataset, model_pred)

        
        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "Batch": batch_no_tot,
                "Epoch": epoch,
                "validation_correlation": validation_correlation,
                "mean_validation_correlation": validation_correlation.mean(),
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
        torch.save(model.state_dict(), f"{checkpoint_save_path}{checkpoint_save_prefix}_final.pth")

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["oracle"], device=device, as_dict=False, per_neuron=True, from_deeplake=from_deeplake,
    )
    print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")

    output = {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)
    if use_wandb:
        wandb.finish()

    # removing the checkpoints except the last one
    to_clean = os.listdir(checkpoint_save_path)
    for f2c in to_clean:
        if "epoch_" in f2c and f2c[-4:] == ".pth":
            os.remove(f"{checkpoint_save_path}{f2c}")

    return score, output, model.state_dict()