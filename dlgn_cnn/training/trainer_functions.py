import os
from functools import partial

import numpy as np
import torch
import wandb
from neuralpredictors.measures import modules
from neuralpredictors.training import LongCycler, early_stopping
from nnfabrik.utility.nn_helpers import set_random_seed
from tqdm import tqdm

import matplotlib.pyplot as plt

from . import scores

from .scores import get_correlations, get_poisson_loss

from lipstick import update_fig

from dlgn_cnn.dataloading import viz

import ipdb

# the following dictionaries can be fed as arguments to the trainer.
# these parameter values will be supplied by the experimenter (me, currently).
# I can create constructors for the dicts feeding in default values, e.g. with lambda functions.  Theese can be stored in this file and imported into jupyter notebooks

# the following functions construct dicts for arguments input to trainer, which just serves to organize parameters a bit more based on their specific use in training
# they make use of locals(), which returns the namespace internal to a function as a dict, allowing for default values along with user-defined values to be easily organized before input to trainer
# all default values taken from sensorium 2023 standard_trainer inputs


## TODO - add loading from checkpoints in case of train stop
def standard_trainer_with_plots(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    detach_core=False,
    use_wandb=True,
    wandb_project="factorised_core_parameter_search",
    wandb_entity="movies_parameter_search",
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None,
    print_step=1000,
    save_checkpoints=True,
    checkpoint_save_path="local/",
    checkpoint_save_prefix = None,
    chpt_save_step=15,
    from_deeplake=False,
    pred_latency=0, # try [0, 1, 3, 8]
    standardize_response_plots=True,
    in_name = 'videos',
    out_name = 'responses',
    opto_name = 'opto',
    **kwargs,
):
    """

    Args:
        model: VideoFiringRateEncoder instance
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to   accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        cb: whether to execute callback function
        **kwargs:

    Returns:

    """

    def full_objective(model, loader_dict, dataset_key, *args, **kwargs): # args = list of data, **kwargs = dict of batch, 
        loss_scale = (
            np.sqrt(len(loader_dict[dataset_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        # todo - think how to avoid sum in model.core.regularizer()
        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(
                not detach_core
            ) * model.core.regularizer() + model.readout.regularizer(dataset_key)
        else:
            regularizers = int(not detach_core) * sum(
                model.core.regularizer()
            ) + model.readout.regularizer(dataset_key)
        if from_deeplake:
            for k in kwargs.keys():
                if k not in ["id", "index"]:
                    kwargs[k] = torch.Tensor(np.asarray(kwargs[k])).to(device) # necessary to convert Deeplake tensor to np then to torch (then on device)
        model_output = model(kwargs[in_name].to(device), data_key=dataset_key, **kwargs) # data_key is passed into readout, so the model knows which readout to use (there are multiple)

        time_left = model_output.shape[1] # time axis is 1 in model output, for some reason...
        # print(time_left)
        true_output = kwargs[out_name].transpose(2, 1)[:, -time_left:, :].to(device)

        total_loss = (loss_scale * criterion(model_output,true_output) + regularizers)
        
        return total_loss, criterion(model_output,true_output), regularizers

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss) # fetch loss function from neuralpredictors
    
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["val"],
        device=device,
        deeplake_ds=from_deeplake,
        scalar=True,
        skip=pred_latency
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
        for batch_no, (dataset_key, data) in tqdm( # data_key is the same for every batch, indexing different loaders from different datasets.  data is batch as dict
            enumerate(LongCycler(dataloaders["train"])), #LongCycler iterates through batches and returns scan key each time
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)
            batch_kwargs = data._asdict() if not isinstance(data, dict) else data

            loss, pred_loss, reg_loss = full_objective(
                model,
                dataloaders["train"],
                dataset_key,
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

        val_corr, val_corr_per_set = get_correlations(
            model,
            dataloaders["val"],
            device=device,
            per_neuron=True, # change for per_neuron
            from_deeplake=from_deeplake,
            skip = pred_latency
        )
        val_loss, val_loss_objective, val_loss_reg = full_objective(
            model,
            dataloaders["val"],
            dataset_key,
            *batch_args,
            **batch_kwargs,
            detach_core=detach_core,
        )
        val_loss = val_loss.detach()
        val_loss_objective = val_loss_objective.detach()
        val_loss_reg = val_loss_reg.detach()
        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={val_corr.mean()}")

        # Prepare logging dict

        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                # "Batch": batch_no_tot,
                # "Epoch": epoch,
                # 'Validation Correlation per Dataset': val_corr_per_set,
                "validation_correlation": val_corr,
                "Mean validation correlation": val_corr.mean(),
                "Epoch total validation loss": val_loss,
                "Epoch validation loss - Poisson loss": val_loss_objective,
                "Epoch validation loss - Regularization": val_loss_reg,
                "Epoch": epoch,
            }

         ## PLOTTING ##
        if epoch % 3 == 0: # every 3rd epoch
            
            with torch.no_grad():
                print('Neurons with 3 highest correlations with model predictions')
                rng = np.random.default_rng()
                num_example_neurons = 3
                num_example_scenes = 3
                for k,v in dataloaders['val'].items():

                    print('Neurons with 3 highest correlations with model predictions')
                    best_unit_idx = np.argsort(val_corr_per_set[k])[-num_example_neurons:].tolist()
                    print(f'{k}: {best_unit_idx}, with respective correlations {np.sort(val_corr_per_set[k])[-num_example_neurons:].tolist()}')

                    # i =  np.random.choice(len(dataloaders['val']))
                    # data_key = dataloaders['val'].keys()[i]
                    
                    batch_ind = rng.integers(len(iter(v))-1) # select random batch, excluding last batch, as it could be fewer scenes than we want to visualize, which crashes training

                    for i, test_batch in enumerate(v):
                        if i == batch_ind:
                            break

                    test_vids = torch.Tensor( test_batch[in_name].numpy() ).to(device) # will need this for opto as well
                    test_resp = test_batch[out_name]
                    # test_opto = test_batch[opto_name]

                    # ipdb.set_trace()

                    # pick best and worst correlated units
                    # best_unit_idx, worst_unit_idx = np.argsort(val_corr)[:3].tolist(), np.argsort(val_corr)[-3:].tolist() # illustrate worst and best predictions
                    
                    # get neural responses to scene
                    # scale response to match training data
                    # TODO: generalize this to apply dataloaders.transform or whatever
                    # selected_resp = selected_resp / val_ds.info.statistics['responses']['std'][best_unit_idx+worst_unit_idx]
                    # get model responses to scene


                    val_out = ( model( test_vids, data_key=dataset_key ).detach().to('cpu')[:,:,best_unit_idx] ).transpose(1,2) # torch.Size([1, n, t ]) detach and select prediction of relevant neurons
                    neur_ID = [ v.dataset.info.neuron_ids[ind] for ind in best_unit_idx ]

                    scene_idx = np.random.choice(np.arange(test_vids.shape[0]),
                                                 size=num_example_scenes,replace=False) # select random scenes from batch


                    fig, axs = plt.subplots(len(scene_idx),len(best_unit_idx),dpi=100, figsize=(len(best_unit_idx)*4,len(scene_idx)*2))
                    t = np.arange(0,val_out.shape[2]/60,step=1/60) # 1 sec clips at 60Hz, 300 bins
                
                    for i, scene in enumerate(scene_idx):
                        for j, neur in enumerate(best_unit_idx):
                            ax = axs[i,j]
                            ax.plot(t[pred_latency::],test_resp[i,j][pred_latency::],label='True')
                            ax.plot(t[pred_latency::],val_out[i,j][pred_latency::],label='model')
                            ax.set(title=f'Unit {neur_ID[j]}, Scene {scene}')
                            ax.text(ax.get_xlim()[1]*0.7,ax.get_ylim()[1]*0.9,f'$r^2={100*val_corr_per_set[k][neur]:.02}%$',fontdict={'size':10})  
                    for ax in axs.flat:
                        ax.set(xticks=[], yticks=[])
                    plt.tight_layout()

                    wandb_dict[f'Results/Example Predictions - Dataset {k}'] = fig
                    
                
            # ipdb.set_trace()
                # batch_size = 4
                # sampled_ro_loc = model.readout.sample_grid(batch_size,sample=False) # with sample=False, this will just return the mean
                
                # fig, axs = plt.subplots(len(scene_idx),len(neur_idx),dpi=100, figsize=(len(best_unit_idx)*4,len(scene_idx)*2))
                # for j, neur in enumerate(neur_idx):
                    # ax.imshow(val_ds.info.strf[j,])
                    # plt.scatter(sampled_ro_loc[j,:])
                # log dict
                wandb.log(wandb_dict)
                # update_fig(fig,axs)
        else:
            wandb.log(wandb_dict)

        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
        torch.save(model.state_dict(), f"{checkpoint_save_path}{checkpoint_save_prefix}_final.pth")

    # Compute avg validation and test correlation
    val_corr, val_corr_by_set  = get_correlations(
        model, dataloaders["val"], device=device, as_dict=False, per_neuron=True, from_deeplake=from_deeplake,
    )
    print(f"\n\n FINAL validation_correlation {val_corr} \n\n")

    output = {}
    output['validation_corr'] = val_corr
    output['validation_corr_by_set'] = val_corr

    score = np.mean(val_corr)
    if use_wandb:
        wandb.finish()

    # removing the checkpoints except the last one
    to_clean = os.listdir(checkpoint_save_path)
    for f2c in to_clean:
        if "epoch_" in f2c and f2c[-4:] == ".pth":
            os.remove(f"{checkpoint_save_path}{f2c}")

    return score, output, model.state_dict()


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
        dataloaders=dataloaders["val"],
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
            dataloaders["val"],
            device=device,
            as_dict=False,
            per_neuron=True, # change for per_neuron
            from_deeplake=from_deeplake,
        )
        val_loss, _, _ = full_objective(
            model,
            dataloaders["val"],
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
        model, dataloaders["val"], device=device, as_dict=False, per_neuron=True, from_deeplake=from_deeplake,
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


### TRAINER FUNCTIONS
# standard_trainer is copied from sensorium 2023
# standard_trainer_with_plots plots random pred vs. obs traces during training and logs it with wandb
# new_standard_trainer was an attempt at increased concision, which i realized was perhaps not a priority

        
## TODO - add loading from checkpoints in case of train stop
def new_standard_trainer(
    model,
    dataloaders, # dataloaders
    seed, # seed
    from_deeplake=False,
    device='cuda',
    stopping_config = make_stopping_config(),
    loss_config = make_loss_config(),
    optim_config = make_optim_config(),
    lr_sched_config = make_lr_sched_config(),
    wandb_config = make_wandb_config(),
    **kwargs,
):
    pass

    # TODO: FINISH THIS!!
    # create cycler
    # """

    # Args:
    #     model: model to be trained
    #     dataloaders: dataloaders containing the data to train the model with
    #     seed: random seed
    #     from_deeplake: whether dataloaders passed in return from deeplake dataset object, or are pytorch dataloaders from other format, e.g. neuralpredictors filetree dataset format
    #     **kwargs:
    # Returns:

    # """

    # model.to(device)
    # set_random_seed(seed)
    # model.train()

    # criterion = getattr(modules, loss_function)(avg=avg_loss) # fetch loss function from neuralpredictors
    
    # stop_closure = partial(
    #     getattr(scores, stop_function),
    #     dataloaders=dataloaders["val"],
    #     device=device,
    #     per_neuron=False,
    #     avg=True,
    #     from_deeplake=from_deeplake,
    # )

    # n_iterations = len(LongCycler(dataloaders["train"]))

    # optimizer = make_optimizer(*optim_config)
    # # make optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    # # make scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="max" if maximize else "min",
    #     factor=lr_decay_factor,
    #     patience=patience,
    #     threshold=tolerance,
    #     min_lr=min_lr,
    #     verbose=verbose,
    #     threshold_mode="abs",
    # )



    # # set the number of iterations over which you would like to accummulate gradients
    # optim_step_count = (
    #     len(dataloaders["train"].keys())
    #     if loss_accum_batch_n is None
    #     else loss_accum_batch_n
    # )
    # print(f"optim_step_count = {optim_step_count}")
    # # this is 1...but why would it be the number of different scans in the training set?  i think i'll get rid of this

    # # construct optimizer
    # # construct lr scheduler
    # # construct early stopper (takes in scheduler, uses stop fn (not loss fn) on validation set)
    # # iterate through epochs of stopper
    # #   iterate through batches
    # #       update step: pass through model, compute loss, pass gradient back through optimizer
    # #       for some subsample of batches / epochs 
    # #           compute validation correlation per neuron
    # #           plot pred vs. obs spike count traces, for neurons with highest validation correlation
    # #           plot readout locations (sampled if sampled, mean if determin.) on the super imposed STRFs
    # #               [ STRFs should be stored in info construct of Deeplake Dataset so it can be accessed from dataloader ]

    # if use_wandb:
    #     wandb.init(
    #         project=wandb_project,
    #         entity=wandb_entity,
    #         # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #         name=wandb_name,
    #         # Track hyperparameters and run metadata
    #         config={
    #             "learning_rate": lr_init,
    #             "architecture": wandb_model_config,
    #             "dataset": wandb_dataset_config,
    #             "cur_epochs": max_iter,
    #             "starting epoch": epoch,
    #             "lr_decay_steps": lr_decay_steps,
    #             "lr_decay_factor": lr_decay_factor,
    #             "min_lr": min_lr,
    #         },
    #     )

    #     wandb.define_metric(name="Epoch", hidden=True)
    #     wandb.define_metric(name="Batch", hidden=True)

    # batch_no_tot = 0
    # # train over epochs

    # keys = [input_key, output_key, pupil_key, behavior_key, opto_key]

   
    
    # lr_scheduler = make_lr_scheduler(*lr_sched_config)
    
    # # stop_fn = getattr()
    # # loss_fn = getattr()

    # early_stopping = early_stopping(model, stop_fn, *stop_config)

    # for epoch, val_obj in early_stopping:
    #     for batch in dataloader:
    #         input = batch[input_key]
    #         behavior = batch[behavior_key]
    #         pupil = batch[pupil_key]
    #         opto = batch[opto_key]
    #         output = batch[output_key]

    #         model_output = model(input, behavior, pupil, opto)

    #         loss = loss_fn(model_output, output)
        



    # def full_objective(model, dataloader, data_key, *args, **kwargs):
    #     loss_scale = (
    #         np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
    #         if scale_loss
    #         else 1.0
    #     )
    #     # todo - think how to avoid sum in model.core.regularizer()
    #     if not isinstance(model.core.regularizer(), tuple):
    #         regularizers = int(
    #             not detach_core
    #         ) * model.core.regularizer() + model.readout.regularizer(data_key)
    #     else:
    #         regularizers = int(not detach_core) * sum(
    #             model.core.regularizer()
    #         ) + model.readout.regularizer(data_key)
    #     if from_deeplake:
    #         for k in kwargs.keys():
    #             if k not in ["id", "index"]:
    #                 kwargs[k] = torch.Tensor(np.asarray(kwargs[k])).to(device)
    #     model_output = model(args[0].to(device), data_key=data_key, **kwargs)
    #     time_left = model_output.shape[1]

    #     original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

    #     total_loss = (loss_scale * criterion(model_output,original_data,) + regularizers)
        
    #     return total_loss, criterion(model_output,original_data), regularizers

    # ##### Model training ####################################################################################################
    # model.to(device)
    # set_random_seed(seed)
    # model.train()

    # criterion = getattr(modules, loss_function)(avg=avg_loss) # fetch loss function from neuralpredictors
    
    # stop_closure = partial(
    #     getattr(scores, stop_function),
    #     dataloaders=dataloaders["val"],
    #     device=device,
    #     per_neuron=False,
    #     avg=True,
    #     from_deeplake=from_deeplake,
    # )

    # n_iterations = len(LongCycler(dataloaders["train"]))
    
    # # make optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    # # make scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="max" if maximize else "min",
    #     factor=lr_decay_factor,
    #     patience=patience,
    #     threshold=tolerance,
    #     min_lr=min_lr,
    #     verbose=verbose,
    #     threshold_mode="abs",
    # )



    # # set the number of iterations over which you would like to accummulate gradients
    # optim_step_count = (
    #     len(dataloaders["train"].keys())
    #     if loss_accum_batch_n is None
    #     else loss_accum_batch_n
    # )
    # print(f"optim_step_count = {optim_step_count}")
    # # this is 1...but why would it be the number of different scans in the training set?  i think i'll get rid of this

    # # construct optimizer
    # # construct lr scheduler
    # # construct early stopper (takes in scheduler, uses stop fn (not loss fn) on validation set)
    # # iterate through epochs of stopper
    # #   iterate through batches
    # #       update step: pass through model, compute loss, pass gradient back through optimizer
    # #       for some subsample of batches / epochs 
    # #           compute validation correlation per neuron
    # #           plot pred vs. obs spike count traces, for neurons with highest validation correlation
    # #           plot readout locations (sampled if sampled, mean if determin.) on the super imposed STRFs
    # #               [ STRFs should be stored in info construct of Deeplake Dataset so it can be accessed from dataloader ]

    # if use_wandb:
    #     wandb.init(
    #         project=wandb_project,
    #         entity=wandb_entity,
    #         # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #         name=wandb_name,
    #         # Track hyperparameters and run metadata
    #         config={
    #             "learning_rate": lr_init,
    #             "architecture": wandb_model_config,
    #             "dataset": wandb_dataset_config,
    #             "cur_epochs": max_iter,
    #             "starting epoch": epoch,
    #             "lr_decay_steps": lr_decay_steps,
    #             "lr_decay_factor": lr_decay_factor,
    #             "min_lr": min_lr,
    #         },
    #     )

    #     wandb.define_metric(name="Epoch", hidden=True)
    #     wandb.define_metric(name="Batch", hidden=True)

    # batch_no_tot = 0
    # # train over epochs
    # for epoch, val_obj in early_stopping( # early_stopping creates a generator object over epochs, and calls to the specified objective function (stop_closure, in this case get_correlations) it also contains conditionals to decrease lr as according to scheduler and objective values
    #     model,
    #     stop_closure,
    #     interval=interval,
    #     patience=patience,
    #     start=epoch,
    #     max_iter=max_iter,
    #     maximize=maximize,
    #     tolerance=tolerance,
    #     restore_best=restore_best,
    #     scheduler=scheduler,
    #     lr_decay_steps=lr_decay_steps,
    # ):
    #     # executes callback function if passed in keyword args
    #     if cb is not None:
    #         cb()

    #     # train over batches
    #     optimizer.zero_grad(set_to_none=True)
    #     epoch_loss = 0
    #     epoch_val_loss = 0

    #     train_cycler = LongCycler(dataloaders["train"])

    #     for batch_no, (dataset_key, data_batch) in tqdm( #(number of batch, (key of individual dataloader, batch of data as IterableOrderedDict)
    #         enumerate(train_cycler), 
    #         total=n_iterations,
    #         desc="Epoch {}".format(epoch),
    #     ):
        
    #         batch_no_tot += 1
    #         batch_values_list = list(data_batch)

    #         batch_dict = data_batch._asdict() if not isinstance(data_batch, dict) else data_batch

    #         loss, pred_loss, reg_loss = full_objective(
    #             model,
    #             dataloaders["train"],
    #             dataset_key,
    #             *batch_values_list,
    #             **batch_dict,
    #             detach_core=detach_core,
    #         )

    #         loss.backward()

    #         epoch_loss += loss.detach()
    #         if (batch_no + 1) % optim_step_count == 0:
    #             optimizer.step()

    #             #                 optimizer.zero_grad(set_to_none=False)
    #             optimizer.zero_grad(set_to_none=True)

    #     model.eval()
    #     if save_checkpoints:
    #         if epoch % chpt_save_step == 0:
    #             torch.save(
    #                 model.state_dict(), f"{checkpoint_save_path}epoch_{epoch}.pth"
    #             )

    #     ## after - epoch-analysis

    #     validation_correlation = get_correlations(
    #         model,
    #         dataloaders["val"],
    #         device=device,
    #         as_dict=False,
    #         per_neuron=True, # change for per_neuron
    #         from_deeplake=from_deeplake,
    #     )
    #     val_loss, _, _ = full_objective(
    #         model,
    #         dataloaders["val"],
    #         dataset_key,
    #         *batch_args,
    #         **batch_kwargs,
    #         detach_core=detach_core,
    #     ).detach()
    #     print(
    #         f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
    #     )
    #     print(f"EPOCH={epoch}  validation_correlation={validation_correlation.mean()}")

    #     if use_wandb:
    #         wandb_dict = {
    #             "Epoch Train loss": epoch_loss,
    #             "Batch": batch_no_tot,
    #             "Epoch": epoch,
    #             "validation_correlation": validation_correlation,
    #             "mean_validation_correlation": validation_correlation.mean(),
    #             "Epoch validation loss": val_loss,
    #             "Epoch": epoch,
    #         }
    #         wandb.log(wandb_dict)
    #     model.train()

    # ##### Model evaluation ####################################################################################################
    # model.eval()
    # if save_checkpoints:
    #     torch.save(model.state_dict(), f"{checkpoint_save_path}{checkpoint_save_prefix}_final.pth")

    # # Compute avg validation and test correlation
    # validation_correlation = get_correlations(
    #     model, dataloaders["val"], device=device, as_dict=False, per_neuron=True, from_deeplake=from_deeplake,
    # )
    # print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")

    # output = {}
    # output["validation_corr"] = validation_correlation

    # score = np.mean(validation_correlation)
    # if use_wandb:
    #     wandb.finish()

    # # removing the checkpoints except the last one
    # to_clean = os.listdir(checkpoint_save_path)
    # for f2c in to_clean:
    #     if "epoch_" in f2c and f2c[-4:] == ".pth":
    #         os.remove(f"{checkpoint_save_path}{f2c}")

    # return score, output, model.state_dict()
