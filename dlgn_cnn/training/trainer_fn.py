import os
from functools import partial
import numpy as np
import torch
import wandb
# from neuralpredictors.measures import modules
from nnfabrik.utility.nn_helpers import set_random_seed
from neuralpredictors.training import LongCycler, early_stopping
from nnfabrik.builder import resolve_fn
from dlgn_cnn.training.trainer_components import make_scheduler

from tqdm import tqdm
import matplotlib.pyplot as plt
from lipstick import update_fig
# from dlgn_cnn.dataloading import viz
import ipdb
from typing import *

from .utils import *


def trainer_fn_no_train(
    model: torch.nn.Module, 
    dataloaders: Dict, 
    seed: int, 
    uid: Dict, 
    cb: Callable, 
    **config
) -> Tuple[float, Any, Dict]:
    """
    Args:
        model: initialized model to train
        data_loaders: containing "train", "validation" and "test" data loaders
        seed: random seed
        uid: database keys that uniquely identify this trainer call
        cb: callback function to ping the database and potentially save the checkpoint
    Returns:
        score: performance score of the model
        output: user specified validation object based on the 'stop function'
        model_state: the full state_dict() of the trained model
    """

    # resolve loss, stopping, other score functions
    loss = partial(resolve_fn(config.get('loss_config').get('loss_fn'), 'dlgn_cnn.training.losses_scores'),
                    truncate = model.truncate, **(config.get('loss_config')) 
                   ) # loss does not depend on model...at least as we use it now
    # generally, score depends on some properties of model (namely, whether forward pass truncates temporal length), so partial call is left to Trainer constructor
    score_fn = partial( resolve_fn(config.get('score_config').get('score_fn'), 'dlgn_cnn.training.losses_scores'),
                       truncate = model.truncate, **config.get('score_config'))
    
    # for now - score is used for early stopping.  later we will add option for arbitrary add'l scores
    # score = partial( resolve_fn(config.get('score_fn'), 'dlgn_cnn.training.losses_scores'), )

    # if '3' in str(model.core.__class__):
    #     # TODO: generalize following line to cores with more than one 3D conv layer
    #     # probably want to write some utils function for this
    #     truncate = model.core.features.layer0.conv.kernel_size[0] # WARNING: because of the inconsistent naming of 1st vs. later layers in neuralpredictors this is only correct for 3dconv cores where only the 1st layer is actually 3d (after that, temporal length is 1)
    # else:
    #     truncate = config.get('prediction_latency')
    
    # score depends on truncation, device parameters

    optim = resolve_fn(config.get('optim'),'torch.optim')
    scheduler = make_scheduler(config.get('sched_type'),config.get('sched_config'))
    # stopper = make_stopper(model, scheduler)
            # objective: objective function that is used for early stopping. The function must accept single positional argument `model`
            # and return a single scalar quantity.

    trainer = TrainerClass(model, dataloaders, seed, loss_fn=loss, score_fn=score_fn,
                           optimizer = optim, lr_sched = scheduler, 
                           **config)
    
    # print(f'Starting Training of model {model} with dataloaders {dataloaders}')
    ex_loader = next(iter(next(iter(dataloaders.values())).values()))
    print(f'Dataloaders have {len(ex_loader._transform.composite_transform.transforms[-2].grab_channel)} neurons randomly selected')
    print(f'Selected neuron IDs: {ex_loader._transform.composite_transform.transforms[-2].grab_channel}')

    # out = trainer.train()

    return trainer


def trainer_fn(
    model: torch.nn.Module, 
    dataloaders: Dict, 
    seed: int, 
    uid: Dict, 
    cb: Callable, 
    **config
) -> Tuple[float, Any, Dict]:
    """
    Args:
        model: initialized model to train
        data_loaders: containing "train", "validation" and "test" data loaders
        seed: random seed
        uid: database keys that uniquely identify this trainer call
        cb: callback function to ping the database and potentially save the checkpoint
    Returns:
        score: performance score of the model
        output: user specified validation object based on the 'stop function'
        model_state: the full state_dict() of the trained model
    """

    # resolve loss, stopping, other score functions
    loss = partial(resolve_fn(config.get('loss_config').get('loss_fn'), 'dlgn_cnn.training.losses_scores'),
                    truncate = model.truncate, **(config.get('loss_config')) 
                   ) # loss does not depend on model...at least as we use it now
    # generally, score depends on some properties of model (namely, whether forward pass truncates temporal length), so partial call is left to Trainer constructor
    score_fn = partial( resolve_fn(config.get('score_config').get('score_fn'), 'dlgn_cnn.training.losses_scores'),
                       truncate = model.truncate, **config.get('score_config'))
    
    # for now - score is used for early stopping.  later we will add option for arbitrary add'l scores
    # score = partial( resolve_fn(config.get('score_fn'), 'dlgn_cnn.training.losses_scores'), )

    # if '3' in str(model.core.__class__):
    #     # TODO: generalize following line to cores with more than one 3D conv layer
    #     # probably want to write some utils function for this
    #     truncate = model.core.features.layer0.conv.kernel_size[0] # WARNING: because of the inconsistent naming of 1st vs. later layers in neuralpredictors this is only correct for 3dconv cores where only the 1st layer is actually 3d (after that, temporal length is 1)
    # else:
    #     truncate = config.get('prediction_latency')
    
    # score depends on truncation, device parameters

    optim = resolve_fn(config.get('optim'),'torch.optim')
    scheduler = make_scheduler(config.get('sched_type'),config.get('sched_config'))
    # stopper = make_stopper(model, scheduler)
            # objective: objective function that is used for early stopping. The function must accept single positional argument `model`
            # and return a single scalar quantity.

    trainer = TrainerClass(model, dataloaders, seed, loss_fn=loss, score_fn=score_fn,
                           optimizer = optim, lr_sched = scheduler, 
                           **config)
    
    print(f'Starting Training of model {model} with dataloaders {dataloaders}')
    ex_loader = next(iter(next(iter(dataloaders.values())).values()))
    print(f'Dataloaders have {len(ex_loader._transform.composite_transform.transforms[-2].grab_channel)} neurons randomly selected')
    print(f'Selected neuron IDs: {ex_loader._transform.composite_transform.transforms[-2].grab_channel}')

    out = trainer.train()

    return out


class TrainerClass:
    def __init__(
        self,
        model,
        dataloaders,
        seed: int,
        loss_fn,
        score_fn,
        lr_sched,
        optimizer,
        **config,
    ) -> None:
        # hard coding hyperparams that we won't change anytime soon
        self.samp_rate, self.in_name, self.out_name = 60, 'videos', 'responses'

        # trainer has model attribute, dataloader attribute, seed attribute
        self.model = model
        self.loaders = dataloaders
        self.seed = seed
        self.optim = optimizer
        self.scheduler = lr_sched
        
        # self.stopper = make_stopper(self.model, self.scheduler)
        self.device = config.get('device')

        self.stopping_config = config.get('stopping_config')
        self.sched_config = config.get('sched_config')

        # TODO: allow arbitrary number of score fns not used for early stopping

        if '3' in str(model.core.__class__):
            # TODO: generalize following line to cores with more than one 3D conv layer
            # probably want to write some utils function for this
            self.truncate = model.core.features.layer0.conv.kernel_size[0]-1 # WARNING: because of the inconsistent naming of 1st vs. later layers in neuralpredictors this is only correct for 3dconv cores where only the 1st layer is actually 3d (after that, temporal length is 1)
        else:
            self.truncate = config.get('prediction_latency')
        
        self.loss_fn = loss_fn # 2nd argument is just default base, in case loss_fn string does not include base module (so just make sure that it does)
        self.loss_kwargs = config.get('loss_config')
        self.score_fn = score_fn
        self.score_kwargs = config.get('score_config')
        # self.score_fn = partial(score_fn, device=self.device, truncate=self.truncate, **config.get('score_config'))

        # define optimizer, loss function, max epochs, and
        self.optimizer = optimizer(self.model.parameters(),config.get('lr_init'))
        self.max_epochs = config.get('max_epochs')
        self.optim_step_count = len(self.loaders['train'].keys()) if not config.get('loss_accum_batch_n') else config.get('loss_accum_batch_n') # the number of batches to accumulate gradients over before updating weights - if not specified, number of sessions, so that updates are always performed with respect to gradients for all sessions, rather than one at a time

        self.wandb_config = config.get('wandb_config')


    def batch_loss(self, batch_dict, loader_key, model_output = None, only_total = False, **loss_config):
        '''
        :param batch_dict: dict of data returned by dataloader
        :param loader_key: session ID of loader, used to index readout  
        :param model_output: output if already computed, for example in plotting function where result of forward pass is explicitly referenced outside of loss computation
        :param only_total: whether to return sum of data driven and regularization terms, or tuple of individual terms and sum
        '''
        # put batch on device if not already
        batch = {k: torch.Tensor(v.numpy()).to(self.device) for k, v in batch_dict.items() } #???
        model_output = self.model.to(self.device)(batch, loader_key) if model_output is None else model_output
        # compute loss between prediction and output
        data_loss = self.loss_fn(model_output, batch[self.out_name], **loss_config) # loss originating from model's failure to match data
        core_reg = sum(self.model.core.regularizer()) if isinstance(self.model.core.regularizer(), tuple) else (self.model.core.regularizer(),)
        # compute loss from regularization on parameters
        param_loss = core_reg + self.model.readout.regularizer(loader_key) # loss originating from regularization of model's parameters
        total_loss = param_loss + data_loss
        # ipdb.set_trace()
        return total_loss if only_total else (data_loss, param_loss, total_loss)

    def loader_loss(self, loader_key, tier='val'):
        '''
        :return data_loss: 
        :return param_loss:  
        '''

        data_loss, param_loss = 0, 0
        for batch in self.loaders[tier][loader_key]: # iterate through batches in specified tier and scan/session
            loss1, loss2, _ = self.batch_loss(batch,loader_key) # compute loss over each batch
            data_loss += loss1
            param_loss += loss2
        return data_loss, param_loss, data_loss + param_loss

    def tier_loss(self, tier, asdict=True):
        data_loss, param_loss = 0, 0
        for loader_key in self.loaders[tier].keys(): # iterate through separate scans/sessions
            loss1, loss2, _ = self.loader_loss(loader_key, tier) # compute loss over all batches for each
            data_loss += loss1
            param_loss += loss2
        return data_loss, param_loss, data_loss + param_loss

    def scalar_grand_score(self, model=None):
        model = self.model if not model else model # early_stopping requires model parameter, otherwise it would be removed
        # dummy method to act as objective argument to early_stopping
        return self.tier_score('val',asdict=False).mean().numpy() # 
        # important to cast to numpy!
        # early_stopping uses ~ np.isfinite( ) on output of stop_fn
        # for some reason, this turns 1 to 254 if passed a torch Tensor of type uint8

    def batch_score(self, batch_dict, loader_key):
        '''
        :param batch_dict: dict returned by dataloader
        :param loader_key: key of dataset, used to index readout
        :return score: score array computed from prediction and true response
        '''
        cuda_batch = {k: torch.Tensor(v.numpy()).to(self.device) for k, v in batch_dict.items() } #???
        model_output = self.model.to(self.device)(cuda_batch, loader_key)
        score = self.score_fn(model_output, cuda_batch[self.model.out_name] )
        return score

    def loader_score(self, loader_key, tier='val'):
        '''
        :param str loader_key: session ID used to index self.loaders dict  
        :param str tier: tier to select loader from (each loader_key will have split in train/val tiers)
        :return torch.Tensor loader_score_arr: (B,c) shape, score of each individual time series
        '''
        batch_scores = []
        for batch in self.loaders[tier][loader_key]: # iterate through batches in specified tier and scan/session
            batch_scores.append(self.batch_score(batch,loader_key)) # compute loss over each batch
        loader_score_arr = torch.cat(batch_scores,0) # (B total samples, c total neurons) in given loader
        # ipdb.set_trace()
        return loader_score_arr

    def tier_score(self, tier, asdict = True):
        '''
        :param str tier: tier of self.loaders dict from which all sessions are selected
        :param Boolean asdict: tier of self.loaders dict from which all sessions are selected
                
        :return loader_scores :
            asdict == True -> dict {k: (B,c)} k = session ID, B = total samples in dataset, c = |neurons| per set
            asdict == False -> torch.Tensor (B, c_tot), c_tot = sum_k c_k over loaders k  
        '''
        loader_scores = {}
        for key in self.loaders[tier].keys(): # iterate through separate scans/sessions
            loader_scores[key] = self.loader_score(key, tier) # compute loss over all batches for each
        if not asdict: # then concatenate scores across neuron axis of all sessions
            loader_scores = cat_tensor_dict(loader_scores) #dict [k] = (B,c_k,1), score arrays should agree on (0,2)
        return loader_scores

    def training_iter(self, batch: Dict, loader_key) -> Tuple[float, int]:
        # forward:
        self.optimizer.zero_grad()
        loss = self.batch_loss(batch,loader_key)
        # backward:
        loss.backward()
        self.optimizer.step()
        # keep track of accuracy:
        return loss
        
    def plot_outputs(self, batch, loader_key, neurons = None, scenes = None, plot_loss=False,tier='train'):
        output = self.model(batch, loader_key).detach().cpu() # (b,c,t)
        neurons = np.arange(output.shape[1]) if neurons is None else neurons # (1) or (2) ?
        scenes = np.arange(len(output)) if scenes is None else scenes # (1) or (2) ?
        # ipdb.set_trace()
        output = output[scenes][:,neurons.flatten().tolist(),::]

        fig, axs = plt.subplots(output.shape[0], output.shape[1], dpi=100, figsize=(output.shape[1]*4,output.shape[0]*2))
        t = np.arange(0,output.shape[2]) / self.samp_rate # 1 sec clips at 60Hz

        neuron_ids = self.loaders[tier][loader_key].dataset.info['neuron_ids']
        deeplake_index = self.loaders[tier][loader_key].dataset.index

        if plot_loss:
            loss = self.batch_loss(batch, loader_key, model_output=output, only_total=True, reduction=None)[scenes,neurons,::].detach().cpu()

        for i, scene in enumerate(scenes):
            # print(f'doing scene {scene}')
            for j, neuron in enumerate(neurons):
                # print(f'doing neuron {neuron}')
                ax = axs[i,j]
                ax.plot(t, output[i, j], label='model prediction') # Plot output
                ax.plot(t, batch[self.out_name][i, j, self.truncate:], label = 'ground truth')
                # ipdb.set_trace()
                if plot_loss:
                    ax.plot(t,loss[i,j], alpha = 0.4, label='instantaneous loss')

                if isinstance(neuron_ids[int(neuron)], Iterable): # dlgn
                    neuron_title = '_'.join( [ f'{a}={b}' for a,b in zip( ['mouse','sess','exp','unit'], neuron_ids[neuron]) ] )
                else:
                    neuron_title = neuron_ids[int(neuron)] # v1 only has int IDs
                    ax.set(title=f"Scene {''.join([x for x in str(deeplake_index.__getitem__(int(scene))) if x.isdigit()])}, Neuron {neuron_title}" )
                        #         ax.text(ax.get_xlim()[1]*0.7,ax.get_ylim()[1]*0.9,
                        #                 f'$r^2={100*val_corr_per_set[k][neur]:.02}%$',
                        #                 fontdict={'size':8})
                        # plt.show()
            ax.legend() # only one legend, on last plot
        fig.tight_layout()
        # plt.show()

        # if save_plots:
        #     # what to save... epoch number, neuron ID, dataset key, scene ID,
        #     plt_filename = f'model_predictions_ep{epoch}_dataset{os.path.split(k)[1]}.png' 
        #     plt.savefig(f'{checkpoint_save_path}{checkpoint_save_prefix}{plt_filename}')

        return fig


    def init_wandb(self):
        wandb.init(
            project=self.wandb_config.get('project'),
            entity=self.wandb_config.get('entity'),
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=self.wandb_config.get('name'),
            # Track hyperparameters and run metadata
            config={
                "learning_rate": self.sched_config.get('lr_init'),
                "architecture": self.wandb_config.get('wandb_model_config'),
                "dataset": self.wandb_config.get('wandb_dataset_config'),
                "max epochs": self.stopping_config.get('max_iter'),
                "starting epoch": self.stopping_config.get('start_pt'),
                "lr_decay_steps": self.sched_config.get('lr_decay_steps'),
                "lr_decay_factor": self.sched_config.get('lr_decay_factor'),
                "min_lr": self.sched_config.get('min_lr'),
            },
        )

    
    def log_wandb(log_dict):
    #     # log epochwise scalar metrics
    # compute metrics specified in self.wandb_dict
    # make specified plots
    # log to wandb
        pass
    

    def train(self, plotting=False) -> Tuple[float, Tuple[List[float], int], Dict]:
    # early_stopping creates a generator object over epochs, 
    # and calls to the specified objective function (stop_closure, in this case get_correlations) 
    # it also contains conditionals to decrease lr as according to scheduler and objective values
        self.init_wandb()
        for epoch, val_score in early_stopping(self.model, objective=self.scalar_grand_score, **self.stopping_config):
        # executes callback function if passed in keyword args
            # if cb is not None:
            #     cb()
            # train over batches
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0
            # batch_no_tot = 0
            for batch_no, (loader_key, batch_dict) in tqdm( # data_key is the same for every batch, indexing different loaders from different datasets.  data is batch as dict
                enumerate(LongCycler(self.loaders['train'])), #LongCycler iterates through batches and returns scan key each time
                total=len(LongCycler(self.loaders['train'])), # total should be number of total batches??
                desc="Epoch {}".format(epoch),
            ):
                # batch_no_tot += 1
    
                _, _, loss = self.batch_loss(batch_dict, loader_key)
                
                loss.backward()
    
                epoch_loss += loss.detach()
                if (batch_no + 1) % self.optim_step_count == 0:
                    self.optimizer.step()
                    # optimizer.zero_grad(set_to_none=False)
                    self.optimizer.zero_grad(set_to_none=True)

            # POST UPDATE
            self.model.eval()

            # TODO: implement checkpoint saving with checkpoint table
            # if save_checkpoints:
            #     if epoch % chpt_save_step == 0:
            #         torch.save(
            #             model.state_dict(), f"{checkpoint_save_path}{checkpoint_save_prefix}epoch_{epoch}.pth"
            #         )
    
            ## after - epoch-analysis
            
            # train_loss_data, train_loss_param, train_loss_total = detach_tuple( self.tier_loss('train') )
            val_loss_data, val_loss_param, val_loss_total = detach_tuple( self.tier_loss('val') )
            train_loss_data, train_loss_param, train_loss_total = detach_tuple( self.tier_loss('train') )

            # val_loss_reg = val_loss_reg.detach()
            # print(
            #     f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
            # )
            # print(f"EPOCH={epoch},  training loss = {train_loss_total}, mean train corr = { cat_tensor_dict(self.tier_score('train')).mean().numpy() }")
            print(f"EPOCH={epoch},  validation loss = {val_loss_total}, mean val corr = {self.scalar_grand_score()}")
    
            # Prepare logging dict
            val_score_keyed = detach_dict( self.tier_score('val') ) 
            train_score_keyed = detach_dict( self.tier_score('train') ) 
            if self.wandb_config:
                wandb_dict = {
                    'Epoch train loss': epoch_loss.cpu(),
                    'Scores by Session': {
                        'Val mean score': { sess: score.mean().cpu() for sess, score in val_score_keyed.items()},
                        'Val score by neuron': { sess: score.mean().cpu() for sess, score in val_score_keyed.items()}, # (1,)
                        # 'Train val score': { sess: score.mean(0).cpu() for sess, score in train_score_keyed.items()},
                        # 'Train score by neuron': { sess: score.mean(0).cpu() for sess, score in train_score_keyed.items()}, # (1,)
                    },
                    # 'Grand mean val score': self.scalar_grand_score(),
                    # 'Mean val score per neuron': self.tier_score(), 

                    # 'Validation score per neuron': val_score.mean(0), # (c,)

                    'Epoch total validation set loss': val_loss_total.cpu(),
                    'Epoch Poisson validation set loss': val_loss_data.cpu(),
                    'Epoch regularization validation set loss': val_loss_param.cpu(),
                    'Epoch total training set loss': val_loss_total.cpu(),
                    'Epoch Poisson training set loss': val_loss_data.cpu(),
                    'Epoch regularization training set loss': val_loss_param.cpu(),
                }

            ## PLOTTING ##
            if not plotting or epoch % 3 !=0:
                wandb.log(wandb_dict)
            else: # every 3rd epoch
                with torch.no_grad():
                    for loader_key, loader in self.loaders['train'].items():
                        for batch in loader:
                            fig = self.plot_outputs(batch, loader_key, plot_loss=True)
                            fig.show()
                            # only plots last batch ^^ for debugging with subset of data which is only one batch this is fine
                            wandb_dict[f'Results/Example Predictions'] = fig
                            wandb.log(wandb_dict)
    
                # ipdb.set_trace()
                    # batch_size = 4
                    # sampled_ro_loc = model.readout.sample_grid(batch_size,sample=False) # with sample=False, this will just return the mean
                    
                    # fig, axs = plt.subplots(len(scene_idx),len(neur_idx),dpi=100, figsize=(len(best_unit_idx)*4,len(scene_idx)*2))
                    # for j, neur in enumerate(neur_idx):
                        # ax.imshow(val_ds.info.strf[j,])
                        # plt.scatter(sampled_ro_loc[j,:])
                    # log dict
                    # ipdb.set_trace()

                    # print('finished logging')
                    # update_fig(fig,axs)
            
            self.model.train()
    
        ##### Model evaluation ####################################################################################################
        self.model.eval()
        # if save_checkpoints:
        #     torch.save(model.state_dict(), f"{checkpoint_save_path}{checkpoint_save_prefix}_final.pth")
    
        # # Compute avg validation and test correlation
        # val_corr, _  = get_correlations(
        #     model, dataloaders["val"], device=device, as_dict=False, per_neuron=True, from_deeplake=from_deeplake, latency=pred_latency
        # )
        val_score_final = self.scalar_grand_score()
        print(f"\n\n FINAL mean validation correlation {val_score_final.mean()} \n\n")
    
        output = {}
        output['validation correlation'] = val_score_final
        # output['validation_corr'] = val_corr
        # output['validation_corr_by_set'] = val_corr
    
        # score = np.mean(val_corr)
        if self.wandb_config:
            wandb.finish()
    
        # removing the checkpoints except the last one
        # to_clean = os.listdir(checkpoint_save_path)
        # for f2c in to_clean:
        #     if "epoch_" in f2c and f2c[-4:] == ".pth":
        #         os.remove(f"{checkpoint_save_path}{f2c}")
    
        
        return val_score_final, output, self.model.state_dict()
    
    
    def save(self, epoch: int, score: float) -> None:
        # method for saving state of training process
        state = {
            "action": "save",
            "score": score,
            "maximize_score": True,
            "tracker": self.accs, # scores/accuracies up to this point?
            "optimizer": self.optimizer,
            **self.chkpt_options,
        }
        self.trained_model_cb(
            uid=self.uid,
            epoch=epoch,
            model=self.model,
            state=state,
        )  # save model

    def restore(self) -> int:
        # method for loading checkpoint
        loaded_state = {
            "action": "last",
            "maximize_score": True,
            "tracker": self.accs, # scores/accuracies up to this point?
            "optimizer": self.optimizer.state_dict(),
        }
        self.trained_model_cb(
            uid=self.uid, epoch=-1, model=self.model, state=loaded_state
        )  # load the last epoch if existing
        epoch = loaded_state.get("epoch", -1) + 1
        return epoch