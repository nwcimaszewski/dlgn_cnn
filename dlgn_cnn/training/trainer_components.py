from torch.optim.lr_scheduler import ReduceLROnPlateau


def make_scheduler(type, optimizer, **config):
    '''Construct scheduler object.  Schedulers are out the box from Pytorch, so do not require dataloaders or model as input, only one config dict''' 
    if type == 'Plateau':
        return ReduceLROnPlateau(optimizer,**config)


# def make_stopper(model, scheduler, objective, **config):
#     '''Construct early stopping object.  These are imported from neuralpredictors.training.early_stopping, see docs there'''
#     # stop_fn = partial()
#     return early_stopping(model, objective=objective, scheduler=scheduler,
#                           interval=config.get('interval'),
#                           patience=config.get('patience'), 
#                           start=config.get('start_pt'), 
#                           max_iter=config.get('max_iter'),
#                           maximize = config.get('maximize'),
#                           tolerance= config.get('tolerance'),
#                           restore_best=config.get('restore_best'),
#                           lr_decay_steps=config.get('lr_decay_steps'))

# class EarlyStopper():
#     def __init__(self, model, stop_fn, **config):
#         self.model = model
#         self.stop_fn = partial(stop_fn, **config)
#     def __call__(self, ):
#         return early_stopping(model, objective=self.stop_fn, **config)
