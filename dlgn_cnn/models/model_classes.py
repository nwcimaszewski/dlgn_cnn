from torch import nn
import torch
from .utils import *

'''
Is this file redundant with neuralpredictors.layers.encoders files?  Maybe but I want to do one myself and don't like their formatting
At least one objective novelty is the presence of a binary optogenetic signal, an intervention on the system recorded.
This, I would consider a qualitatively different type of data from others recorded.  So, 
'''


class BaseResponseModel(nn.Module):
    def __init__(
        self,
        core,
        readout,
        gru = None,
        shifter = None,
        modulator = None, #{'nonlinearity_type':'elu','elu_offset':0.0}
        final_nonlin = None,
        device = 'cuda',
        in_name = 'videos',
        out_name = 'responses',
        pupil_name = None, # 'pupil_center'
        behavior_name = None, # 'behavior'
        **kwargs
    ):
        super().__init__()
        # keys of batch_dict from dataloader used for core input, (output for computing performance metrics), shifter input, modulator input
        self.in_name, self.out_name, self.pupil_name, self.behav_name = in_name, out_name, pupil_name, behavior_name
        self.device = device
        # mandatory components
        self.core = core
        self.readout = readout
        self.twoDcore = '3' in str(core) # flag for reshape hack to fit video tensor into either 2d or 3d cores
        # optional components
        self.gru = gru
        self.shifter = shifter
        self.modulator = modulator
        self.final_nonlin = final_nonlin        

        if '3' in str(self.core.__class__):
            # TODO: generalize following line to cores with more than one 3D conv layer
            # probably want to write some utils function for this
            self.truncate = self.core.features.layer0.conv.kernel_size[0]-1 # WARNING: because of the inconsistent naming of 1st vs. later layers in neuralpredictors this is only correct for 3dconv cores where only the 1st layer is actually 3d (after that, temporal length is 1)
        else:
            self.truncate = 0

    # TODO: write reshape method to make sure same video tensor can be passed into 2d and 3d cores
    # def reshape_core_input(self, input):
    #     in_shp = input.shape
    #     # if 2d core
    #     if len(in_shp) == 5:
    #         # return input.reshape(len(input), in_shp[1]*in_shp[2], in_shp[3:])
    #     else:
    #         return input

    # TODO: write method to compute temporal-shortening of core (so basically sum up temporal filter lengths - 1 for each)
    # def truncation_length(self):
    #     T = 0
    #     for layer in self.core:
    #         T += layer.
    #     return T

    def forward(self, data_batch, detach_core = False):
        input, behavior = data_batch.get(self.in_name), data_batch.get(self.behav_name) # assumes in_name is just one string

        X = input.to(self.device)
        b, _, t, _, _ = X.shape
        X = self.core(X).detach() if detach_core else self.core(X) # for 2d core X must be reshaped!
        t_ = X.shape[2] # length after truncation (if core 3d conv) could also set this as model property (probably should)

        X = self.gru(X) if self.gru else X
        
        # necessary to collapse vid to 4d to pass to readout, treating each frame as independent set of features
        X = self.readout(vid_5dto4d(X), shift = self.compute_shift(data_batch))

        X = self.modulator(X, behavior = behavior.to(self.device)) if self.modulator else X
        X = self.final_nonlin(X) if self.final_nonlin else X

        return resp_2dto3d(X,b,t_)

    def compute_shift(self, data_batch):
        pupil_loc = data_batch.get(self.pupil_name)
        return self.shifter(pupil_loc.to(self.device)) if self.shifter else None # general to models without shifter specified
        
        


class ResponseModelOpto(BaseResponseModel):

    def __init__(
        self,
        core,
        readout,
        gru = None,
        shifter = None,
        modulator = None, #{'nonlinearity_type':'elu','elu_offset':0.0}
        final_nonlin = None,
        device = 'cuda',
        in_name = 'videos',
        out_name = 'responses',
        # *args,
        **kwargs
    ):
        super().__init__()

        

    def forward(self, data_batch):
        input, output = data_batch.get(self.in_name), data_batch.get(self.in_name)
        