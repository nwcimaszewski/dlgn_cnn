import torch
import torch.nn.init as init
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.init import xavier_normal


# ---- general RNN core cell
class RNNCore:
    """
    RNN Core taken from: https://github.com/sinzlab/Sinz2018_NIPS/blob/master/nips2018/architectures/cores.py
    """

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            xavier_normal(m.weight.data)
            if m.bias is not None:
                init.constant(m.bias.data, 0.0)

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: not x.startswith("_") and "gamma" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


# ---------------- GRU core --------------------------------


class ConvGRUCellOpto(RNNCore, nn.Module):
    """
    Convolutional GRU cell taken from: https://github.com/sinzlab/Sinz2018_NIPS/blob/master/nips2018/architectures/cores.py
    """

    def __init__(self, input_channels, rec_channels, input_kern, rec_kern, gamma_rec=0, pad_input=True, **kwargs):
        super().__init__()

        input_padding = input_kern // 2 if pad_input else 0
        rec_padding = rec_kern // 2

        self.rec_channels = rec_channels
        self._shrinkage = 0 if pad_input else input_kern - 1

        self.gamma_rec = gamma_rec
        self.reset_gate_input = nn.Conv2d(input_channels, rec_channels, input_kern, padding=input_padding)
        self.reset_gate_hidden = nn.Conv2d(rec_channels, rec_channels, rec_kern, padding=rec_padding)

        self.update_gate_input = nn.Conv2d(input_channels, rec_channels, input_kern, padding=input_padding)
        self.update_gate_hidden = nn.Conv2d(rec_channels, rec_channels, rec_kern, padding=rec_padding)

        self.out_gate_input = nn.Conv2d(input_channels, rec_channels, input_kern, padding=input_padding)
        self.out_gate_hidden = nn.Conv2d(rec_channels, rec_channels, rec_kern, padding=rec_padding)

        self.apply(self.init_conv)
        self.register_parameter("_prev_state", None)

    def init_state(self, input_):
        batch_size, _, *spatial_size = input_.data.size()
        state_size = [batch_size, self.rec_channels] + [s - self._shrinkage for s in spatial_size]
        prev_state = torch.zeros(*state_size)
        if input_.is_cuda:
            prev_state = prev_state.cuda()
        prev_state = Parameter(prev_state)
        return prev_state

    def forward(self, input_, prev_state, opto):
        # get batch and spatial sizes

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_state(input_)

        update = self.update_gate_input(input_) + self.update_gate_hidden(prev_state)
        update = F.sigmoid(update)

        # I think the reset gate could be the most appropriate place to integrate optogenetic signal
        # np.full or torch.something (input_[:,0,::].shape, opto_signal)
        # the reset tensor/array/gate variable whatever,, is determined by one linear transform on the input and one on the previous state.
        # I think the silencing of feedback is more likely to affect the integration of information from previous time bins, 
        #   due to the inherent delay implicated by spikes propagating from thalamus to V1 back to thalamus
        # So we can compare: opto as channel in input, opto as channel in 
        # 
        # update vs. reset
        # update determines how much new state (output) is determined by prev_state 
        #   -> ( if update is 0, the state is identical to prev_state, unchanged )
        #   -> update computed as sigmoid of linear transform of input and prev_state
        # reset acts as a mask?  Or linear transform on prev_state before input into formula for instantaneous, internal hidden state h_t
        #   -> 
        # at the end of the day, the output of a 2d core is input to this, then this is shifted and input to the readout...

        # 1. add opto as channel to input_, or add it as channel to prev_state
        # 2. add opto as channel for one/both of these^, but only for reset
        # 3. same as 2., but only for update
        # 4. add opto as channel to inputs for h_t equation

        # self.opto_reset_input, self.opto_reset_prev_state,
        # self.opto_update_input, self.opto_update_prev_state,
        # self.opto_ht_input, self.opto_ht_prev_state
        # so 6 Booleans we can compare significance of...2^6 possibilities

        
        reset = self.reset_gate_input(input_) + self.reset_gate_hidden(prev_state) # reset determined by conv. transform of input and prev_state
        reset = F.sigmoid(reset)

        out = self.out_gate_input(input_) + self.out_gate_hidden(prev_state * reset) # contribution of prev_state to out gated by reset
        h_t = F.tanh(out)
        new_state = prev_state * (1 - update) + h_t * update

        return new_state

    def regularizer(self):
        return self.gamma_rec * self.bias_l1()

    def bias_l1(self):
        return (
            self.reset_gate_hidden.bias.abs().mean() / 3
            + self.update_gate_hidden.weight.abs().mean() / 3
            + self.out_gate_hidden.bias.abs().mean() / 3
        )


class GRU_Module(nn.Module):
    def __init__(self, input_channels, rec_channels, input_kern, rec_kern, gamma_rec=0, pad_input=True, **kwargs):
        """
        A GRU module for video data to add between the core and the readout.
        Recieves as input the output of a 3Dcore. Expected dimentions:
            -(Batch,Channels,Frames,Height, Widht) or (Channels,Frames,Height, Widht)
        The input is fed sequentially to a convolutional GRU cell, based on the frames chanell. The output has the same dimentions as the input.
        """
        super().__init__()
        self.gru = ConvGRUCell(
            input_channels, rec_channels, input_kern, rec_kern, gamma_rec=gamma_rec, pad_input=pad_input
        )

    def forward(self, x):
        """
        Forward pass definition based on https://github.com/sinzlab/Sinz2018_NIPS/blob/3a99f7a6985ae8dec17a5f2c54f550c2cbf74263/nips2018/architectures/cores.py#L556
        Modified to also accept 4 dimentional inputs (assuming no batch dimention is provided).
        """
        if len(x.shape) not in [4, 5]:
            raise RuntimeError(
                f"Expected 4D (unbatched) or 5D (batched) input to ConvGRUCell, but got input of size: {x.shape}"
            )

        batch = True
        if len(x.shape) == 4:
            batch = False
            x = torch.unsqueeze(x, dim=0)

        states = []
        hidden = None
        frame_pos = 2

        for frame in range(x.shape[frame_pos]):
            slice_channel = [frame if frame_pos == i else slice(None) for i in range(len(x.shape))]
            hidden = self.gru(x[slice_channel], hidden)
            states.append(hidden)
        out = torch.stack(states, frame_pos)
        if not batch:
            out = torch.squeeze(out, dim=0)
        return out
