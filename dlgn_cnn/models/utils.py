import torch


def add_opto_channel(tensor, c_axis, opto):
    '''
    function to broadcast optogenetic signal to the shape one channel in a tensor representation, then append this to the representation itself
    tensor: representation to be appended to, most likely of shape (b, c, t, ...)
    opto: binary optogenetic signal, of shape (t,) or maybe (t,1)
    '''
    channel_shape = tensor.shape[:c_axis,c_axis+1:] # shape of tensor along all axes except channel = size of one channel "slice"
    opto_channel = torch.ones(channel_shape) * opto # i hope this will just figure out the broadcasting nicely by itself?

    return torch.cat((tensor,opto_channel),axis=c_axis)