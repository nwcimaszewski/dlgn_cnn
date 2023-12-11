import torch

def detach_dict(dict):
    return { k: v.detach() for k, v in dict.items()}
    
def detach_tuple(tuple):
    return (x.detach() for x in tuple)

def cat_tensor_dict(dict, axis=1):
    '''
    Typically used to concat dict of scores across different loaders, so axis=1 is neuron dim (B,c_k,t)
    :param dict: {k: (B, c_k, 1)}
    :param axis: concat axis, for most cases 1
    :return torch.Tensor: (B, C, 1) [C = sum_k c_k]
    '''
    return torch.cat( list( dict.values() ), axis=axis )