
from torch import Tensor

def get_distance(x: Tensor, y: Tensor):
    
    """
    Euclidean distance between two tensors of shape `[..., n, dim].
    Taken from: https://github.com/ai4co/rl4co/blob/main/rl4co/utils/ops.py

    Args:
        x(torch.Tensor): Point x.
        y(torch.Tensor): Point y.

    Returns:
        torch.Tensor: Distance between x and y.
    """
    return (x - y).norm(p=2, dim=-1)


def gather_by_index(src, idx, dim=1, squeeze=True):
    """
    https://github.com/ai4co/rl4co
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)