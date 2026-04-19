from maenvs4vrp.learning.mardam.layers import MultiHeadAttention

import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, head_count, model_size, ff_size):
        super().__init__()
        self.mha = MultiHeadAttention(head_count, model_size)
        self.bn1 = nn.BatchNorm1d(model_size)
        self.ff1 = nn.Linear(model_size, ff_size)
        self.ff2 = nn.Linear(ff_size, model_size)
        self.bn2 = nn.BatchNorm1d(model_size)

    def forward(self, h_in, mask=None):
        r"""
        :param proj_in: :math:`N \times L \times D_M`
        :param mask:    :math:`N \times L`
        :return:        :math:`N \times L \times D_M`
        """
        att = self.mha(h_in, mask=mask)
        att = self.bn1((h_in + att).permute(0, 2, 1)).permute(0, 2, 1)

        h_out = F.relu(self.ff1(att))
        h_out = self.ff2(h_out)
        h_out = self.bn2((att + h_out).permute(0, 2, 1)).permute(0, 2, 1)

        if mask is not None:
            h_out[mask] = 0

        return h_out


class TransformerEncoder(nn.Module):
    r"""Neural Network module implementing a self-attention mechanism used as encoder.
    This layer structure was first introduced in "Attention Is All You Need" by \
            `[Vaswani et al. (2017)] <http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_
    """

    def __init__(self, layer_count, head_count, model_size, ff_size):
        super().__init__()
        for l in range(layer_count):
            self.add_module(
                str(l), TransformerEncoderLayer(head_count, model_size, ff_size)
            )

    def forward(self, inputs, mask=None):
        r"""
        :param inputs: :math:`N \times L \times D_M`
        :param mask:   :math:`N \times L`
        :return:       :math:`N \times L \times D_M`
        """
        h = inputs
        for child in self.children():
            h = child(h, mask)
        return h
