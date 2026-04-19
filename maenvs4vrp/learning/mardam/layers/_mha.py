import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_prod_attention(queries, keys, values, mask=None):
    r"""
    :param queries: :math:`... \times l_q  \times d_k`
    :param keys:    :math:`... \times l_kv \times d_k`
    :param values:  :math:`... \times l_kv \times d_v`
    :param mask:    :math:`... \times l_q  \times l_kv`
    :type queries:  torch.Tensor(dtype=torch.float32)
    :type keys:     torch.Tensor(dtype=torch.float32)
    :type values:   torch.Tensor(dtype=torch.float32)
    :type mask:     torch.Tensor(dtype=torch.uint8)
    :return:        :math:`... \times l_q  \times d_v`
    :rtype:         torch.Tensor(dtype=torch.float32)
    """
    weights = queries.matmul(keys.transpose(-1, -2))  # ... x l_q x l_kv
    weights *= keys.size(-1) ** -0.5
    if mask is not None:
        if mask.dim() == weights.dim() - 1:
            mask = mask.unsqueeze(-2).expand_as(weights)
        weights[mask] = -float("inf")
    weights = F.softmax(weights, dim=-1)
    return weights.matmul(values)


class _MHA_V1(nn.Module):
    def __init__(
        self,
        head_count,
        query_size,
        key_size=None,
        value_size=None,
        key_size_per_head=None,
        value_size_per_head=None,
    ):
        super().__init__()
        self.head_count = head_count

        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size

        self.key_size_per_head = (
            self.key_size // self.head_count
            if key_size_per_head is None
            else key_size_per_head
        )
        self.value_size_per_head = (
            self.value_size // self.head_count
            if value_size_per_head is None
            else value_size_per_head
        )

        self.query_project = nn.Linear(
            self.query_size, self.head_count * self.key_size_per_head, bias=False
        )
        self.key_project = nn.Linear(
            self.key_size, self.head_count * self.key_size_per_head, bias=False
        )
        self.value_project = nn.Linear(
            self.value_size, self.head_count * self.value_size_per_head, bias=False
        )
        self.recombine = nn.Linear(
            self.head_count * self.value_size_per_head, self.value_size, bias=False
        )

    def forward(self, queries, keys, values, mask=None):
        r"""
        :param queries: :math:`... \times l_q  \times d_q`
        :param keys:    :math:`... \times l_kv \times d_k`
        :param values:  :math:`... \times l_kv \times d_v`
        :param mask:    :math:`... \times l_q  \times l_kv`
        :type queries:  torch.Tensor(dtype=torch.float32)
        :type keys:     torch.Tensor(dtype=torch.float32)
        :type values:   torch.Tensor(dtype=torch.float32)
        :type mask:     torch.Tensor(dtype=torch.uint8)
        :return:        :math:`... \times l_q  \times d_v`
        :rtype:         torch.Tensor(dtype=torch.float32)
        """
        q_proj = self.query_project(queries).chunk(self.head_count, dim=-1)
        k_proj = self.key_project(keys).chunk(self.head_count, dim=-1)
        v_proj = self.value_project(values).chunk(self.head_count, dim=-1)

        att_applied = tuple(
            map(
                scaled_dot_prod_attention,
                q_proj,
                k_proj,
                v_proj,
                (mask for _ in range(self.head_count)),
            )
        )

        return self.recombine(torch.cat(att_applied, dim=-1))


class _MHA_V2(nn.Module):
    def __init__(
        self,
        head_count,
        query_size,
        key_size=None,
        value_size=None,
        key_size_per_head=None,
        value_size_per_head=None,
    ):
        super().__init__()
        self.head_count = head_count

        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size

        self.key_size_per_head = (
            self.key_size // self.head_count
            if key_size_per_head is None
            else key_size_per_head
        )
        self.value_size_per_head = (
            self.value_size // self.head_count
            if value_size_per_head is None
            else value_size_per_head
        )

        self._inv_sqrt_d = self.key_size_per_head**-0.5

        self.query_project = nn.Linear(
            self.query_size, self.head_count * self.key_size_per_head, bias=False
        )
        self.key_project = nn.Linear(
            self.key_size, self.head_count * self.key_size_per_head, bias=False
        )
        self.value_project = nn.Linear(
            self.value_size, self.head_count * self.value_size_per_head, bias=False
        )
        self.recombine = nn.Linear(
            self.head_count * self.value_size_per_head, self.value_size, bias=False
        )

        self._k_proj = None
        self._v_proj = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.uniform_(self.query_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        nn.init.uniform_(self.key_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        inv_sq_dv = self.value_size_per_head**-0.5
        nn.init.uniform_(self.value_project.weight, -inv_sq_dv, inv_sq_dv)

    def precompute(self, keys, values=None):
        values = keys if values is None else values
        l_kv = keys.size(-2)
        self._k_proj = (
            self.key_project(keys)
            .view(-1, l_kv, self.head_count, self.key_size_per_head)
            .permute(0, 2, 3, 1)
        )
        self._v_proj = (
            self.value_project(values)
            .view(-1, l_kv, self.head_count, self.value_size_per_head)
            .permute(0, 2, 1, 3)
        )

    def forward(self, queries, keys=None, values=None, mask=None):
        *size, l_q, _ = queries.size()

        q_proj = (
            self.query_project(queries)
            .view(-1, l_q, self.head_count, self.key_size_per_head)
            .permute(0, 2, 1, 3)
        )

        if keys is None:
            if self._k_proj is None:  # self-attention
                l_kv = l_q
                k_proj = (
                    self.key_project(queries)
                    .view(-1, l_kv, self.head_count, self.key_size_per_head)
                    .permute(0, 2, 3, 1)
                )
            else:  # pre-computed
                l_kv = self._k_proj.size(-1)
                k_proj = self._k_proj
        else:
            l_kv = keys.size(-2)
            k_proj = (
                self.key_project(keys)
                .view(-1, l_kv, self.head_count, self.key_size_per_head)
                .permute(0, 2, 3, 1)
            )

        if values is None:
            if self._v_proj is None:  # self-attention
                v_proj = (
                    self.value_project(queries)
                    .view(-1, l_kv, self.head_count, self.value_size_per_head)
                    .permute(0, 2, 1, 3)
                )
            else:  # pre-computed
                v_proj = self._v_proj
        else:
            v_proj = (
                self.value_project(values)
                .view(-1, l_kv, self.head_count, self.value_size_per_head)
                .permute(0, 2, 1, 3)
            )

        weights = q_proj.matmul(k_proj)
        weights *= self._inv_sqrt_d
        if mask is not None:
            if mask.numel() * self.head_count == weights.numel():  # one mask per query
                m = mask.view(-1, 1, l_q, l_kv).expand_as(weights)
            else:  # common mask for all queries
                m = mask.view(-1, 1, 1, l_kv).expand_as(weights)
            weights[m] = -float("inf")

        weights = F.softmax(weights, dim=-1)

        att_applied = (
            weights.matmul(v_proj)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(*size, l_q, self.head_count * self.value_size_per_head)
        )
        return self.recombine(att_applied)
