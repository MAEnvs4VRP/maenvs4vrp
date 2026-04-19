import torch
import torch.nn as nn


import math

"""
Adapted from: https://github.com/cpwan/RLOR/tree/main

"""


class AttentionScore(nn.Module):
    r"""
    A helper class for attention operations.
    There are no parameters in this module.
    This module computes the alignment score with mask
    and return only the attention score.

    The default operation is

    .. math::
         \pmb{u} = \mathrm{Attention}(q,\pmb{k}, \mathrm{mask})

    where for each key :math:`k_j`, we have

    .. math::
        u_j =
        \begin{cases}
             &\frac{q^Tk_j}{\sqrt{\smash{d_q}}} & \text{ if } j \notin \mathrm{mask}\\
             &-\infty & \text{ otherwise. }
        \end{cases}

    If ``use_tanh`` is ``True``, apply clipping on the logits :math:`u_j` before masking:

    .. math::
        u_j =
        \begin{cases}
             &C\mathrm{tanh}\left(\frac{q^Tk_j}{\sqrt{\smash{d_q}}}\right) & \text{ if } j \notin \mathrm{mask}\\
             &-\infty & \text{ otherwise. }
        \end{cases}

    Args:
        use_tanh: if True, use clipping on the logits
        C: the range of the clipping [-C,C]
    Inputs: query, keys, mask
        * **query** : [..., 1, h_dim]
        * **keys**: [..., graph_size, h_dim]
        * **mask**: [..., graph_size] ``logits[...,j]==-inf`` if ``mask[...,j]==True``.
    Outputs: logits
        * **logits**: [..., 1, graph_size] The attention score for each key.
    """

    def __init__(self, use_tanh=False, C=10):
        super(AttentionScore, self).__init__()
        self.use_tanh = use_tanh
        self.C = C

    def forward(self, query, key, mask=None):
        u = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if self.use_tanh:
            logits = torch.tanh(u) * self.C
        else:
            logits = u
        if mask is not None:
            # logits[mask.expand_as(logits)] = float("-inf")  # masked after clipping
            logits = logits.masked_fill(
                mask.unsqueeze(1).expand_as(logits) == False, float("-inf")
            )

        return logits


class MultiHeadAttention(nn.Module):
    r"""
    Compute the multi-head attention.

    .. math::
        q^\prime = \mathrm{MultiHeadAttention}(q,\pmb{k},\pmb{v},\mathrm{mask})

    The following is computed:

    .. math::
        \begin{aligned}
        \pmb{a}^{(j)} &= \mathrm{Softmax}(\mathrm{AttentionScore}(q^{(j)},\pmb{k}^{(j)}, \mathrm{mask}))\\
        h^{(j)} &= \sum\nolimits_i \pmb{a}^{(j)}_i\pmb{v}_i \\
        q^\prime &= W^O \left[h^{(1)},...,h^{(J)}\right]
        \end{aligned}

    Args:
        embedding_dim: dimension of the query, keys, values
        n_head: number of heads
    Inputs: query, keys, value, mask
        * **query** : [batch, n_querys, embedding_dim]
        * **keys**: [batch, n_keys, embedding_dim]
        * **value**: [batch, n_keys, embedding_dim]
        * **mask**: [batch, 1, n_keys] ``logits[batch,j]==-inf`` if ``mask[batch, 0, j]==True``
    Outputs: logits, out
        * **out**: [batch, 1, embedding_dim] The output of the multi-head attention
    """

    def __init__(self, embed_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attentionScore = AttentionScore()
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, mask):
        query_heads = self._make_heads(query)
        key_heads = self._make_heads(key)
        value_heads = self._make_heads(value)

        # [n_heads, batch, 1, nkeys]
        compatibility = self.attentionScore(query_heads, key_heads, mask)

        # [n_heads, batch, 1, head_dim]
        out_heads = torch.matmul(torch.softmax(compatibility, dim=-1), value_heads)

        # from multihead [nhead, batch, 1, head_dim] -> [batch, 1, nhead* head_dim]
        out = self.project_out(self._unmake_heads(out_heads))
        return out

    def _make_heads(self, v):
        batch_size, nkeys, h_dim = v.shape
        #  [batch_size, ..., n_heads* head_dim] --> [n_heads, batch_size, ..., head_dim]
        out = v.reshape(batch_size, nkeys, self.n_heads, h_dim // self.n_heads).movedim(
            -2, 0
        )
        return out

    def _unmake_heads(self, v):
        #  [n_heads, batch_size, ..., head_dim] --> [batch_size, ..., n_heads* head_dim]
        out = v.movedim(0, -2).flatten(-2)
        return out


class MultiHeadAttentionProj(nn.Module):
    r"""
    Compute the multi-head attention with projection.
    Different from :class:`.MultiHeadAttention` which accepts precomputed query, keys, and values,
    this module computes linear projections from the inputs to query, keys, and values.

    .. math::
        q^\prime = \mathrm{MultiHeadAttentionProj}(q_0,\pmb{h},\mathrm{mask})

    The following is computed:

    .. math::
        \begin{aligned}
        q, \pmb{k}, \pmb{v} &= W^Qq_0, W^K\pmb{h}, W^V\pmb{h}\\
        \pmb{a}^{(j)} &= \mathrm{Softmax}(\mathrm{AttentionScore}(q^{(j)},\pmb{k}^{(j)}, \mathrm{mask}))\\
        h^{(j)} &= \sum\nolimits_i \pmb{a}^{(j)}_i\pmb{v}_i \\
        q^\prime &= W^O \left[h^{(1)},...,h^{(J)}\right]
        \end{aligned}

    if :math:`\pmb{h}` is not given. This module will compute the self attention of :math:`q_0`.

    .. warning::
        The results of the in-projection of query, key, value are
        slightly different (order of ``1e-6``) with the original implementation.
        This is due to the numerical accuracy.
        The two implementations differ by the way of multiplying matrix.
        Thus, different internal implementation libraries of pytorch are called
        and the results are slightly different.
        See the pytorch docs on `numerical accruacy <https://pytorch.org/docs/stable/notes/numerical_accuracy.html>`_ for detail.

    Args:
        embedding_dim: dimension of the query, keys, values
        n_head: number of heads
    Inputs: q, h, mask
        * **q** : [batch, n_querys, embedding_dim]
        * **h**: [batch, n_keys, embedding_dim]
        * **mask**: [batch, n_keys] ``logits[batch,j]==-inf`` if ``mask[batch,j]==True``
    Outputs: out
        * **out**: [batch, n_querys, embedding_dim] The output of the multi-head attention


    """

    def __init__(self, embed_dim, n_heads=8):
        super(MultiHeadAttentionProj, self).__init__()

        self.queryEncoder = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keyEncoder = nn.Linear(embed_dim, embed_dim, bias=False)
        self.valueEncoder = nn.Linear(embed_dim, embed_dim, bias=False)

        self.MHA = MultiHeadAttention(embed_dim, n_heads)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        query = self.queryEncoder(q)
        key = self.keyEncoder(h)
        value = self.valueEncoder(h)

        out = self.MHA(query, key, value, mask)

        return out


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class GatingMechanism(nn.Module):
    def __init__(self, module, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.module = module

        self.Wr = nn.Linear(d_input, d_input)
        self.Ur = nn.Linear(d_input, d_input)
        self.Wz = nn.Linear(d_input, d_input)
        self.Uz = nn.Linear(d_input, d_input)
        self.Wg = nn.Linear(d_input, d_input)
        self.Ug = nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input):
        y = self.module(input)

        r = self.sigmoid(self.Wr(y) + self.Ur(input))
        z = self.sigmoid(self.Wz(y) + self.Uz(input) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, input)))
        g = torch.mul(1 - z, input) + torch.mul(z, h)
        return g


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()

        self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)

    def forward(self, input):
        #         out = self.normalizer(input.permute(0,2,1)).permute(0,2,1) # slightly different 3e-6
        #         return out
        return self.normalizer(input.view(-1, input.size(-1))).view(input.size())


class GatedMultiHeadAttentionLayer(nn.Sequential):
    r"""
    A layer with attention mechanism and normalization.

    For an embedding :math:`\pmb{x}`,

    .. math::
        \pmb{h} = \mathrm{MultiHeadAttentionLayer}(\pmb{x})

    The following is executed:

    .. math::
        \begin{aligned}
        \pmb{x}_0&=\pmb{x}+\mathrm{MultiHeadAttentionProj}(\pmb{x})  \\
        \pmb{x}_1&=\mathrm{BatchNorm}(\pmb{x}_0)                      \\
        \pmb{x}_2&=\pmb{x}_1+\mathrm{MLP_{\text{2 layers}}}(\pmb{x}_1)\\
        \pmb{h} &=\mathrm{BatchNorm}(\pmb{x}_2)
        \end{aligned}

    .. seealso::
        The :math:`\mathrm{MultiHeadAttentionProj}` computes the self attention
        of the embedding  :math:`\pmb{x}`. Check :class:`~.MultiHeadAttentionProj` for details.

    Args:
        n_heads : number of heads
        embedding_dim : dimension of the query, keys, values
        feed_forward_hidden : size of the hidden layer in the MLP
    Inputs: inputs
        * **inputs**: embeddin :math:`\pmb{x}`. [batch, graph_size, embedding_dim]
    Outputs: out
        * **out**: the output :math:`\pmb{h}` [batch, graph_size, embedding_dim]
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden=512,
    ):
        super(GatedMultiHeadAttentionLayer, self).__init__(
            GatingMechanism(
                MultiHeadAttentionProj(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                ),
                embed_dim,
            ),
            Normalization(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim),
        )


class MultiHeadAttentionLayer(nn.Sequential):
    r"""
    A layer with attention mechanism and normalization.

    For an embedding :math:`\pmb{x}`,

    .. math::
        \pmb{h} = \mathrm{MultiHeadAttentionLayer}(\pmb{x})

    The following is executed:

    .. math::
        \begin{aligned}
        \pmb{x}_0&=\pmb{x}+\mathrm{MultiHeadAttentionProj}(\pmb{x})  \\
        \pmb{x}_1&=\mathrm{BatchNorm}(\pmb{x}_0)                      \\
        \pmb{x}_2&=\pmb{x}_1+\mathrm{MLP_{\text{2 layers}}}(\pmb{x}_1)\\
        \pmb{h} &=\mathrm{BatchNorm}(\pmb{x}_2)
        \end{aligned}

    .. seealso::
        The :math:`\mathrm{MultiHeadAttentionProj}` computes the self attention
        of the embedding  :math:`\pmb{x}`. Check :class:`~.MultiHeadAttentionProj` for details.

    Args:
        n_heads : number of heads
        embedding_dim : dimension of the query, keys, values
        feed_forward_hidden : size of the hidden layer in the MLP
    Inputs: inputs
        * **inputs**: embeddin :math:`\pmb{x}`. [batch, graph_size, embedding_dim]
    Outputs: out
        * **out**: the output :math:`\pmb{h}` [batch, graph_size, embedding_dim]
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden=512,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttentionProj(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                )
            ),
            Normalization(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim),
        )


class GraphAttentionEncoder(nn.Module):
    r"""
    Graph attention by self attention on graph nodes.

    For an embedding :math:`\pmb{x}`, repeat ``n_layers`` time:

    .. math::
        \pmb{h} = \mathrm{MultiHeadAttentionLayer}(\pmb{x})

    .. seealso::
        Check :class:`~.MultiHeadAttentionLayer` for details.

    Args:
        n_heads : number of heads
        embedding_dim : dimension of the query, keys, values
        n_layers : number of :class:`~.MultiHeadAttentionLayer` to iterate.
        feed_forward_hidden : size of the hidden layer in the MLP
    Inputs: x
        * **x**: embeddin :math:`\pmb{x}`. [batch, graph_size, embedding_dim]
    Outputs: (h, h_mean)
        * **h**: the output :math:`\pmb{h}` [batch, graph_size, embedding_dim]
    """

    def __init__(self, n_heads, embed_dim, n_layers, feed_forward_hidden=512):
        super(GraphAttentionEncoder, self).__init__()

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
                for _ in range(n_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "TODO mask not yet supported!"

        h = self.layers(x)

        return (h, h.mean(dim=1))


class DynamicEmbedding(nn.Module):
    """Dynamic embedding"""

    def __init__(self, nodes_dyn_dim, embed_dim, linear_bias=False):
        super(DynamicEmbedding, self).__init__()
        self.projection = nn.Linear(nodes_dyn_dim, 3 * embed_dim, bias=linear_bias)

    def forward(self, dyn_node_obs):
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            dyn_node_obs
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class PolicyNet(nn.Module):
    def __init__(
        self,
        nodes_stat_obs_dim,
        nodes_dyn_obs_dim,
        agent_obs_dim,
        agents_obs_dim,
        global_obs_dim,
        embed_dim,
    ):
        super(PolicyNet, self).__init__()

        self.nodes_embedding = nn.Linear(nodes_stat_obs_dim, embed_dim, bias=False)

        self.nodes_dyn_embedding = DynamicEmbedding(nodes_dyn_obs_dim, embed_dim)
        self.agent_embedding = nn.Linear(
            agent_obs_dim + global_obs_dim, embed_dim, bias=False
        )

        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        self.nodes_encoder = GraphAttentionEncoder(
            n_heads=8,
            embed_dim=embed_dim,
            n_layers=3,
        )

        self.nodes_glimpse = MultiHeadAttention(embed_dim=embed_dim, n_heads=8)
        self.agent_glimpse = MultiHeadAttentionProj(embed_dim=embed_dim, n_heads=8)

        self.pointer = AttentionScore(use_tanh=True, C=10)

        self.active_agents_embedding = nn.Linear(agents_obs_dim, embed_dim, bias=False)

        self._initialize_parameters()
        self.cache = None

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def make_cache_(self, nodes_obs):
        nodes_emb = self.nodes_embedding(nodes_obs)
        nodes_encoded, _ = self.nodes_encoder(nodes_emb)
        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(
            nodes_encoded
        ).chunk(3, dim=-1)

        self.cached_embed = (glimpse_key, glimpse_val, logit_key)

    def forward(
        self, nodes_dyn_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
    ):
        agent_global_obs = torch.concat((self_obs, global_obs), dim=-1)

        agent_embed = self.agent_embedding(agent_global_obs)

        agents_embed = self.active_agents_embedding(agents_obs)

        glimpse_K, glimpse_V, logit_K = self.cached_embed

        query = agent_embed.unsqueeze(1)
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = (
            self.nodes_dyn_embedding(nodes_dyn_obs)
        )
        glimpse_K = glimpse_K + glimpse_key_dynamic
        glimpse_V = glimpse_V + glimpse_val_dynamic
        logit_K = logit_K + logit_key_dynamic

        nodes_glimpse = self.nodes_glimpse(query, glimpse_K, glimpse_V, action_mask)
        agent_glimpse = self.agent_glimpse(query, agents_embed, agents_mask)

        glimpse = nodes_glimpse + agent_glimpse
        logits = self.pointer(glimpse, logit_K, action_mask).squeeze(1)
        return logits, glimpse.squeeze(1)

    def get_action(
        self,
        nodes_obs,
        self_obs,
        agents_obs,
        agents_mask,
        global_obs,
        action_mask=None,
        deterministic=False,
    ):
        action_logits, glimpse = self.forward(
            nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
        )
        probs = torch.distributions.Categorical(logits=action_logits)
        if deterministic:
            return probs.mode
        else:
            return probs.sample()

    def get_action_and_logs(
        self,
        nodes_obs,
        self_obs,
        agents_obs,
        agents_mask,
        global_obs,
        action_mask=None,
        action=None,
        deterministic=False,
    ):
        action_logits, glimpse = self.forward(
            nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
        )
        probs = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            if deterministic:
                action = probs.mode
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()


class CriticNet(nn.Module):
    def __init__(self, hidden_dim):
        super(CriticNet, self).__init__()

        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, agent_state):
        return self.critic_net(agent_state)


class Learner(nn.Module):
    def __init__(
        self,
        nodes_stat_obs_dim,
        nodes_dyn_obs_dim,
        agent_obs_dim,
        agents_obs_dim,
        global_obs_dim,
        embed_dim=128,
    ):
        super(Learner, self).__init__()

        self.policy = PolicyNet(
            nodes_stat_obs_dim,
            nodes_dyn_obs_dim,
            agent_obs_dim,
            agents_obs_dim,
            global_obs_dim,
            embed_dim,
        )
        self.critic_net = CriticNet(embed_dim)

    def get_action_and_logs(
        self,
        nodes_obs,
        self_obs,
        agents_obs,
        agents_mask,
        global_obs,
        action_mask=None,
        action=None,
        deterministic=False,
    ):
        action_logits, glimpse = self.policy.forward(
            nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
        )
        probs = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            if deterministic:
                action = probs.mode
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic_net(glimpse)
