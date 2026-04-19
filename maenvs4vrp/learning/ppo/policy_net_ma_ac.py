import torch
import torch.nn as nn


import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = (
            torch.bmm(q, k.transpose(1, 2)) / self.scale_factor
        )  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            #    assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask == False, float("-inf"))

        attn = self.softmax(attn)
        outputs = torch.bmm(attn, v)  # outputs: [b_size x len_q x d_v]
        return outputs, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(_MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_v))

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q, k, v, attn_mask=None, use_adj_mask=False):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(
            n_heads, -1, d_model
        )  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(
            n_heads, -1, d_model
        )  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(
            n_heads, -1, d_model
        )  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(
            b_size * n_heads, -1, d_k
        )  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(
            b_size * n_heads, -1, d_k
        )  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(
            b_size * n_heads, -1, d_v
        )  # [b_size * n_heads x len_v x d_v]

        if attn_mask is not None:
            if use_adj_mask:
                outputs, attn = self.attention(
                    q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_heads, 1, 1)
                )
            else:
                outputs, attn = self.attention(
                    q_s,
                    k_s,
                    v_s,
                    attn_mask=attn_mask.unsqueeze(1).repeat(n_heads, 1, 1),
                )
        else:
            outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=None)

        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.attention = _MultiHeadAttention(d_model, n_heads)
        self.proj = nn.Linear(n_heads * self.d_k, d_model)
        # self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask=None, use_adj_mask=False):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)

        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(
            q, k, v, attn_mask=attn_mask, use_adj_mask=use_adj_mask
        )
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)
        return outputs


class DotAttention(nn.Module):
    #  Dot Attention
    def __init__(self, in_size1, in_size2, hidden_size, is_glimpse=False, C=10):
        super(DotAttention, self).__init__()

        self.scale_factor = np.sqrt(hidden_size)

        self.tanh = nn.Tanh()
        self.C = C  # tanh exploration

        self.Wk = nn.Linear(in_size1, hidden_size, bias=False)
        self.Wq = nn.Linear(in_size2, hidden_size, bias=False)
        self.Wv = nn.Linear(in_size1, hidden_size)

        self.is_glimpse = is_glimpse
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_outputs, h0, mask=None):
        k = self.Wk(enc_outputs)
        q = self.Wq(h0).unsqueeze(1)
        v = self.Wv(enc_outputs)

        logits = (
            torch.bmm(q, k.transpose(1, 2)) / self.scale_factor
        )  # attn: [b_size x len_q x len_k]
        if self.is_glimpse:
            if mask is not None:
                logits = logits.masked_fill(
                    mask.unsqueeze(1) == False, float("-inf")
                )  # mask out invalid actions
            att = self.softmax(logits)
            out = torch.bmm(att, v).squeeze(1)
            return out
        else:
            logits = self.C * self.tanh(logits).squeeze(1)
            if mask is not None:
                logits = logits.masked_fill(
                    mask == False, float("-inf")
                )  # mask out invalid actions
            return logits


class Attention(nn.Module):
    #  Bahdanau Attention (sum)
    def __init__(self, in_size1, in_size2, hidden_size, is_glimpse=False, C=10):
        super(Attention, self).__init__()

        self.tanh = nn.Tanh()
        self.C = C  # tanh exploration
        self.W1 = nn.Linear(in_size1, hidden_size, bias=False)
        self.W2 = nn.Linear(in_size2, hidden_size)
        self.V = nn.Parameter(torch.zeros((hidden_size, 1), requires_grad=True))
        self.is_glimpse = is_glimpse
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc_outputs, h0, mask=None):
        w1e = self.W1(enc_outputs)
        w2h = self.W2(h0).unsqueeze(1)
        u = self.tanh(w1e + w2h)
        logits = u.matmul(self.V)
        if self.is_glimpse:
            if mask is not None:
                logits = logits.masked_fill(
                    mask.unsqueeze(-1) == False, float("-inf")
                )  # mask out invalid actions
            att = self.softmax(logits).transpose(1, 2)
            out = torch.bmm(att, enc_outputs).squeeze(1)
            return out
        else:
            logits = self.C * self.tanh(logits).squeeze(-1)
            if mask is not None:
                logits = logits.masked_fill(
                    mask == False, float("-inf")
                )  # mask out invalid actions
            return logits


class Pointer(nn.Module):
    def __init__(self, in_size1, in_size2, hidden_size, has_glimpse=False):
        super(Pointer, self).__init__()

        self.has_glimpse = has_glimpse

        self.pointer = DotAttention(in_size1, in_size2, hidden_size)
        if self.has_glimpse:
            self.glimpse = DotAttention(
                in_size1, in_size2, hidden_size, is_glimpse=True
            )

    def forward(self, enc_outputs, hidden, mask):
        if self.has_glimpse:
            glimpse_h0 = self.glimpse(enc_outputs, hidden, mask)
            policy = self.pointer(enc_outputs, glimpse_h0, mask)
        else:
            policy = self.pointer(enc_outputs, hidden, mask)
        return policy


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, pre_lnorm=False):
        super(EncoderLayer, self).__init__()

        self.pre_lnorm = pre_lnorm
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, src, self_attn_mask, use_adj_mask=False):
        if self.pre_lnorm:
            src = self.layer_norm1(src)
            src2 = self.self_attn(
                src, src, src, attn_mask=self_attn_mask, use_adj_mask=use_adj_mask
            )
            src = src + self.dropout1(src2)
            src = self.layer_norm2(src)
            src2 = self.w_2(self.dropout2(self.relu(self.w_1(src))))
            src = src + self.dropout3(src2)
        else:
            src2 = self.self_attn(
                src, src, src, attn_mask=self_attn_mask, use_adj_mask=use_adj_mask
            )
            src = src + self.dropout1(src2)
            src = self.layer_norm1(src)
            src2 = self.w_2(self.dropout2(self.relu(self.w_1(src))))
            src = src + self.dropout3(src2)
            src = self.layer_norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()

        n_heads = 8  # args.n_heads # number of heads
        ff_dim = 256  # args.ff_dim # feed_forward_hidden
        n_layers = 2  # args.n_layers # number of Layers
        dropout = 0.0  # args.dropout
        self.pre_lnorm = False  # args.pre_lnorm

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim, ff_dim, n_heads, dropout, pre_lnorm=self.pre_lnorm
                )
                for _ in range(n_layers)
            ]
        )
        self.last_norm = nn.LayerNorm(hidden_dim)
        self.use_adj_mask = False  # args.use_lookahead

    def forward(self, emb_inp, mask):
        for layer in self.layers:
            emb_inp = layer(emb_inp, mask, self.use_adj_mask)

        if self.pre_lnorm:
            emb_inp = self.last_norm(emb_inp)
        return emb_inp


class PolicyNet(nn.Module):
    def __init__(
        self,
        nodes_feat_dim,
        agent_feat_dim,
        agents_feat_dim,
        global_feat_dim,
        hidden_dim,
    ):
        super(PolicyNet, self).__init__()

        self.nodes_embedding = nn.Linear(nodes_feat_dim, hidden_dim, bias=False)
        self.agent_embedding = nn.Linear(
            agent_feat_dim + global_feat_dim, hidden_dim, bias=False
        )

        self.nodes_encoder = Encoder(hidden_dim)
        self.active_agents_encoder = Encoder(hidden_dim)

        self.active_agents_embedding = nn.Linear(
            agents_feat_dim, hidden_dim, bias=False
        )

        self.active_agents_glimpse = DotAttention(
            hidden_dim, hidden_dim, hidden_dim, is_glimpse=True
        )
        self.nodes_glimpse = DotAttention(
            hidden_dim, hidden_dim, hidden_dim, is_glimpse=True
        )

        self.action_net = Pointer(
            hidden_dim, 2 * hidden_dim, hidden_dim, has_glimpse=False
        )

        self.critic_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self, nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
    ):
        agent_global_obs = torch.concat((self_obs, global_obs), dim=-1)

        nodes_emb = self.nodes_embedding(nodes_obs)
        agent_emb = self.agent_embedding(agent_global_obs)
        agents_emb = self.active_agents_embedding(agents_obs)

        nodes_encoded = self.nodes_encoder(nodes_emb, action_mask)
        agents_encoded = self.active_agents_encoder(agents_emb, agents_mask)

        agents_state = self.active_agents_glimpse(
            agents_encoded, agent_emb, agents_mask
        )
        # assert not torch.isnan(agents_state).any()

        nodes_state = self.nodes_glimpse(nodes_encoded, agent_emb, action_mask)
        agent_state = torch.concat((agents_state, nodes_state), dim=-1)

        return nodes_encoded, agent_state

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
        nodes_encoded, agent_encoded = self.forward(
            nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
        )

        action_logits = self.action_net(nodes_encoded, agent_encoded, action_mask)

        probs = torch.distributions.Categorical(logits=action_logits)
        if not deterministic:
            return probs.sample()
        else:
            return probs.mode

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
        nodes_encoded, agent_state = self.forward(
            nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
        )
        action_logits = self.action_net(nodes_encoded, agent_state, action_mask)

        probs = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            if not deterministic:
                action = probs.sample()
            else:
                action = probs.mode

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_net(agent_state),
        )


class EgoPolicyNet(nn.Module):
    def __init__(
        self,
        nodes_feat_dim,
        agent_feat_dim,
        agents_feat_dim,
        global_feat_dim,
        hidden_dim,
    ):
        super(EgoPolicyNet, self).__init__()

        self.nodes_embedding = nn.Linear(nodes_feat_dim, hidden_dim, bias=False)
        self.agent_embedding = nn.Linear(
            agent_feat_dim + global_feat_dim, hidden_dim, bias=False
        )

        self.nodes_encoder = Encoder(hidden_dim)
        self.active_agents_encoder = Encoder(hidden_dim)

        self.nodes_glimpse = DotAttention(
            hidden_dim, hidden_dim, hidden_dim, is_glimpse=True
        )

        self.action_net = Pointer(hidden_dim, hidden_dim, hidden_dim, has_glimpse=False)

        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self, nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
    ):
        agent_global_obs = torch.concat((self_obs, global_obs), dim=-1)

        nodes_emb = self.nodes_embedding(nodes_obs)
        agent_emb = self.agent_embedding(agent_global_obs)

        nodes_encoded = self.nodes_encoder(nodes_emb, action_mask)
        agent_state = self.nodes_glimpse(nodes_encoded, agent_emb, action_mask)

        return nodes_encoded, agent_state

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
        nodes_encoded, agent_encoded = self.forward(
            nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
        )

        action_logits = self.action_net(nodes_encoded, agent_encoded, action_mask)

        probs = torch.distributions.Categorical(logits=action_logits)
        if not deterministic:
            return probs.sample()
        else:
            return probs.mode

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
        nodes_encoded, agent_state = self.forward(
            nodes_obs, self_obs, agents_obs, agents_mask, global_obs, action_mask
        )
        action_logits = self.action_net(nodes_encoded, agent_state, action_mask)

        probs = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            if not deterministic:
                action = probs.sample()
            else:
                action = probs.mode

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_net(agent_state),
        )
