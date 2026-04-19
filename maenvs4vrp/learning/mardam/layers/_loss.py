import torch
import torch.nn.functional as F

from itertools import repeat


def reinforce_loss(
    logprobs, rewards, baseline=None, weights=None, discount=1.0, reduction="mean"
):
    r"""
    :param logprobs:  Iterable of length :math:`L` on tensors of size :math:`N \times 1`
    :param rewards:   Iterable of length :math:`L` on tensors of size :math:`N \times 1`
                    or single tensor of size :math:`N \times 1` to use rewards cumulated on the whole trajectory
    :param baseline:  Iterable of length :math:`L` on tensors of size :math:`N \times 1`
                    or single tensor of size :math:`N \times 1` to use rewards cumulated on the whole trajectory
    :param weights:   Iterable of length :math:`L` on tensors of size :math:`N \times 1`
    :param discount:  Discount applied to cumulated future reward
    :param reduction: 'none' No reduction,
                      'sum'  Compute sum of loss on batch,
                      'mean' Compute mean of loss on batch
    """
    if weights is None:
        weights = repeat(1.0)

    if isinstance(rewards, torch.Tensor):
        if baseline is None:
            baseline = torch.zeros_like(rewards)

        loss = torch.stack([-logp * w for logp, w in zip(logprobs, weights)]).sum(dim=0)
        loss *= rewards - baseline.detach()

        if baseline.requires_grad:
            loss += F.smooth_l1_loss(baseline, rewards)

    else:
        if baseline is None:
            baseline = repeat(torch.zeros_like(rewards[0]))

        cumul = torch.zeros_like(rewards[0])
        vals = []
        for r in reversed(rewards):
            cumul = r + discount * cumul
            vals.append(cumul)
        vals.reverse()

        loss = []
        bl_loss = []
        for val, logp, bl, w in zip(vals, logprobs, baseline, weights):
            loss.append(-logp * (val - bl.detach()) * w)
            if bl.requires_grad:
                bl_loss.append(F.smooth_l1_loss(bl, val))
        loss = torch.stack(loss).sum(dim=0)

        if bl_loss:
            loss += torch.stack(bl_loss).sum(dim=0)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:  # reduction == 'mean'
        return loss.mean()


def reinforce_mask_loss(
    logprobs,
    rewards,
    baseline=None,
    step_mask=None,
    weights=None,
    discount=1.0,
    reduction="mean",
):
    r"""
    :param logprobs:  Iterable of length :math:`L` on tensors of size :math:`N \times 1`
    :param rewards:   Iterable of length :math:`L` on tensors of size :math:`N \times 1`
                    or single tensor of size :math:`N \times 1` to use rewards cumulated on the whole trajectory
    :param baseline:  Iterable of length :math:`L` on tensors of size :math:`N \times 1`
                    or single tensor of size :math:`N \times 1` to use rewards cumulated on the whole trajectory
    :param weights:   Iterable of length :math:`L` on tensors of size :math:`N \times 1`
    :param discount:  Discount applied to cumulated future reward
    :param reduction: 'none' No reduction,
                      'sum'  Compute sum of loss on batch,
                      'mean' Compute mean of loss on batch
    """
    if weights is None:
        weights = repeat(1.0)

    step_mask = torch.stack(step_mask)

    if isinstance(rewards, torch.Tensor):
        if baseline is None:
            baseline = torch.zeros_like(rewards)

        loss = torch.stack(
            [-logp * w for logp, w in zip(logprobs, weights)]
        )  # .sum(dim = 0)
        loss *= rewards - baseline.detach()

        if baseline.requires_grad:
            loss += F.smooth_l1_loss(baseline, rewards)

    else:
        if baseline is None:
            baseline = repeat(torch.zeros_like(rewards[0]))

        cumul = torch.zeros_like(rewards[0])
        vals = []
        for r in reversed(rewards):
            cumul = r + discount * cumul
            vals.append(cumul)
        vals.reverse()

        loss = []
        bl_loss = []
        for val, logp, bl, w in zip(vals, logprobs, baseline, weights):
            loss.append(-logp * (val - bl.detach()) * w)
            if bl.requires_grad:
                bl_loss.append(F.smooth_l1_loss(bl, val, reduction="none"))
        loss = torch.stack(loss)  # .sum(dim = 0)

        if bl_loss:
            loss += torch.stack(bl_loss)  # .sum(dim = 0)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:  # reduction == 'mean'
        return (
            (loss * step_mask).sum(dim=0).mean()
        )  # loss[step_mask].sum() / torch.sum(step_mask)
