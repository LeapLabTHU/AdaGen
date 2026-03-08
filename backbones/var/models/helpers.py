import torch
from torch import nn as nn
from torch.nn import functional as F


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: torch.Tensor = None, top_p: torch.Tensor = None, rng=None, num_samples=1, manual_samp_temp=None) -> torch.Tensor:
    B, l, V = logits_BlV.shape

    # Process top-k filtering
    if top_k is not None:
        for i in range(B):
            k = int(top_k[i].item())
            if k <= 0:
                continue  # Skip invalid k values
            logits_i = logits_BlV[i]  # (l, V)
            top_k_vals = logits_i.topk(k, dim=-1, largest=True, sorted=False).values  # (l, k)
            min_top_k = top_k_vals.amin(dim=-1, keepdim=True)  # (l, 1)
            mask = logits_i < min_top_k
            logits_BlV[i].masked_fill_(mask, -torch.inf)

    # Process top-p filtering (vectorized)
    if top_p is not None:
        logits_2d = logits_BlV.view(-1, V)  # (B*l, V)
        top_p_expanded = top_p.repeat_interleave(l).view(-1)  # (B*l,)

        # Sort logits in ascending order
        sorted_logits, sorted_indices = logits_2d.sort(dim=-1, descending=False)
        probs = sorted_logits.softmax(dim=-1)
        cum_probs = probs.cumsum(dim=-1)  # (B*l, V)

        # Remove tokens with cumulative probability <= (1 - top_p)
        threshold = 1 - top_p_expanded.unsqueeze(-1)
        sorted_mask = cum_probs <= threshold

        # Ensure at least one token is kept
        sorted_mask[..., -1:] = False

        # Scatter mask to original indices
        idx_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits_2d.masked_fill_(idx_to_remove, -torch.inf)

    # Sampling
    logits_BlV = logits_BlV / manual_samp_temp.reshape(-1, 1, 1)
    probs = logits_BlV.softmax(dim=-1)
    num_samples = abs(num_samples)
    replacement = num_samples >= 0
    samples = torch.multinomial(
        probs.view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng
    ).view(B, l, num_samples)

    return samples


def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'
