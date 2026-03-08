import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from core.policy_backbone import PolicyBackbone


ACTIVATION_FNS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "manual_gen_ratios": torch.sigmoid,
    "manual_temp": F.softplus,
    "manual_samp_temp": F.softplus,
    "manual_cfg": F.softplus,
    "timesteps": torch.sigmoid,
    "cfg": F.softplus,
    "top_k": lambda x: torch.sigmoid(x) * 4096.0,
    "top_p": torch.sigmoid,
}


def apply_activation(key: str, x: torch.Tensor) -> torch.Tensor:
    return ACTIVATION_FNS[key](x)


def apply_heuristic(
    action_dict: Dict[str, torch.Tensor],
    timestep: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {k: action_dict[k] + HEURISTIC_RESIDUALS[k][timestep] for k in action_dict}


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    return torch.log(x / (1 - x))


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.exp(-x) + 1)


INVERSE_ACTIVATION_FNS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "manual_gen_ratios": inverse_sigmoid,
    "manual_temp": inverse_softplus,
    "manual_samp_temp": inverse_softplus,
    "manual_cfg": inverse_softplus,
    "timesteps": inverse_sigmoid,
    "cfg": inverse_softplus,
    "top_k": lambda x: inverse_sigmoid(x / 4096.0),
    "top_p": inverse_sigmoid,
}


def inverse_activation(key: str, x: torch.Tensor) -> torch.Tensor:
    return INVERSE_ACTIVATION_FNS[key](x)


def apply_smooth(
    prev_action: Dict[str, torch.Tensor] | None,
    cur_action: Dict[str, torch.Tensor],
    smooth: float,
) -> Dict[str, torch.Tensor]:
    if prev_action is None:
        return cur_action
    return {k: smooth * prev_action[k] + (1 - smooth) * cur_action[k] for k in cur_action}


def inverse_smooth(seq: torch.Tensor, smooth: float) -> torch.Tensor:
    if smooth == 0:
        return seq
    out = [seq[0]]
    for i in range(1, len(seq)):
        out.append((seq[i] - smooth * seq[i - 1]) / (1 - smooth))
    return torch.stack(out)


HEURISTIC_RESIDUALS: Dict[str, torch.Tensor] | None = None


def build_heuristic_residual(args, device: torch.device) -> Dict[str, torch.Tensor]:
    global HEURISTIC_RESIDUALS

    if args.model_family == "maskgit":
        ratio = (torch.arange(args.gen_steps, device=device) + 1 - 1e-3) / args.gen_steps
        schedule = {
            "manual_gen_ratios": torch.cos(ratio * math.pi * 0.5),
            "manual_temp": (1 - ratio) * args.temp_init,
            "manual_samp_temp": torch.full((args.gen_steps,), args.manual_samp_temp_init, device=device),
            "manual_cfg": ((torch.arange(args.gen_steps, device=device) + 1e-3) / args.gen_steps) * args.cfg_init,
        }
    elif args.model_family == "dit":
        schedule = {
            "timesteps": torch.linspace(0.9, 1e-2, args.gen_steps + 1, device=device),
            "manual_cfg": torch.full((args.gen_steps + 1,), args.cfg_init, device=device),
        }
    elif args.model_family == "sit":
        schedule = {
            "timesteps": torch.linspace(0.01, 0.99, args.gen_steps + 1, device=device),
            "manual_cfg": torch.full((args.gen_steps + 1,), args.cfg_init, device=device),
        }
    elif args.model_family == "var":
        schedule = {
            "cfg": torch.full((args.gen_steps,), args.cfg_init, device=device),
            "top_k": torch.full((args.gen_steps,), args.topk, device=device),
            "top_p": torch.full((args.gen_steps,), args.topp, device=device),
            "manual_samp_temp": torch.full((args.gen_steps,), args.manual_samp_temp_init, device=device),
        }
    else:
        raise ValueError(f"unsupported model family: {args.model_family}")

    HEURISTIC_RESIDUALS = {
        k: inverse_smooth(inverse_activation(k, schedule[k]), args.action_smooth)
        for k in args.action_keys
    }
    return HEURISTIC_RESIDUALS


def build_eval_agent(
    ckpt_path: str,
    action_dim: int,
    device: torch.device,
    args,
) -> torch.nn.Module:
    from core.policy_net import PolicyNet

    state_dict = torch.load(ckpt_path, map_location="cpu")

    inferred_feat_dim = int(state_dict["x_embedder.weight"].shape[1])
    args.feat_dim = inferred_feat_dim

    backbone = PolicyBackbone(
        in_channels=inferred_feat_dim,
        out_channels=action_dim,
        hidden_size=512,
    ).to(device)
    backbone.load_state_dict(state_dict, strict=True)

    build_heuristic_residual(args, device)
    policy = PolicyNet(
        backbone=backbone,
        action_keys=list(args.action_keys),
        action_smooth=float(args.action_smooth),
    ).to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    return policy
