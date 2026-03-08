from typing import Dict, List

import torch
import torch.nn as nn

from core.helpers import apply_activation, apply_heuristic, apply_smooth


class PolicyNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_keys: List[str],
        action_smooth: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.keys = action_keys
        self.action_smooth = action_smooth
        self.prev_action: Dict[str, torch.Tensor] | None = None

    def reset(self):
        self.prev_action = None

    def _postprocess_actions(self, action_tensor: torch.Tensor, timestep) -> Dict[str, torch.Tensor]:
        action_tensor = action_tensor.reshape(action_tensor.shape[0], -1)
        timestep = timestep.long()
        action_dict = {k: action_tensor[:, i] for i, k in enumerate(self.keys)}

        action_dict = apply_heuristic(action_dict, timestep)
        action_dict = apply_smooth(self.prev_action, action_dict, self.action_smooth)
        self.prev_action = {k: v.detach() for k, v in action_dict.items()}
        return {k: apply_activation(k, action_dict[k]) for k in self.keys}

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        action_tensor = self.backbone(state)
        action_dict = self._postprocess_actions(action_tensor, timestep=state["timestep"])
        return action_dict
