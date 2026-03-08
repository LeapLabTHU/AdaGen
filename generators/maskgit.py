import numpy as np
import torch
from einops import rearrange

from backbones.libs.taming.models import vqgan
from backbones.maskgit.model import UViT


def build_maskgit_model():
    return UViT(
        img_size=16,
        codebook_size=1024,
        embed_dim=768,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        num_classes=1001,
        use_checkpoint=False,
        skip=True,
    )


class MaskGITGenerator:
    def __init__(self, args, device, policy):
        self.args = args
        self.device = device
        self.policy = policy

        self.seq_len = 256
        self.mask_ind = 1024
        self.nnet = self._prepare_nnet()
        self.autoencoder = vqgan.get_model().to(device)
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
        )

    def _prepare_nnet(self):
        nnet = build_maskgit_model().to(self.device)
        nnet.load_state_dict(torch.load(self.args.state_dict_path, map_location="cpu"))
        nnet.eval()
        nnet.requires_grad_(False)
        return nnet

    def _add_gumbel_noise(self, t, temperature):
        g = self.gumbel_dist.sample(t.shape).squeeze()
        return t + temperature.unsqueeze(1) * g

    def reset(self, contexts):
        masked_ids = torch.full((len(contexts), self.seq_len), self.mask_ind, dtype=torch.long, device=self.device)
        self.state = self.nnet(masked_ids, context=contexts, return_dict=True)
        self.state["masked_ids"] = masked_ids
        self.state["prev_mask_ratio"] = torch.ones((len(contexts),), dtype=torch.float32, device=self.device)
        self.state["timestep"] = torch.zeros((len(contexts),), dtype=torch.long, device=self.device)
        self.state["contexts"] = contexts
        return self.state

    def step(self, actdict):
        manual_gen_ratios = actdict["manual_gen_ratios"]
        manual_temp = actdict["manual_temp"]
        manual_samp_temp = actdict["manual_samp_temp"]
        manual_cfg = actdict["manual_cfg"]

        logits = self.state["logits"]
        empty_ctx = torch.full((len(manual_gen_ratios), 1), 1000, dtype=torch.long, device=self.device)
        logits_wo_cfg = self.nnet(self.state["masked_ids"], context=empty_ctx)
        logits = logits + manual_cfg.unsqueeze(1).unsqueeze(1) * (logits - logits_wo_cfg)

        is_mask = self.state["masked_ids"] == self.mask_ind
        sampled_ids = torch.distributions.Categorical(
            logits=logits / manual_samp_temp.unsqueeze(1).unsqueeze(2)
        ).sample()
        logits = torch.log_softmax(logits, dim=-1)
        sampled_logits = torch.squeeze(torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)

        sampled_ids = torch.where(is_mask, sampled_ids, self.state["masked_ids"])
        sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()

        mask_len = torch.floor(self.seq_len * manual_gen_ratios).clamp(max=self.seq_len - 1)
        confidence = self._add_gumbel_noise(sampled_logits, manual_temp)
        sorted_confidence, _ = torch.sort(confidence, axis=-1)
        cut_off = torch.gather(sorted_confidence, 1, mask_len.long().unsqueeze(1))
        masking = confidence < cut_off
        masked_ids = torch.where(masking, self.mask_ind, sampled_ids)

        self.state.update(self.nnet(masked_ids, context=self.state["contexts"], return_dict=True))
        self.state["masked_ids"] = masked_ids
        self.state["sampled_ids"] = sampled_ids
        self.state["prev_mask_ratio"] = manual_gen_ratios
        self.state["timestep"] = self.state["timestep"] + 1
        return self.state

    @torch.no_grad()
    def decode(self):
        z = rearrange(self.state["sampled_ids"], "b (i j) -> b i j", i=16, j=16)
        images = self.autoencoder.decode_code(z)
        return images.clamp_(0.0, 1.0)

    @torch.no_grad()
    def full_step(self, contexts):
        contexts = contexts.unsqueeze(1)
        self.policy.reset()
        state = self.reset(contexts=contexts)
        for _ in range(self.args.gen_steps):
            actdict = self.policy(state)
            state = self.step(actdict)
        return self.decode()
