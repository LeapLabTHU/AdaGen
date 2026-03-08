import torch
from einops import rearrange
from torch.nn import functional as F

from backbones.var.models import build_vae_var
from backbones.var.models.helpers import sample_with_top_k_top_p_


class VARGenerator:
    def __init__(self, args, device, policy):
        self.args = args
        self.device = device
        self.policy = policy
        self.nnet, self.vae = self._prepare_nnet()
        self.feat_dim = int(self.nnet.C)

    def _prepare_nnet(self):
        vae, nnet = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,
            device="cpu",
            patch_nums=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
            num_classes=1000,
            depth=self.args.model_depth,
            shared_aln=False,
        )
        vae.load_state_dict(torch.load(self.args.vae_state_dict_path, map_location="cpu"), strict=True)
        nnet.load_state_dict(torch.load(self.args.var_state_dict_path, map_location="cpu"), strict=True)

        vae = vae.to(self.device)
        nnet = nnet.to(self.device)
        vae.eval()
        nnet.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        for p in nnet.parameters():
            p.requires_grad_(False)
        return nnet, vae

    def reset(self, contexts):
        label_B = contexts
        B = len(label_B)

        sos = cond_BD = self.nnet.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=1000)), dim=0)
        )
        lvl_pos = self.nnet.lvl_embed(self.nnet.lvl_1L) + self.nnet.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand(2 * B, self.nnet.first_l, -1)
            + self.nnet.pos_start.expand(2 * B, self.nnet.first_l, -1)
            + lvl_pos[:, : self.nnet.first_l]
        )

        f_hat = sos.new_zeros(B, self.nnet.Cvae, self.nnet.patch_nums[-1], self.nnet.patch_nums[-1])
        self.state = {
            "f_hat": f_hat,
            "next_token_map": next_token_map,
            "cur_L": 0,
            "contexts": label_B,
            "label_B": label_B,
            "cond_BD": cond_BD,
            "timestep": torch.zeros((len(label_B),), dtype=torch.long, device=self.device),
            "cur_index": 0,
            "lvl_pos": lvl_pos,
        }

        for b in self.nnet.blocks:
            b.attn.kv_caching(True)
        x = next_token_map
        for b in self.nnet.blocks:
            x = b(x=x, cond_BD=self.nnet.shared_ada_lin(cond_BD), attn_bias=None)
        self.state["x"] = x

        feat = F.interpolate(
            x[:B].transpose(1, 2).reshape(B, self.feat_dim, 1, 1), size=(16, 16), mode="bicubic"
        )
        self.state["feat"] = rearrange(feat, "B C H W -> B (H W) C")
        return self.state

    def step(self, actdict):
        cfg = actdict["cfg"]
        top_k = actdict["top_k"]
        top_p = actdict["top_p"]
        manual_samp_temp = actdict["manual_samp_temp"]

        lvl_pos = self.state["lvl_pos"]
        si = self.state["cur_index"]
        ratio = si / self.nnet.num_stages_minus_1
        pn = self.nnet.patch_nums[si]
        cond_BD = self.state["cond_BD"]
        B = len(self.state["label_B"])
        x = self.state["x"]
        cur_L = self.state["cur_L"] + pn ** 2
        f_hat = self.state["f_hat"]

        logits_BlV = self.nnet.get_logits(x, cond_BD)
        t = (cfg * ratio).view(B, 1, 1)
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(
            logits_BlV,
            rng=None,
            top_k=top_k,
            top_p=top_p,
            num_samples=1,
            manual_samp_temp=manual_samp_temp,
        )[:, :, 0]
        h_BChw = self.nnet.vae_quant_proxy[0].embedding(idx_Bl)
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.nnet.Cvae, pn, pn)
        f_hat, next_token_map = self.nnet.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(self.nnet.patch_nums), f_hat, h_BChw
        )

        if si != self.nnet.num_stages_minus_1:
            next_token_map = next_token_map.view(B, self.nnet.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.nnet.word_embed(next_token_map)
                + lvl_pos[:, cur_L: cur_L + self.nnet.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(2, 1, 1)

        self.state["next_token_map"] = next_token_map
        self.state["cur_L"] = cur_L
        self.state["cur_index"] = si + 1
        self.state["timestep"] = self.state["timestep"] + 1
        self.state["f_hat"] = f_hat
        self.state["cond_BD"] = cond_BD

        if si != self.nnet.num_stages_minus_1:
            x = next_token_map
            for b in self.nnet.blocks:
                x = b(x=x, cond_BD=self.nnet.shared_ada_lin(cond_BD), attn_bias=None)
            self.state["x"] = x
            feat = F.interpolate(
                x[:B]
                .transpose(1, 2)
                .reshape(B, self.feat_dim, self.nnet.patch_nums[si + 1], self.nnet.patch_nums[si + 1]),
                size=(16, 16),
                mode="bicubic",
            )
            self.state["feat"] = rearrange(feat, "B C H W -> B (H W) C")
        return self.state

    @torch.no_grad()
    def decode(self):
        for b in self.nnet.blocks:
            b.attn.kv_caching(False)
        return self.nnet.vae_proxy[0].fhat_to_img(self.state["f_hat"]).add_(1).mul_(0.5)

    @torch.no_grad()
    def full_step(self, contexts):
        self.policy.reset()
        state = self.reset(contexts=contexts)
        for _ in range(self.args.gen_steps):
            actdict = self.policy(state)
            state = self.step(actdict)
        return self.decode()
