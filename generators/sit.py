import torch

import backbones.libs.autoencoder
from backbones.sit.model import SiT_models
from backbones.sit.solver import DPM_Solver, NoiseScheduleVP
from backbones.sit.transport import create_transport


def unpreprocess(v):
    v = 0.5 * (v + 1.0)
    v.clamp_(0.0, 1.0)
    return v


class SiTGenerator:
    def __init__(self, args, device, policy):
        self.args = args
        self.device = device
        self.policy = policy

        self.noise_schedule = NoiseScheduleVP(schedule="linear")

        self.nnet = SiT_models[args.sit_model](input_size=32, num_classes=1000).to(device)
        self.transport = create_transport(path_type="Linear", prediction="velocity")
        self.nnet.load_state_dict(torch.load(args.state_dict_path, map_location="cpu"))
        self.nnet.eval()
        self.nnet.requires_grad_(False)

        self.dpm_solver = DPM_Solver(self.model_fn, self.noise_schedule, predict_x0=True, thresholding=False)
        self.autoencoder = backbones.libs.autoencoder.get_model("assets/autoencoder_kl_ema.pth").to(device)

        self.t = 0
        self.y = None

    def model_fn(self, x, t_continuous):
        t = t_continuous
        cond, feat = self.nnet(x, t, y=self.y)

        state = {
            "feat": feat,
            "timestep": torch.full((x.size(0),), self.t, device=self.device),
            "contexts": self.y,
        }
        actdict = self._callback(state, t_continuous)

        new_t = actdict["timesteps"]
        cfg = actdict["manual_cfg"]

        uncond, _ = self.nnet(
            x,
            t,
            y=torch.tensor([1000] * x.size(0), device=self.device).unsqueeze(1),
        )
        cfg = cfg.reshape(-1, 1, 1, 1)
        res = cond + cfg * (cond - uncond)
        res[:, 3: res.size(1)] = cond[:, 3: res.size(1)]
        res = self.transport.path_sampler.get_noise_from_velocity(res, x, t)
        self.t += 1
        return res, new_t

    def _callback(self, state, t_continuous=None):
        actdict = self.policy(state)

        if t_continuous is not None:
            actdict["timesteps"] = torch.clamp(actdict["timesteps"], min=t_continuous + 0.01)
        actdict["timesteps"] = torch.clamp(actdict["timesteps"], max=0.99)

        return actdict

    @torch.no_grad()
    def full_step(self, contexts):
        self.t = 0
        self.y = contexts.unsqueeze(1)
        self.policy.reset()
        self.dpm_solver.prev_t = None

        z_init = torch.randn(len(self.y), 4, 32, 32, device=self.device)
        first_state = {
            "feat": torch.zeros(len(self.y), 256, self.args.feat_dim, device=self.device),
            "timestep": torch.full((len(self.y),), self.t, device=self.device),
            "contexts": self.y,
        }
        first = self._callback(first_state)
        self.t += 1

        self.dpm_solver.prev_t = first["timesteps"]
        samples = self.dpm_solver.sample(
            z_init,
            steps=self.args.gen_steps,
            order=2,
            method="singlestep",
        )
        samples = self.autoencoder.decode(samples)
        return unpreprocess(samples)
