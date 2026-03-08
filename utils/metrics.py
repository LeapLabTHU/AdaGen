import numpy as np
import torch
from loguru import logger
from torch_fidelity import FeatureExtractorInceptionV3
from tqdm import tqdm


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def calc_fid(pred_tensor, m2, s2):
    m1 = torch.mean(pred_tensor, dim=0)
    pred_centered = pred_tensor - pred_tensor.mean(dim=0)
    s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)

    m1 = m1.double()
    s1 = s1.double()
    a = (m1 - m2).square().sum(dim=-1)
    b = s1.trace() + s2.trace()
    eigvals = torch.from_numpy(np.linalg.eigvals((s1 @ s2).cpu().numpy())).to(m1.device)
    c = eigvals.sqrt().real.sum(dim=-1)
    return (a + b - 2 * c).item()


def isc_features_to_metric(feature, splits=10, shuffle=True, rng_seed=2020):
    assert torch.is_tensor(feature) and feature.dim() == 2
    n, _ = feature.shape
    logger.info("isc feature shape is: {},{}", *feature.shape)
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        feature = feature[rng.permutation(n), :]
    feature = feature.double()

    p = feature.softmax(dim=1)
    log_p = feature.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * n // splits): ((i + 1) * n // splits), :]
        log_p_chunk = log_p[(i * n // splits): ((i + 1) * n // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        scores.append(kl.sum(dim=1).mean().exp().item())
    return float(np.mean(scores))


class FIDCalculator:
    def __init__(
        self,
        ref_stats_path: str,
        n_samples: int,
        device: torch.device,
        accelerator,
        test_bsz: int,
        inception_weights_path: str,
        gather: bool = True,
    ):
        self.device = device
        self.accelerator = accelerator
        self.test_bsz = test_bsz
        self.gather = gather
        self.n_samples = n_samples

        self.inception = FeatureExtractorInceptionV3(
            name="inception-v3",
            features_list=["2048", "logits_unbiased"],
            feature_extractor_internal_dtype="float32",
            feature_extractor_weights_path=inception_weights_path,
        ).to(device)

        with np.load(ref_stats_path) as f:
            mu = torch.from_numpy(f["mu"][:]).to(device)
            sigma = torch.from_numpy(f["sigma"][:]).to(device)
        self.ref_stats = (mu, sigma)

        batch_size = test_bsz * accelerator.num_processes if gather else test_bsz
        self.total_iters = len(amortize(n_samples, batch_size))
        self.has_init = False

    def init(self):
        self.pred_tensor = torch.empty((self.n_samples * 2, 2048), device=self.device)
        self.logits_tensor = torch.empty((self.n_samples * 2, 1008), device=self.device)

        if self.n_samples % 1000 != 0:
            raise ValueError("n_samples must be divisible by 1000 for balanced label sampling")
        labels = torch.from_numpy(np.arange(0, 1000).repeat(self.n_samples // 1000)).to(self.device)
        self.labels = torch.cat([labels, torch.zeros(self.n_samples, device=self.device)]).long()

        self.labels_generated = True
        self.idx = 0
        self.pbar = tqdm(total=self.total_iters, desc="FID", leave=True)
        self.has_init = True

    def get_labels(self):
        assert self.labels_generated, "samples corresponding to previous labels are not generated"
        self.labels_generated = False
        offset = self.total_iters * self.test_bsz * self.accelerator.process_index + self.idx
        return self.labels[offset:offset + self.test_bsz]

    def _collate_tensor(self, tensor: torch.Tensor):
        tensor = tensor[: self.idx]
        if self.gather:
            tensor = self.accelerator.gather(tensor)
        assert self.n_samples <= tensor.shape[0]
        return tensor[: self.n_samples]

    def get_metrics(self):
        pred = self._collate_tensor(self.pred_tensor)
        fid = calc_fid(pred, *self.ref_stats)

        logits = self._collate_tensor(self.logits_tensor)
        isc = isc_features_to_metric(logits)

        self.has_init = False
        logger.info("FID: {}", fid)
        logger.info("ISC: {}", isc)
        return {f"fid_{self.n_samples}": fid, f"isc_{self.n_samples}": isc}

    def add(self, samples: torch.Tensor):
        if not self.has_init:
            self.init()

        samples = samples.clamp_(0.0, 1.0)
        feat2048, logits = self.inception(samples.float())
        self.pred_tensor[self.idx:self.idx + feat2048.shape[0]] = feat2048
        self.logits_tensor[self.idx:self.idx + logits.shape[0]] = logits

        self.labels_generated = True
        self.idx += feat2048.shape[0]
        self.pbar.update(1)

        if self.pbar.n == self.total_iters:
            return self.get_metrics()
        return None
