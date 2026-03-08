import argparse
from pathlib import Path

import accelerate
import yaml
from accelerate import DistributedDataParallelKwargs
from loguru import logger

from core.helpers import build_eval_agent
from utils.metrics import FIDCalculator

REF_STATS_PATH = "assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"
INCEPTION_WEIGHTS_PATH = "assets/pt_inception-2015-12-05-6726825d.pth"


def parse_args():
    p = argparse.ArgumentParser("Unified eval-only entry (v5)")

    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--eval_path", type=str, required=True)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mixed_precision", type=str, default="fp16")
    p.add_argument("--test_bsz", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=50000)
    return p.parse_args()


def load_recipe(model_config: str) -> tuple[Path, dict]:
    path = Path(model_config)
    with open(path, "r", encoding="utf-8") as f:
        return path, yaml.safe_load(f)


def apply_config(args, config: dict):
    args.model_type = Path(args.model_config).stem
    for k, v in config.items():
        setattr(args, k, v)


def build_generator(args, device, policy):
    if args.model_family == "maskgit":
        from generators.maskgit import MaskGITGenerator

        return MaskGITGenerator(args, device, policy)

    if args.model_family == "dit":
        from generators.dit import DiTGenerator

        return DiTGenerator(args, device, policy)

    if args.model_family == "sit":
        from generators.sit import SiTGenerator

        return SiTGenerator(args, device, policy)

    from generators.var import VARGenerator

    return VARGenerator(args, device, policy)


def evaluate_once(args, accelerator, device, ckpt_path):
    action_dim = len(args.action_keys)

    policy = build_eval_agent(
        ckpt_path=ckpt_path,
        action_dim=action_dim,
        device=device,
        args=args,
    )

    generator = build_generator(args, device, policy)

    fid_calculator = FIDCalculator(
        ref_stats_path=REF_STATS_PATH,
        n_samples=args.n_samples,
        device=device,
        accelerator=accelerator,
        test_bsz=args.test_bsz,
        inception_weights_path=INCEPTION_WEIGHTS_PATH,
        gather=True,
    )
    fid_calculator.init()

    while True:
        labels = fid_calculator.get_labels()
        images = generator.full_step(contexts=labels)

        res = fid_calculator.add(images)
        if res is not None:
            return {
                "ckpt": ckpt_path,
                "model_type": args.model_type,
                "fid": float(res[f"fid_{args.n_samples}"]),
                "isc": float(res[f"isc_{args.n_samples}"]),
                "n_samples": args.n_samples,
            }


def main():
    args = parse_args()
    recipe_path, recipe = load_recipe(args.model_config)
    apply_config(args, recipe)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
    device = accelerator.device

    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info(
        "model_config={} model_type={} family={} mixed_precision={} n_samples={}",
        recipe_path,
        args.model_type,
        args.model_family,
        args.mixed_precision,
        args.n_samples,
    )

    result = evaluate_once(args, accelerator, device, args.eval_path)
    if accelerator.is_main_process:
        logger.info(
            "EVAL_RESULT model_type={} ckpt={} fid={} isc={} n_samples={}",
            result["model_type"],
            result["ckpt"],
            result["fid"],
            result["isc"],
            result["n_samples"],
        )


if __name__ == "__main__":
    main()
