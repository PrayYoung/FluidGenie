"""Run explicit context-only and oracle LAM protocols on an unseen episode."""

from dataclasses import dataclass
from pathlib import Path
import tyro
from configs.eval_configs import RolloutConfig, apply_ckpt_config_to_rollout
from configs.config_io import load_config_json
from fluidgenie.data.splits import assert_disjoint
from fluidgenie.eval.rollout_runner import run_rollout


@dataclass
class Args:
    tokenizer: str
    dynamics: str
    lam: str
    train: str
    heldout: str
    stats: str
    out: str
    horizon: int = 40


def main(a):
    assert_disjoint(a.train, a.heldout)
    for mode in ["context", "oracle"]:
        cfg = RolloutConfig(
            npz_path=str(Path(a.heldout) / "episode_000000.npy"),
            vq_ckpt=a.tokenizer,
            dyn_ckpt=a.dynamics,
            out_dir=str(Path(a.out) / mode),
            context=2,
            horizon=a.horizon,
            tokenizer_arch="st",
            model_type="st_maskgit",
            stats_path=a.stats,
            use_lam=True,
            lam_ckpt=a.lam,
            lam_rollout_mode=mode,
        )
        cfg = apply_ckpt_config_to_rollout(cfg)
        run_rollout(cfg)


if __name__ == "__main__":
    main(tyro.cli(Args))
