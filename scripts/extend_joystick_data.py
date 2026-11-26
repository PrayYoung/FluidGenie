"""Extend the frozen joystick split without regenerating any original episode."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import tyro
from configs.gen_data_configs import NS2DConfig
from fluidgenie.data.compute_stats import compute_stats
from fluidgenie.data.gen_phiflow_ns2d import generate_episode, save_episode_npy
from fluidgenie.data.splits import assert_disjoint

@dataclass
class Args:
    source: str = 'data/repair_v4/joystick'
    out: str = 'data/lam_v5/joystick'
    train_episodes: int = 60


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_original(source, manifest):
    actual = assert_disjoint(source / 'train', source / 'heldout')
    for split in ('train', 'heldout'):
        if actual[split] != manifest[split]:
            raise ValueError('Original split differs from frozen manifest')
        for entry in actual[split]:
            p = source / split / entry['file']
            meta = json.loads(p.with_suffix('.json').read_text())
            if not np.array_equal(np.load(p.parent / 'actions' / p.name), meta['actions']):
                raise ValueError('Action sidecar differs from original metadata')
    return actual


def extend(args):
    source, out = Path(args.source), Path(args.out)
    saved = json.loads((source / 'split_manifest.json').read_text())
    original = verify_original(source, saved)
    n = len(original['train'])
    if [r['seed'] for r in original['train']] != list(range(n)):
        raise ValueError('Expected original consecutive train seeds starting at zero')
    if args.train_episodes <= n:
        raise ValueError('Extended split must add training episodes')
    cfg = NS2DConfig(**saved['generator_config'])
    if cfg.action_mode != 'joystick5_localized':
        raise ValueError('Expected frozen joystick generator')
    if set(range(args.train_episodes)) & {r['seed'] for r in original['heldout']}:
        raise ValueError('Extended training seeds overlap heldout')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    # Hardlink unchanged inputs when possible; never write through these links.
    for split in ('train', 'heldout'):
        for p in (source / split).rglob('*'):
            if p.is_file():
                dest = out / split / p.relative_to(source / split)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(p, dest)
                except OSError:
                    shutil.copy2(p, dest)
    original_files = {str(p.relative_to(source)): sha(p)
                      for split in ('train', 'heldout')
                      for p in (source / split).rglob('*') if p.is_file()}
    for seed in range(n, args.train_episodes):
        episode, metadata = generate_episode(replace(cfg, seed=seed))
        save_episode_npy(out / 'train' / f'episode_{seed:06d}.npy', episode, metadata)
    certified = assert_disjoint(out / 'train', out / 'heldout')
    if certified['train'][:n] != original['train'] or certified['heldout'] != original['heldout']:
        raise ValueError('Original episode content was changed')
    for relative, expected in original_files.items():
        if sha(source / relative) != expected or sha(out / relative) != expected:
            raise ValueError('Original metadata or actions changed')
    mean, std, _, _ = compute_stats(str(out / 'train'))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid training-only statistics')
    np.savez(out / 'train_stats.npz', mean=mean, std=std)
    certified.update(protocol='joystick-lam-v5', generator_config=asdict(cfg),
                     statistics_source='train', train_episodes=args.train_episodes,
                     original_manifest_sha256=sha(source / 'split_manifest.json'),
                     original_file_hashes=original_files,
                     reused_train_episodes=n, added_train_seeds=list(range(n, args.train_episodes)),
                     original_source=args.source)
    (out / 'split_manifest.json').write_text(json.dumps(certified, indent=2)+'\n')
    return certified

if __name__ == '__main__':
    extend(tyro.cli(Args))
