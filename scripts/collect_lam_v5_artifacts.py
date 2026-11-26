"""Build a small, hashed delivery bundle without intermediate checkpoint copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def collect(project=Path('.'), *, campaign='lam_v5', include_new_data=True):
    output = project / 'outputs' / campaign
    data = project / 'data/lam_v5/joystick'
    delivery = project / 'delivery' / campaign
    if not output.is_dir():
        return None
    if delivery.exists():
        shutil.rmtree(delivery)
    records = []

    def record(path, relative):
        records.append({"path":str(relative), "bytes":path.stat().st_size,
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})

    def retain(path):
        relative = path.relative_to(project)
        target = delivery / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        record(target, relative)

    # A fresh Orbax save retains only referenced arrays, excluding stale OCDBT
    # blocks left by periodic overwrites. Restore and save the same tensor tree.
    from orbax.checkpoint import PyTreeCheckpointer
    import jax
    import numpy as np
    for model in ('lam', 'action_fm'):
        for name in ('latest', 'train_state'):
            source = output / model / name
            if not source.is_dir():
                continue
            target = delivery / source.relative_to(project)
            tree = PyTreeCheckpointer().restore(source.resolve())
            PyTreeCheckpointer().save(target.resolve(), tree)
            restored = PyTreeCheckpointer().restore(target.resolve())
            if jax.tree.structure(tree) != jax.tree.structure(restored):
                raise ValueError('Compacted checkpoint structure changed')
            for a, b in zip(jax.tree.leaves(tree), jax.tree.leaves(restored)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            for p in target.rglob('*'):
                if p.is_file():
                    record(p, p.relative_to(delivery))
    for p in output.rglob('*'):
        if not p.is_file():
            continue
        relative = p.relative_to(output)
        # Final == latest. Keep one evaluation checkpoint and one optimizer/RNG
        # state per model, rather than all 1000-step snapshots.
        if len(relative.parts) >= 2 and relative.parts[0] in ('lam', 'action_fm') and relative.parts[1] in ('latest', 'train_state'):
            continue
        if any(part == 'final' or part.startswith('step_') or part.startswith('.')
               for part in relative.parts):
            continue
        retain(p)
    if include_new_data and data.is_dir():
        for name in ('train_stats.npz', 'split_manifest.json'):
            if (data / name).is_file():
                retain(data / name)
        for seed in range(30,60):
            for relative in (f'train/episode_{seed:06d}.npy',f'train/episode_{seed:06d}.json',
                             f'train/actions/episode_{seed:06d}.npy'):
                p=data/relative
                if p.is_file():
                    retain(p)
    report={'files':records,'logical_bytes':sum(r['bytes'] for r in records),
            'original_inputs_reused':('data/repair_v4/joystick; seeds 0..29 and heldout unchanged' if include_new_data else 'data/lam_v5/joystick; all 60/10 episodes reused byte for byte'),
            'policy':'Keep reports/media, latest evaluation parameters and resumable training state. Exclude intermediate/final checkpoint copies and reused inputs.',
            'includes_new_inputs':include_new_data}
    (delivery/'delivery_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path('.'))
    parser.add_argument('--campaign',default='lam_v5')
    parser.add_argument('--reuse-data',action='store_true')
    args=parser.parse_args()
    report=collect(args.project,campaign=args.campaign,include_new_data=not args.reuse_data)
    print(json.dumps({'delivery_bytes':report['logical_bytes'] if report else 0}))
