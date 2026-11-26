"""Small supervised CPU probes of action visibility; never a LAM benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np


def extract_split(root, split):
    groups = {key: [] for key in ('state', 'delta', 'labels', 'local_delta')}
    for path in sorted((root / split).glob('episode_*.npy')):
        fields = np.asarray(np.load(path, mmap_mode='r')[:64, :, :, :2])
        frames, height, width, _ = fields.shape
        if (frames, height, width) != (64, 64, 64):
            raise ValueError('This diagnostic is frozen to 64 frames at 64x64')
        pooled = fields.reshape(64, 8, 8, 8, 8, 2).mean(axis=(2, 4)).reshape(64, -1)
        coords = (np.arange(64) + .5) / 64
        distance = np.minimum(abs(coords - .5), 1 - abs(coords - .5))
        kernel = np.exp(-(distance[:, None]**2 + distance[None, :]**2) / (2 * .14**2))
        kernel -= kernel.mean()
        impulse = np.einsum('thwc,hw->tc', np.diff(fields, axis=0), kernel) / (kernel*kernel).sum()
        groups['state'].append(pooled[:-1])
        groups['delta'].append(np.diff(pooled, axis=0))
        groups['labels'].append(np.load(root / split / 'actions' / path.name)[:63])
        groups['local_delta'].append(impulse)
    return {key: np.concatenate(values) for key, values in groups.items()}


def scores(predicted, labels):
    confusion = np.zeros((5, 5), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    return {'heldout_accuracy': float(np.mean(predicted == labels)), 'confusion': confusion.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/lam_v5/joystick'))
    parser.add_argument('--out', type=Path, default=Path('research/results/lam_v5/diagnostics/physics_probe.json'))
    args = parser.parse_args()
    train, test = [extract_split(args.data, split) for split in ('train', 'heldout')]
    report = {'scope': 'Supervised read-only diagnostic on the existing 60/10 episode split, first 63 transitions. Not an unsupervised LAM score; no heldout tuning.', 'probes': {}}
    for key in ('state', 'delta'):
        x, y = train[key], test[key]
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
        x = np.column_stack([(x-mean)/std, np.ones(len(x))])
        y = np.column_stack([(y-mean)/std, np.ones(len(y))])
        penalty = np.eye(x.shape[1]) * .01
        penalty[-1, -1] = 0
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ np.eye(5)[train['labels']])
        report['probes'][key+'_ridge'] = {
            **scores((y @ weights).argmax(1), test['labels']),
            'train_accuracy': float(np.mean((x @ weights).argmax(1) == train['labels'])),
            'regularization': .01,
        }
    centers = np.stack([train['local_delta'][train['labels'] == action].mean(0) for action in range(5)])
    predicted = ((test['local_delta'][:, None] - centers[None])**2).sum(-1).argmin(1)
    report['probes']['local_delta_centroids'] = {**scores(predicted, test['labels']), 'centers': centers.tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value['heldout_accuracy'] for key, value in report['probes'].items()}))

if __name__ == '__main__':
    main()
