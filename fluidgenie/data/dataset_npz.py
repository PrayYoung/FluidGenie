import os
import glob
import grain
from typing import Tuple

import numpy as np
from jaxtyping import Array, Float

class NPZSequenceDataset:
    def __init__(self, data_dir: str, context: int = 2, pred: int = 1, stats_path: str | None = None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        assert self.files, f"No npy found in {data_dir}"
        self.context = context
        self.pred = pred
        self._memmaps: list[np.ndarray] = []
        self.mean = None
        self.std = None
        if stats_path:
            stats = np.load(stats_path)
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

        # build an index mapping (file_idx, start_t)
        self.index = []
        for i, f in enumerate(self.files):
            try:
                data = np.load(f, mmap_mode="r")
                self._memmaps.append(data)
                T = data.shape[0]
            except Exception:
                raise ValueError(f"Error loading {f}. Ensure it is a .npy array with shape [T,H,W,C].")
            # need context frames + pred frames
            for t in range(0, T - (context + pred) + 1):
                self.index.append((i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Float[Array, "context h w c"], Float[Array, "pred h w c"]]:
        file_idx, t = self.index[idx]
        fields = self._memmaps[file_idx]
        x = fields[t : t + self.context]          # [context,H,W,C]
        y = fields[t + self.context : t + self.context + self.pred]  # [pred,H,W,C]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)
            y = (y - self.mean) / (self.std + 1e-6)
        return x, y


def create_grain_dataloader(
    data_dir: str,
    batch_size: int,
    context: int = 2,
    seed: int = 0,
    num_workers: int = 4,
    stats_path: str | None = None,
    worker_buffer_size: int = 4,
) -> grain.DataLoader:
    source = NPZSequenceDataset(data_dir, context=context, pred=1, stats_path=stats_path)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=seed,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        num_epochs=None,
    )

    operations = [
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=True),
    ]
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=worker_buffer_size,
    )
    return loader
