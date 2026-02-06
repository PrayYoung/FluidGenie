import os, glob
import numpy as np

class NPZSequenceDataset:
    def __init__(self, data_dir: str, context: int = 2, pred: int = 1, stats_path: str | None = None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert self.files, f"No npz found in {data_dir}"
        self.context = context
        self.pred = pred
        self.mean = None
        self.std = None
        if stats_path:
            stats = np.load(stats_path)
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

        # build an index mapping (file_idx, start_t)
        self.index = []
        for i, f in enumerate(self.files):
            arr = np.load(f)["fields"]
            T = arr.shape[0]
            # need context frames + pred frames
            for t in range(0, T - (context + pred) + 1):
                self.index.append((i, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        file_idx, t = self.index[idx]
        data = np.load(self.files[file_idx])
        fields = data["fields"]  # [T,H,W,C]
        x = fields[t : t + self.context]          # [context,H,W,C]
        y = fields[t + self.context : t + self.context + self.pred]  # [pred,H,W,C]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)
            y = (y - self.mean) / (self.std + 1e-6)
        return x, y
