import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

path = "data/ns2d/episode_000008.npz"  # 改成你的文件
data = np.load(path, allow_pickle=True)
fields = data["fields"]  # [T,H,W,C]

u = fields[..., 0]
v = fields[..., 1]
speed = np.sqrt(u**2 + v**2)
has_density = fields.shape[-1] >= 3
density = fields[..., 2] if has_density else None

fig, axes = plt.subplots(1, 2 if has_density else 1, figsize=(8, 4))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

im0 = axes[0].imshow(speed[0], origin="lower", cmap="viridis")
axes[0].set_title("Speed |v|")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

if has_density:
    im1 = axes[1].imshow(density[0], origin="lower", cmap="magma")
    axes[1].set_title("Density")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

def update(t):
    im0.set_data(speed[t])
    axes[0].set_title(f"Speed |v| (t={t})")
    if has_density:
        im1.set_data(density[t])
        axes[1].set_title(f"Density (t={t})")
    return [im0] + ([im1] if has_density else [])

ani = FuncAnimation(fig, update, frames=fields.shape[0], interval=60, blit=False)
plt.tight_layout()
plt.show()
