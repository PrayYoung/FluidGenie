"""Export saved-model square commands with shared-scale density/vorticity panels."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fluidgenie.cli.steering_demo import SteeringSession
from fluidgenie.eval.showcase import square_frame, save_gif, vorticity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--stats", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--hold", type=int, default=16)
    p.add_argument("--arrays-out", default="")
    a = p.parse_args()
    if a.hold < 1 or a.hold > 22:
        p.error("--hold must be in 1..22 (at most 88 frames)")
    session = SteeringSession(a.checkpoint, a.episode, a.stats, seed=0, ode_steps=16)
    fields = []
    actions = []
    for action in [3, 1, 4, 2]:
        for _ in range(a.hold):
            fields.append(session.step(action))
            actions.append(action)
    fields = np.stack(fields)
    density_limit = max(float(fields[..., 2].max()), 1e-8)
    omega_limit = max(float(np.max(np.abs(vorticity(fields)))), 1e-8)
    frames = [
        square_frame(field, i + 1, action, a.hold, density_limit, omega_limit)
        for i, (field, action) in enumerate(zip(fields, actions))
    ]
    out = Path(a.out)
    meta = save_gif(frames, out)
    arrays = Path(a.arrays_out) if a.arrays_out else out.with_suffix(".npz")
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays, fields=fields, actions=np.asarray(actions), context=session.initial
    )
    meta.update(
        view="smoke_density_and_vorticity",
        density_vmax=density_limit,
        vorticity_symmetric_limit=omega_limit,
        actions=["Right", "Up", "Left", "Down"],
        steps_per_command=a.hold,
        source_arrays=str(arrays),
        source_arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        model_noise_seed=0,
        ode_steps=16,
        checkpoint=a.checkpoint,
        episode=a.episode,
        interpretation="Real saved-model rollout driven by a square command sequence; not proof of a square physical trajectory.",
    )
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
