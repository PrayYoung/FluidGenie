import json

import numpy as np
import pytest

from fluidgenie.data.splits import assert_disjoint
from scripts.repair_campaign import verify_reusable_data


@pytest.mark.parametrize("corruption", ["episode", "actions"])
def test_reuse_rejects_changes_to_certified_episode_or_actions(tmp_path, corruption):
    for split, seed in [("train", 0), ("heldout", 200003)]:
        directory = tmp_path / split
        (directory / "actions").mkdir(parents=True)
        np.save(directory / "episode_000000.npy", np.full((3, 2, 2, 3), seed, np.float32))
        np.save(directory / "actions/episode_000000.npy", np.array([0, 1, 2]))
        (directory / "episode_000000.json").write_text(
            json.dumps({"seed": seed, "actions": [0, 1, 2]})
        )
    manifest = assert_disjoint(tmp_path / "train", tmp_path / "heldout")
    manifest.update(generator_config={"resolution": 2, "steps": 3,
                                     "action_mode": "joystick5_localized"},
                    statistics_source="train")
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest))
    np.savez(tmp_path / "train_stats.npz", mean=np.zeros(3), std=np.ones(3))
    args = (tmp_path, 2, 3, 1, 1, "joystick5_localized")
    assert verify_reusable_data(*args)["disjoint"]
    if corruption == "episode":
        np.save(tmp_path / "train/episode_000000.npy", np.ones((3, 2, 2, 3)))
    else:
        np.save(tmp_path / "train/actions/episode_000000.npy", np.array([4, 4, 4]))
    with pytest.raises(ValueError, match="differ"):
        verify_reusable_data(*args)
