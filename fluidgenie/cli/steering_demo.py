"""Local WASD demo for a trained, five-action conditional flow checkpoint."""

from __future__ import annotations
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import jax
import jax.numpy as jnp
import numpy as np
from configs.config_io import load_config_json
from fluidgenie.models.lam_conditioning import load_frozen_lam
from fluidgenie.models.flow_matching import ConditionalFlowUNet
from fluidgenie.training.checkpoint_utils import load_params
from fluidgenie.eval.flow_matching_rollout import _load_stats, _norm, _denorm
from fluidgenie.eval.samplers.flow_matching import sample_flow_matching_heun


class SteeringSession:
    def __init__(self, checkpoint, episode, stats, seed=0, ode_steps=16):
        saved = load_config_json(checkpoint)
        if not saved or not (
            saved["config"].get("action_dim") == 5 or saved["config"].get("use_lam")
        ):
            raise ValueError("A trained joystick5 flow checkpoint is required")
        cfg = saved["config"]
        fields = np.load(episode, mmap_mode="r")
        from fluidgenie.eval.utils import resolve_stats_path

        self.stats = _load_stats(resolve_stats_path(stats, checkpoint))
        self.initial = np.asarray(fields[: cfg["context"]], dtype=np.float32)
        _, h, w, c = self.initial.shape
        self.model = ConditionalFlowUNet(
            c,
            context_frames=cfg["context"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            time_dim=cfg["time_dim"],
            num_res_blocks=cfg["num_res_blocks"],
            modern=cfg.get("modern", False),
            action_dim=cfg["action_dim"],
        )
        ctx = jnp.asarray(_norm(self.initial, self.stats)[None], dtype=jnp.float32)
        self.action_vectors = jnp.eye(5, dtype=jnp.float32)
        if cfg.get("use_lam"):
            calibration = Path(checkpoint).parent / "action_calibration.json"
            if not calibration.exists():
                calibration = Path(checkpoint) / "action_calibration.json"
            if not calibration.exists():
                raise ValueError(
                    "A discovery-certified action_calibration.json is required"
                )
            mapping = json.loads(calibration.read_text())
            if not mapping.get("discovery_gate_passed") or not mapping.get(
                "five_action_mapping_gate"
            ):
                raise ValueError("Uncertified LAM action mapping")
            from fluidgenie.eval.action_discovery import require_discovery_certificate

            require_discovery_certificate(mapping)
            lam, lam_params, _, _ = load_frozen_lam(cfg["lam_ckpt"], ctx)
            self.action_vectors = lam.apply(
                {"params": lam_params},
                jnp.asarray(mapping["calibrated_code_by_action"]),
                method=lambda module, ids: module.vq.get_codes(ids),
            ).astype(jnp.float32)
        action = self.action_vectors[:1]
        init = self.model.init(
            jax.random.PRNGKey(seed),
            ctx[:, -1],
            jnp.zeros(1, dtype=jnp.float32),
            ctx,
            training=False,
            action_cond=action,
        )["params"]
        self.params = load_params(checkpoint, init)
        self.seed = seed
        self.sample = jax.jit(
            lambda context, key, action: sample_flow_matching_heun(
                self.model, self.params, context, key, ode_steps, action
            )
        )
        self.reset()

    def reset(self, initial=None):
        if initial is not None:
            initial = np.asarray(initial, dtype=np.float32)
            if initial.shape != self.initial.shape or not np.isfinite(initial).all():
                raise ValueError("New context must match the model grid and be finite")
            self.initial = initial.copy()
        self.context = jnp.asarray(
            _norm(self.initial, self.stats)[None], dtype=jnp.float32
        )
        self.key = jax.random.PRNGKey(self.seed)
        self.steps = 0
        return _denorm(np.asarray(self.context[:, -1]), self.stats)[0]

    def step(self, action):
        if action not in range(5):
            raise ValueError("Action must be a joystick5 ID")
        self.key, key = jax.random.split(self.key)
        next_state = self.sample(
            self.context, key, self.action_vectors[action : action + 1]
        )
        if not bool(jnp.isfinite(next_state).all()):
            raise RuntimeError("Model produced a nonfinite field; reset required")
        self.context = jnp.concatenate(
            [self.context[:, 1:], next_state[:, None]], axis=1
        )
        self.steps += 1
        return _denorm(np.asarray(next_state), self.stats)[0]


def make_handler(session, page):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            if self.path not in {"/step", "/reset"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 256:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                with lock:
                    field = (
                        session.reset()
                        if self.path == "/reset"
                        else session.step(int(body["action"]))
                    )
                    payload = {
                        "density": field[..., 2].tolist(),
                        "speed": np.linalg.norm(field[..., :2], axis=-1).tolist(),
                        "vorticity": (
                            (
                                np.roll(field[..., 1], -1, axis=1)
                                - np.roll(field[..., 1], 1, axis=1)
                            )
                            * (field.shape[1] / 2)
                            - (
                                np.roll(field[..., 0], -1, axis=0)
                                - np.roll(field[..., 0], 1, axis=0)
                            )
                            * (field.shape[0] / 2)
                        ).tolist(),
                        "step": session.steps,
                    }
                data = json.dumps(payload).encode()
                self.send_response(200)
            except (ValueError, TypeError, RuntimeError, KeyError) as error:
                data = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--stats", default="")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    session = SteeringSession(args.checkpoint, args.episode, args.stats)
    page = Path(__file__).with_name("steering_demo.html").read_bytes()

    Handler = make_handler(session, page)
    print(f"Open http://127.0.0.1:{args.port} — WASD/arrows steer; Space is noop")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
