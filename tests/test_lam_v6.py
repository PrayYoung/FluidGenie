"""Version-4 transition indexing, invariance, causality and checkpoint compatibility."""
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from configs.model_configs import LAMConfig
from configs.config_io import save_config_json
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.models.lam_conditioning import load_frozen_lam
from fluidgenie.training.initialization import initialization_rngs
from fluidgenie.training.train_lam import TrainState, train_step
from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.training.checkpoint_utils import save_params, save_training_state, restore_training_state


def model(version=4, dropout=0.):
    return LatentActionModel(3,16,8,8,2,2,2,dropout,dropout,architecture_version=version)


def video():
    # Exactly representable values avoid cancellation roundoff being mistaken
    # for failure of mathematical appearance-offset invariance.
    return jax.random.randint(jax.random.PRNGKey(10),(2,4,8,8,3),-16,17).astype(jnp.float32)/16


def encoded(m,p,x):
    return m.apply({'params':p},x,False,method=m.vq_encode)


def test_difference_encoder_offset_invariance_and_transition_count():
    m=model();x=video();p=m.init(jax.random.PRNGKey(0),{'videos':x},False)['params']
    offset=jax.random.randint(jax.random.PRNGKey(11),(2,1,8,8,3),-8,9).astype(jnp.float32)/8
    a,b=encoded(m,p,x),encoded(m,p,x+offset)
    assert a['indices'].shape==(2*3,) and a['z_q'].shape==(2,3,1,8)
    np.testing.assert_array_equal(a['indices'],b['indices'])
    np.testing.assert_allclose(a['z_q'],b['z_q'],atol=1e-5,rtol=0)
    # The decoder keeps original physical-state patches, rather than deltas.
    from fluidgenie.data.preprocess import patchify
    np.testing.assert_array_equal(a['patches'],patchify(x,2))
    pair=encoded(m,p,x[:,:2])
    assert pair['z_q'].shape==(2,1,1,8)


def test_difference_encoder_strict_temporal_causality():
    m=model();x=video();p=m.init(jax.random.PRNGKey(0),{'videos':x},False)['params']
    original=encoded(m,p,x)
    for future_start in (2,3):
        changed=encoded(m,p,x.at[:,future_start:].add(2))
        count=future_start-1
        np.testing.assert_array_equal(original['indices'].reshape(2,3)[:,:count],changed['indices'].reshape(2,3)[:,:count])
        np.testing.assert_allclose(original['emb'].reshape(2,3,8)[:,:count],changed['emb'].reshape(2,3,8)[:,:count],atol=1e-6)
    # Deterministic temporal attention has no absolute temporal embeddings.
    shifted=encoded(m,p,jnp.concatenate([jnp.repeat(x[:,:1],2,axis=1),x],axis=1))
    # Causality, not context-independence, is required: extra history may matter.
    assert shifted['z_q'].shape[1]==5


def test_v4_position_is_applied_once_and_zero_initialized_output():
    m=model();x=video();variables=m.init(jax.random.PRNGKey(0),{'videos':x},False)
    for part in ('encoder','decoder'):
        params=variables['params'][part]
        assert 'spatial_position' in params and 'output_norm' in params
        keys=str(jax.tree.structure(params))
        assert keys.count('fourier_projection')==1
        assert 'pos_emb' not in keys and 'PositionalEncoding' not in keys
        for key,value in params.items():
            if key.startswith('STBlock'):
                assert 'spatial_position' not in value
    result=m.apply(variables,{'videos':x},False)
    assert result['recon'].shape==(2,3,8,8,3)
    np.testing.assert_array_equal(result['recon'],np.zeros_like(result['recon']))


def test_v4_dropout_finite_training_and_resume_roundtrip(tmp_path):
    m=model(dropout=.1);x=video()
    p=m.init(initialization_rngs(jax.random.PRNGKey(0)),{'videos':x},True)['params']
    initial=TrainState.create(apply_fn=m.apply,params=p,tx=make_optimizer(3e-4,4))
    rng=jax.random.PRNGKey(2);rng,key=jax.random.split(rng)
    state,metrics=train_step(initial,x,.25,key)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((state.params,metrics)))
    usage=metrics['code_usage'];save_training_state(tmp_path,state,rng,auxiliary={'code_usage':usage})
    restored,loaded_rng,aux=restore_training_state(tmp_path,initial,jax.random.PRNGKey(0),iter(range(10)),auxiliary={'code_usage':jnp.zeros_like(usage)})
    np.testing.assert_array_equal(aux['code_usage'],usage)
    np.testing.assert_array_equal(loaded_rng,rng)
    # Serialization must preserve every saved value exactly. Subsequent CUDA
    # arithmetic may differ at the last bits after a separate JIT compilation.
    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):
        assert np.asarray(a).dtype==np.asarray(b).dtype
        np.testing.assert_array_equal(a,b)
    _,key=jax.random.split(rng);expected,_=train_step(state,x,.25,key)
    _,key=jax.random.split(loaded_rng);actual,_=train_step(restored,x,.25,key)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        a,b=np.asarray(a),np.asarray(b)
        if np.issubdtype(a.dtype,np.inexact):
            rtol,atol=(1e-12,1e-14) if a.dtype.itemsize>=8 else (1e-6,1e-8)
            np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)
        else:
            np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('version',[2,3,4])
def test_versioned_lam_checkpoint_loading_preserves_codes(tmp_path,version):
    m=model(version);x=video();p=m.init(jax.random.PRNGKey(0),{'videos':x},False)['params']
    cfg=LAMConfig(data='unused',out=str(tmp_path),architecture_version=version,seq_len=4,
                  model_dim=16,latent_dim=8,num_latents=8,patch_size=2,num_blocks=2,num_heads=2)
    save_config_json(tmp_path,{'kind':'lam','config':asdict(cfg)})
    save_params(tmp_path,'latest',p)
    loaded,params,encode,config=load_frozen_lam(str(tmp_path/'latest'),x)
    np.testing.assert_array_equal(encoded(m,p,x)['indices'],encoded(loaded,params,x)['indices'])
    np.testing.assert_allclose(encode(x),encoded(m,p,x)['z_q'][:,-1,0],rtol=1e-6,atol=1e-6)
    assert config['architecture_version']==version


def test_v6_reuses_data_and_only_changes_the_lam_architecture():
    from pathlib import Path
    from scripts.lam_v6_campaign import stage_commands
    commands=stage_commands(Path('outputs/lam_v6'),Path('data/lam_v5/joystick'))
    assert [name for name,_ in commands]==['cuda_preflight','regression_tests','validate_train','validate_heldout','lam','lam_discovery','action_fm','counterfactual','showcase']
    args=dict(commands)['lam']
    for key,value in [('--batch','16'),('--steps','8000'),('--seq-len','3'),('--architecture-version','4')]:
        assert args[args.index(key)+1]==value
    assert not any('forced_control' in arg or 'generate_disjoint_data' in arg for _,cmd in commands for arg in cmd)


def test_v6_delivery_reuses_all_input_data(tmp_path):
    from scripts.collect_lam_v5_artifacts import collect
    for name in ['outputs/lam_v6/lam/config.json','data/lam_v5/joystick/train_stats.npz',
                 'data/lam_v5/joystick/train/episode_000030.npy']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'preserve')
    result=collect(tmp_path,campaign='lam_v6',include_new_data=False)
    assert not result['includes_new_inputs']
    assert [r['path'] for r in result['files']]==['outputs/lam_v6/lam/config.json']


def test_shared_counterfactual_prefix_and_action_directions(monkeypatch):
    from configs.gen_data_configs import NS2DConfig
    import fluidgenie.data.gen_phiflow_ns2d as sim

    cfg = NS2DConfig(
        resolution=16,
        steps=6,
        burn_in=3,
        substeps=1,
        dt=0.01,
        action_mode="joystick5_localized",
        action_start=1,
        init_velocity_noise=0,
        forcing_strength=0,
        viscosity=0,
        action_strength=1.0,
        with_density=True,
    )
    calls = []
    original = sim.init_state

    def tracked(config):
        calls.append(config.seed)
        return original(config)

    monkeypatch.setattr(sim, "init_state", tracked)
    initialize = sim._initialize_simulation
    snapshots = []

    def capture(config):
        state = initialize(config)
        snapshots.append(state)
        return state

    monkeypatch.setattr(sim, "_initialize_simulation", capture)
    branches = sim.generate_counterfactual_branches(cfg)
    assert len(calls) == 1
    initial = branches[0][0][:2]
    kernel = sim.localized_jet_kernel(16, cfg.action_radius)
    directions = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    for action, (array, metadata) in branches.items():
        np.testing.assert_array_equal(array[:2], initial)
        assert array.shape == (6, 16, 16, 3) and np.isfinite(array).all()
        assert metadata["actions"][:2] == [0, action]
        if action:
            delta = array[2, ..., :2] - branches[0][0][2, ..., :2]
            response = np.einsum("hwc,hw->c", delta, kernel)
            assert np.dot(response, directions[action]) > 0
            assert not np.array_equal(array[2:], branches[0][0][2:])

    # Reuse this exact initialized state, avoiding a new random initialization.
    # Reversing execution must not let any branch mutate another branch's fork.
    monkeypatch.setattr(sim, "_initialize_simulation", lambda config: snapshots[0])
    reversed_branches = sim.generate_counterfactual_branches(cfg, reversed(range(5)))
    for action in range(5):
        np.testing.assert_array_equal(branches[action][0], reversed_branches[action][0])
    assert len(calls) == 1


def test_counterfactual_benchmark_tiny_mock_runtime(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from configs.gen_data_configs import NS2DConfig
    from configs.config_io import save_config_json
    import scripts.benchmark_counterfactual_steering as bench

    cfg = NS2DConfig(
        resolution=8,
        steps=5,
        burn_in=0,
        substeps=1,
        dt=0.01,
        viscosity=0,
        init_velocity_noise=0,
        forcing_strength=0,
        action_mode="joystick5_localized",
    )
    data = tmp_path / "train"
    data.mkdir()
    (data / "episode_000000.json").write_text(json.dumps(asdict(cfg)))
    ckpt = tmp_path / "model"
    ckpt.mkdir()
    save_config_json(ckpt, {"config": {"data": "unavailable_remote_path"}})

    class Session:
        def __init__(self, checkpoint, episode, stats, **kwargs):
            self.initial = np.load(episode)
            self.reset()

        def reset(self, initial=None):
            if initial is not None:
                self.initial = initial
            self.steps = 0

        def step(self, action):
            self.steps += 1
            return self.initial[-1].copy()

    monkeypatch.setattr(bench, "SteeringSession", Session)
    out = tmp_path / "out"
    bench.main(
        bench.Args(
            str(ckpt),
            "",
            str(out),
            horizon=3,
            resolution=8,
            burn_in=0,
            data=str(data),
            seeds=(200003, 200004),
        )
    )
    result = json.loads((out / "counterfactual_summary.json").read_text())
    assert result["same_initial_context"] and result["shared_prefix_max_abs_diff"] == 0
    assert (
        result["evaluation_seeds"] == [200003, 200004] and len(result["episodes"]) == 2
    )
    with np.load(out / "counterfactual_arrays.npz") as z:
        assert z["gt"].shape == z["pred"].shape == (5, 3, 8, 8, 3)
        assert z["context"].shape == (2, 8, 8, 3)
