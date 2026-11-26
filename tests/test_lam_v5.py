"""Nyquist-safe features, original-data preservation, resume, and frozen gates."""
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from fluidgenie.models.st_transformer import Periodic2DPositionalEncoding
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.training.train_lam import TrainState, train_step
from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.training.checkpoint_utils import save_training_state, restore_training_state
from fluidgenie.data.splits import assert_disjoint
from scripts import extend_joystick_data as extension
from scripts.lam_v5_campaign import gate_passed, stage_commands


@pytest.mark.parametrize('shape,width,prefix', [((16,16),256,1),((6,10),13,2),((1,1),8,0)])
def test_periodic_features_are_projected_and_band_limited(shape,width,prefix):
    h,w=shape
    model=Periodic2DPositionalEncoding(width,shape,prefix)
    x=jnp.zeros((1,2,h*w+prefix,width))
    params=model.init(jax.random.PRNGKey(0),x)
    assert params['params']['fourier_projection']['kernel'].shape==(4*min(8,min(h,w)//2),width)
    result=np.asarray(model.apply(params,x))[0,0]
    assert np.isfinite(result).all() and result.shape==(h*w+prefix,width)
    spectrum=np.abs(np.fft.fft2(result[prefix:].reshape(h,w,width),axes=(0,1)))
    fy,fx=np.meshgrid(np.rint(np.fft.fftfreq(h)*h),np.rint(np.fft.fftfreq(w)*w),indexing='ij')
    k=min(8,min(h,w)//2)
    allowed=((fy==0)&(np.abs(fx)<=k))|((fx==0)&(np.abs(fy)<=k))
    assert np.max(spectrum[~allowed],initial=0)<1e-3
    if prefix:
        np.testing.assert_allclose(result[:prefix],params['params']['prefix_embed'])
        assert np.any(result[:prefix]!=0)
        grads=jax.grad(lambda p:jnp.sum(model.apply({'params':p},x)[:,:,:prefix]))(params['params'])
        assert np.all(np.asarray(grads['prefix_embed'])==2)


def test_legacy_lam_has_no_new_encoding_parameters():
    model=LatentActionModel(3,8,4,8,2,1,2,0.,0.,architecture_version=2)
    params=model.init(jax.random.PRNGKey(0),{'videos':jnp.zeros((1,3,4,4,3))},training=False)
    assert 'fourier_projection' not in str(jax.tree.structure(params))
    assert np.isfinite(np.asarray(model.apply(params,{'videos':jnp.zeros((1,3,4,4,3))},training=False)['recon'])).all()


def test_lam_resume_restores_rng_optimizer_step_and_restart_usage(tmp_path):
    model=LatentActionModel(3,8,4,8,2,1,2,0.,0.)
    batch=jax.random.normal(jax.random.PRNGKey(1),(1,3,4,4,3))
    params=model.init(jax.random.PRNGKey(0),{'videos':batch},training=False)['params']
    initial=TrainState.create(apply_fn=model.apply,params=params,tx=make_optimizer(3e-4,4))
    key=jax.random.PRNGKey(2)
    key,step_key=jax.random.split(key)
    state,metrics=train_step(initial,batch,.25,step_key)
    usage=metrics['code_usage']
    save_training_state(tmp_path,state,key,auxiliary={'code_usage':usage})
    iterator=iter(range(10))
    restored,rng,extra=restore_training_state(tmp_path,initial,jax.random.PRNGKey(99),iterator,
                                              auxiliary={'code_usage':jnp.zeros_like(usage)})
    assert next(iterator)==1
    np.testing.assert_array_equal(extra['code_usage'],usage)
    np.testing.assert_array_equal(rng,key)
    key,next_key=jax.random.split(key)
    expected,_=train_step(state,batch,.25,next_key)
    _,next_key=jax.random.split(rng)
    actual,_=train_step(restored,batch,.25,next_key)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        np.testing.assert_array_equal(a,b)


def test_extension_reuses_originals_and_training_only_statistics(tmp_path,monkeypatch):
    from configs.gen_data_configs import NS2DConfig
    from dataclasses import asdict,replace
    from fluidgenie.data.gen_phiflow_ns2d import save_episode_npy
    src=tmp_path/'old';out=tmp_path/'new';cfg=NS2DConfig(resolution=4,steps=3,action_mode='joystick5_localized')
    def generate(c):
        return np.full((3,4,4,3),c.seed+1,dtype=np.float32),{**asdict(c),'actions':[0,1]}
    for split,seeds in [('train',range(2)),('heldout',[200003])]:
        for i,seed in enumerate(seeds):
            x,m=generate(replace(cfg,seed=seed));save_episode_npy(src/split/f'episode_{i:06d}.npy',x,m)
    saved=assert_disjoint(src/'train',src/'heldout');saved['generator_config']=asdict(cfg)
    (src/'split_manifest.json').write_text(json.dumps(saved))
    monkeypatch.setattr(extension,'generate_episode',generate)
    result=extension.extend(extension.Args(str(src),str(out),4))
    assert result['heldout']==saved['heldout'] and result['train'][:2]==saved['train']
    assert result['added_train_seeds']==[2,3]
    with np.load(out/'train_stats.npz') as stats:
        np.testing.assert_allclose(stats['mean'],2.5)
        np.testing.assert_allclose(stats['std'],np.std([1,2,3,4]))
        assert set(stats.files)=={'mean','std'}


def test_targeted_campaign_never_runs_continuous_benchmarks_and_fails_closed():
    commands=stage_commands(Path('outputs/test'),Path('data/test'))
    assert [name for name,_ in commands]==['cuda_preflight','regression_tests','extend_joystick','validate_train','validate_heldout','lam','lam_discovery','action_fm','counterfactual','showcase']
    assert not gate_passed({'discovery_gate_passed':True,'five_action_mapping_gate':False})
    certificate={'discovery_gate_passed':True,'five_action_mapping_gate':True,
                 'gate':{'version':4},'shuffle_protocol':'episode_wide_transition_permutation_v4'}
    assert gate_passed(certificate)
    certificate['gate']['version']=3
    assert not gate_passed(certificate)


def test_delivery_excludes_redundant_checkpoints_and_original_inputs(tmp_path):
    from scripts.collect_lam_v5_artifacts import collect
    files=['outputs/lam_v5/lam/notes.txt','outputs/lam_v5/lam/lam_metrics.jsonl',
           'outputs/lam_v5/lam/final/params','outputs/lam_v5/lam/step_001000/params',
           'outputs/lam_v5/lam/config.json','outputs/lam_v5/lam_discovery.json',
           'data/lam_v5/joystick/train/episode_000030.npy',
           'data/lam_v5/joystick/train/episode_000000.npy',
           'data/lam_v5/joystick/heldout/episode_000000.npy']
    for relative in files:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'preserved')
    from fluidgenie.training.checkpoint_utils import save_params, load_params
    params={'weights':jnp.arange(3,dtype=jnp.float32)}
    save_params(tmp_path/'outputs/lam_v5/lam','latest',params)
    save_params(tmp_path/'outputs/lam_v5/lam','train_state',{'params':params,'step':jnp.asarray(1)})
    report=collect(tmp_path)
    compact=load_params(str(tmp_path/'delivery/lam_v5/outputs/lam_v5/lam/latest'),params)
    np.testing.assert_array_equal(compact['weights'],params['weights'])
    retained={r['path'] for r in report['files']}
    assert files[0] in retained and files[1] in retained and files[5] in retained and files[6] in retained
    assert all(p not in retained for p in [files[2],files[3],files[7],files[8]])
    assert all((tmp_path/p).read_bytes()==b'preserved' for p in files)
