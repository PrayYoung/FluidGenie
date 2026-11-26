"""Diagnose a saved LAM without training or changing its frozen certificate."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.training.checkpoint_utils import load_params
from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.training.train_lam import TrainState
from fluidgenie.data.dataset_npz import load_episode_actions
from scripts.evaluate_lam_discovery import batched_call, discovery_windows


def summarize_codes(labels, codes, mapping, num_codes):
    labels, codes = np.asarray(labels), np.asarray(codes)
    matrix = np.zeros((5,num_codes),dtype=int)
    np.add.at(matrix,(labels,codes),1)
    return {'transitions':len(labels),'confusion':matrix.tolist(),
            'frozen_mapping_accuracy':float(np.mean(np.asarray(mapping)[codes]==labels)),
            'oracle_majority_accuracy_diagnostic_only':float(matrix.max(0).sum()/len(labels))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('outputs/lam_v5'))
    parser.add_argument('--data',type=Path,default=Path('data/lam_v5/joystick'))
    parser.add_argument('--out',type=Path,default=Path('research/results/lam_v5/diagnostics'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((args.root/'lam/config.json').read_text())['config']
    cert=json.loads((args.root/'lam_discovery.json').read_text())
    with np.load(args.data/'train_stats.npz') as z:mean,std=z['mean'],z['std']
    def norm(x):return (np.asarray(x,dtype=np.float32)-mean)/(std+1e-6)
    model=LatentActionModel(in_dim=3,architecture_version=cfg['architecture_version'],**{
        k:cfg[k] for k in ['model_dim','latent_dim','num_latents','patch_size','num_blocks','num_heads','dropout','codebook_dropout']})
    seq=cfg['seq_len'];example=jnp.zeros((1,seq,64,64,3),dtype=jnp.float32)
    initial=model.init(jax.random.PRNGKey(0),{'videos':example},training=False)['params']
    params=load_params(str(args.root/'lam/latest'),initial)
    state=TrainState.create(apply_fn=model.apply,params=initial,tx=make_optimizer(cfg['lr'],cfg['steps']))
    template={'params':state.params,'opt_state':state.opt_state,'step':state.step,'rng':jax.random.PRNGKey(0),
              'auxiliary':{'code_usage':jnp.zeros((cfg['num_latents'],),dtype=jnp.int32)}}
    saved=load_params(str(args.root/'lam/train_state'),template)
    finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(saved))
    same=all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(saved['params'])))
    assert finite and same and int(saved['step'])==cfg['steps']
    @jax.jit
    def encode(video):
        return model.apply({'params':params},video,False,method=model.vq_encode)['indices'].reshape(len(video),video.shape[1]-1)
    rows=[];all_codes=[];all_labels=[];all_episode=[];lags={lag:[0,0] for lag in range(-2,3)}
    started=time.time()
    for split,indices in [('train',list(range(6))+list(range(30,36))),('heldout',list(range(10)))]:
        for index in indices:
            p=args.data/split/f'episode_{index:06d}.npy';episode=np.load(p,mmap_mode='r')[:64]
            videos=np.stack([norm(w) for _,w,_ in discovery_windows(episode,seq)])
            encoded=batched_call(encode,jnp.asarray(videos),batch_size=4)
            codes=np.concatenate([encoded[0],encoded[1:,-1]],axis=0)
            labels=load_episode_actions(p)[:len(codes)]
            row={'split':split,'episode':index,**summarize_codes(labels,codes,cert['code_to_action'],cfg['num_latents'])}
            rows.append(row)
            if split=='heldout':
                all_codes.extend(codes.tolist());all_labels.extend(labels.tolist());all_episode.extend([index]*len(codes))
                predicted=np.asarray(cert['code_to_action'])[codes]
                for lag in lags:
                    t=np.arange(max(0,-lag),min(len(codes),len(codes)-lag))
                    lags[lag][0]+=int((predicted[t]==labels[t+lag]).sum());lags[lag][1]+=len(t)
            print(json.dumps({'split':split,'episode':index,'elapsed_seconds':round(time.time()-started,1)}),flush=True)
    actual=summarize_codes(all_labels,all_codes,cert['code_to_action'],cfg['num_latents'])
    assert actual['confusion']==cert['test_confusion_matrix']
    assert abs(actual['frozen_mapping_accuracy']-cert['out_of_sample_purity'])<1e-12
    # Compare the same trained transition at each available causal context length.
    probe=np.stack([norm(np.load(args.data/'heldout'/f'episode_{i:06d}.npy',mmap_mode='r')[20:20+seq]) for i in range(10)])
    full=batched_call(encode,jnp.asarray(probe),batch_size=4)[:,-1]
    pair=batched_call(encode,jnp.asarray(probe[:,-2:]),batch_size=4)[:,-1]
    np.savez_compressed(args.out/'heldout_codes.npz',codes=np.asarray(all_codes),labels=np.asarray(all_labels),episode=np.asarray(all_episode))
    result={'checkpoint':{'training_step':int(saved['step']),'all_saved_arrays_finite':finite,'latest_matches_training_state':same},
            'heldout_certificate_exactly_reproduced':True,'heldout':actual,'sampled_train_episodes':rows[:12],
            'heldout_episodes':rows[12:],'label_shift_accuracy_diagnostic_only':{str(k):v[0]/v[1] for k,v in lags.items()},
            'same_transition_context_probe':{'samples':10,'fraction_codes_changed_between_pair_and_full_context':float(np.mean(full!=pair)),
                                             'full_codes':full.tolist(),'pair_codes':pair.tolist()},
            'limits':['No thresholds or mapping were changed. Oracle majority is a diagnostic ceiling, not a valid calibrated score.',
                      'Training episodes are sampled; heldout transitions fully reproduce the existing certificate.']}
    (args.out/'checkpoint_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    baseline=Path('research/results/repair_v4/completed_campaign/raw/lam_discovery.json')
    if baseline.exists():
        plot_alignment_comparison(json.loads(baseline.read_text()),cert,args.out/'alignment_comparison.png')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sampled_train_episodes','heldout_episodes']},indent=2))

def plot_alignment_comparison(previous, current, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels=['No-op','Up','Down','Right','Left']
    for ax,title,data in zip(axes,['Previous LAM','LAM v5'],[previous,current]):
        codes=np.asarray(data['test_confusion_matrix']);semantic=np.zeros((5,5))
        for code,action in enumerate(data['code_to_action']):semantic[:,action]+=codes[:,code]
        semantic/=semantic.sum(1,keepdims=True)
        im=ax.imshow(semantic,cmap='Blues',vmin=0,vmax=1)
        ax.set_xticks(range(5),labels);ax.set_yticks(range(5),labels)
        ax.set_xlabel('Calibration-decoded action');ax.set_ylabel('True action')
        ax.set_title(f"{title}: purity {data['out_of_sample_purity']:.1%}")
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{semantic[i,j]:.0%}',ha='center',va='center',color='white' if semantic[i,j]>.55 else '#172033')
    fig.colorbar(im,ax=axes,shrink=.8)
    fig.suptitle('Action alignment degraded in the LAM v5 revision')
    fig.supxlabel('Same 630 held-out transitions; calibration-only mappings. Both fail the required 80% purity.',fontsize=9)
    fig.savefig(out,dpi=150);plt.close(fig)


if __name__=='__main__':main()
