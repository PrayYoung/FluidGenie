"""One frozen LAM hypothesis; continuous-model baselines are never rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def stage_commands(root, data, workers=4):
    train, heldout, stats = str(data/'train'), str(data/'heldout'), str(data/'train_stats.npz')
    lam = root/'lam'
    fm = root/'action_fm'
    return [
        ('cuda_preflight', ['scripts/jax_cuda_preflight.py']),
        ('regression_tests', ['-m','pytest','-q']),
        ('extend_joystick', ['scripts/extend_joystick_data.py','--out',str(data)]),
        ('validate_train', ['scripts/validate_forced_data.py','--data',train,'--out',str(root/'train_quality.json'),'--no-require-stationarity']),
        ('validate_heldout', ['scripts/validate_forced_data.py','--data',heldout,'--out',str(root/'heldout_quality.json'),'--no-require-stationarity']),
        ('lam', ['-m','fluidgenie.training.train_lam','--data',train,'--stats',stats,'--out',str(lam),
                 '--batch','16','--steps','8000','--seq-len','3','--num-latents','8','--latent-dim','32',
                 '--model-dim','256','--num-blocks','4','--num-heads','8','--grain-workers',str(workers),
                 '--checkpoint-every','1000','--tb','0']),
        ('lam_discovery', ['scripts/evaluate_lam_discovery.py','--checkpoint',str(lam/'latest'),
                           '--calibration-dir',train,'--test-dir',heldout,'--stats',stats,
                           '--out',str(root/'lam_discovery.json'),'--frames','64']),
        ('action_fm', ['-m','fluidgenie.training.train_flow_matching','--data',train,'--stats',stats,
                       '--out',str(fm),'--steps','15000','--batch','32','--hidden-dims','64','128','256',
                       '--use-lam','--lam-ckpt',str(lam/'latest'),'--lam-discovery',str(root/'lam_discovery.json'),
                       '--grain-workers',str(workers),'--checkpoint-every','1000','--tb','0','--physics-dx',str(1/64)]),
        ('counterfactual', ['scripts/benchmark_counterfactual_steering.py','--checkpoint',str(fm/'latest'),
                           '--stats',stats,'--out',str(root/'counterfactual'),'--resolution','64','--horizon','40']),
        ('showcase', ['scripts/export_steering_showcase.py','--checkpoint',str(fm/'latest'),
                      '--episode',str(data/'heldout/episode_000000.npy'),'--stats',stats,'--out',str(root/'square_steering.gif')]),
    ]


def gate_passed(discovery):
    from fluidgenie.eval.action_discovery import require_discovery_certificate
    try:
        require_discovery_certificate(discovery)
    except ValueError:
        return False
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('outputs/lam_v5'))
    parser.add_argument('--data',type=Path,default=Path('data/lam_v5/joystick'))
    parser.add_argument('--grain-workers',type=int,default=4)
    args=parser.parse_args()
    root=args.out
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a fresh experiment directory; no automatic scientific retries')
    root.mkdir(parents=True,exist_ok=True)
    protocol={'experiment':'lam-v5-nyquist-budget-diversity','forced_control_reused':True,
              'gate_version':4,'heldout_seeds':list(range(200003,200013)),
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in ('fluidgenie','scripts','configs') for p in Path(folder).rglob('*.py')}}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    journal=[]
    for name,command in stage_commands(root,args.data,args.grain_workers):
        if name=='action_fm':
            discovery=json.loads((root/'lam_discovery.json').read_text())
            passed=gate_passed(discovery)
            (root/'downstream_gate.json').write_text(json.dumps({'state':'passed' if passed else 'scientific_gate_failed',
                'discovery_gate_passed':discovery['discovery_gate_passed'],
                'five_action_mapping_gate':discovery['five_action_mapping_gate'],
                'reason':'Frozen v4 gate; downstream stages run only on a full pass.'},indent=2)+'\n')
            if not passed:
                print('SCIENTIFIC_GATE_FAILED: no retries or conditioned training',flush=True)
                return
        entry={'stage':name,'command':command,'started_unix':time.time(),'state':'running'}
        journal.append(entry)
        (root/'campaign_state.json').write_text(json.dumps(journal,indent=2)+'\n')
        with (root/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT)
        entry.update(exit_code=result.returncode,state='completed' if result.returncode==0 else 'failed',finished_unix=time.time())
        (root/'campaign_state.json').write_text(json.dumps(journal,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    print('LAM_V5_CAMPAIGN_COMPLETED',flush=True)

if __name__=='__main__':
    main()
