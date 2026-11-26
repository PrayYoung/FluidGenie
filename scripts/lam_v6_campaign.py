"""Frozen v4 LAM architecture experiment on the unchanged v5 dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scripts.extend_joystick_data import verify_original
from scripts.lam_v5_campaign import gate_passed, stage_commands as previous_commands


def verify_frozen_data(data):
    saved=json.loads((data/'split_manifest.json').read_text())
    actual=verify_original(data,saved)
    if [r['seed'] for r in actual['train']]!=list(range(60)) or [r['seed'] for r in actual['heldout']]!=list(range(200003,200013)):
        raise ValueError('Expected frozen v5 train/heldout seeds')
    if not saved.get('disjoint') or saved.get('statistics_source')!='train':
        raise ValueError('Expected certified disjoint training-only statistics')
    from fluidgenie.data.compute_stats import compute_stats
    mean,std,_,_=compute_stats(str(data/'train'))
    with np.load(data/'train_stats.npz') as stats:
        np.testing.assert_array_equal(stats['mean'],mean)
        np.testing.assert_array_equal(stats['std'],std)
    return {'split_manifest_sha256':hashlib.sha256((data/'split_manifest.json').read_bytes()).hexdigest(),
            'stats_sha256':hashlib.sha256((data/'train_stats.npz').read_bytes()).hexdigest(),
            'train_episodes':60,'heldout_episodes':10,'disjoint':True}


def stage_commands(root,data,workers=4):
    commands=[]
    for name,args in previous_commands(root,data,workers):
        if name=='extend_joystick':continue
        if name=='lam':args=[*args,'--architecture-version','4']
        commands.append((name,args))
    return commands


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('outputs/lam_v6'))
    parser.add_argument('--data',type=Path,default=Path('data/lam_v5/joystick'))
    args=parser.parse_args();root=args.out
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a fresh experiment directory')
    frozen=verify_frozen_data(args.data)
    root.mkdir(parents=True,exist_ok=True)
    protocol={'experiment':'lam-v6-transition-single-position-v4','architecture_version':4,
              'frozen_data':frozen,'data':str(args.data),'no_regeneration':True,
              'continuous_benchmarks_reused':True,'gate_version':4,
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in ('fluidgenie','scripts','configs') for p in Path(folder).rglob('*.py')}}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    journal=[]
    for name,command in stage_commands(root,args.data):
        if name=='action_fm':
            discovery=json.loads((root/'lam_discovery.json').read_text());passed=gate_passed(discovery)
            (root/'downstream_gate.json').write_text(json.dumps({'state':'passed' if passed else 'scientific_gate_failed',
                'discovery_gate_passed':discovery['discovery_gate_passed'],
                'five_action_mapping_gate':discovery['five_action_mapping_gate'],
                'reason':'Frozen v4 certificate required; no retries on a negative result.'},indent=2)+'\n')
            if not passed:
                print('SCIENTIFIC_GATE_FAILED: no retries or downstream training',flush=True);return
        entry={'stage':name,'command':command,'state':'running','started_unix':time.time()}
        journal.append(entry);(root/'campaign_state.json').write_text(json.dumps(journal,indent=2)+'\n')
        with (root/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT)
        entry.update(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_unix=time.time())
        (root/'campaign_state.json').write_text(json.dumps(journal,indent=2)+'\n');print(json.dumps(entry),flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    print('LAM_V6_CAMPAIGN_COMPLETED',flush=True)

if __name__=='__main__':main()
