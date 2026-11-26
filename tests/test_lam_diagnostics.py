"""Keep diagnostic evidence distinct from frozen discovery decisions."""
import numpy as np
import pytest
from fluidgenie.eval.action_discovery import discovery_metrics
from scripts.evaluate_lam_discovery import save_transition_evidence


def test_calibration_evidence_does_not_replace_heldout_mapping_score():
    actions=np.repeat(np.arange(5),2)
    codes=np.asarray([0,1,1,1,2,2,3,3,4,4])
    result=discovery_metrics(actions,codes,np.arange(5),np.arange(5),5,.09,.2,.2)
    assert result['calibration_mapped_purity']==.9
    assert result['calibration_confusion_matrix'][0][:2]==[1,1]
    assert result['out_of_sample_purity']==1
    assert result['discovery_gate_passed']
    assert result['gate']['version']==4 and result['gate']['purity_min']==.8


def test_transition_evidence_preserves_episode_and_transition_alignment(tmp_path):
    columns={'actions':[4,1,0,2,3], 'codes':[7,2,1,6,0],
             'episode_ids':[0,0,0,1,1], 'transition_indices':[0,1,2,0,1]}
    path=save_transition_evidence(tmp_path/'discovery.json',{'test':columns})
    with np.load(path) as arrays:
        for name,values in columns.items():
            np.testing.assert_array_equal(arrays['test_'+name],values)
        assert arrays['test_codes'].dtype==np.int32
    with pytest.raises(ValueError,match='equal lengths'):
        save_transition_evidence(tmp_path/'bad.json',{'test':{'codes':[0,1],'actions':[0]}})
