"""LAM discovery metrics with a calibration/test separation and ablation gate."""

import numpy as np


def _contingency(actions, codes, num_codes):
    actions, codes = np.asarray(actions, dtype=int), np.asarray(codes, dtype=int)
    if (
        actions.shape != codes.shape
        or not actions.size
        or np.any(actions < 0)
        or np.any(actions >= 5)
        or np.any(codes < 0)
        or np.any(codes >= num_codes)
    ):
        raise ValueError("Invalid joystick labels or latent code indices")
    table = np.zeros((5, num_codes), dtype=float)
    np.add.at(table, (actions, codes), 1)
    return table


def discovery_metrics(
    calibration_actions,
    calibration_codes,
    test_actions,
    test_codes,
    num_codes,
    conditioned_mse,
    zero_mse,
    shuffled_mse,
):
    test_actions = np.asarray(test_actions, dtype=int)
    calibration = _contingency(calibration_actions, calibration_codes, num_codes)
    test = _contingency(test_actions, test_codes, num_codes)
    mapping = calibration.argmax(axis=0)
    known = calibration.sum(axis=0) > 0
    test_codes = np.asarray(test_codes, dtype=int)
    purity = float(np.mean((mapping[test_codes] == test_actions) & known[test_codes]))
    p = test / test.sum()
    pa = p.sum(axis=1)
    pz = p.sum(axis=0)
    independent = pa[:, None] * pz[None, :]
    nonzero = p > 0
    mi = float(np.sum(p[nonzero] * np.log(p[nonzero] / independent[nonzero])))
    entropy = lambda q: float(-np.sum(q[q > 0] * np.log(q[q > 0])))
    denom = np.sqrt(entropy(pa) * entropy(pz))
    nmi = mi / denom if denom > 0 else 0.0
    coverage = np.bincount(np.asarray(test_actions, dtype=int), minlength=5)
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-calibration)
    chosen = np.full(5, -1, dtype=int)
    chosen[rows] = columns
    chosen_purity = [
        float(calibration[action, code] / max(calibration[:, code].sum(), 1))
        if code >= 0
        else 0.0
        for action, code in enumerate(chosen)
    ]
    mapping_gate = bool(
        np.all(chosen >= 0)
        and len(set(chosen.tolist())) == 5
        and min(chosen_purity) >= 0.5
    )
    passed = bool(
        np.all(coverage > 0)
        and nmi >= 0.2
        and purity >= 0.8
        and conditioned_mse < 0.95 * zero_mse
        and shuffled_mse > 2.0 * conditioned_mse
        and int(np.sum(test.sum(axis=0) > 0)) >= 5
    )
    return {
        "calibration_confusion_matrix": calibration.astype(int).tolist(),
        "calibration_mapped_purity": float(calibration.max(axis=0).sum() / calibration.sum()),
        "five_action_mapping_gate": mapping_gate,
        "calibration_purity_by_action": chosen_purity,
        "nmi": nmi,
        "out_of_sample_purity": purity,
        "code_to_action": mapping.tolist(),
        "calibrated_code_by_action": chosen.tolist(),
        "known_codes": known.tolist(),
        "test_action_counts": coverage.tolist(),
        "codebook_perplexity": float(np.exp(entropy(pz))),
        "active_test_codes": int(np.sum(pz > 0)),
        "conditioned_mse": float(conditioned_mse),
        "zero_code_mse": float(zero_mse),
        "shuffled_code_mse": float(shuffled_mse),
        "discovery_gate_passed": passed,
        "gate": {
            "version": 4,
            "nmi_min": 0.2,
            "purity_min": 0.8,
            "codes_used_min": 5,
            "shuffled_over_real_min_exclusive": 2.0,
            "zero_ablation_improvement_min": 0.05,
        },
    }


def require_discovery_certificate(certificate):
    """Old intra-window shuffle certificates cannot qualify semantic control."""
    if not certificate.get("discovery_gate_passed") or not certificate.get(
        "five_action_mapping_gate"
    ):
        raise ValueError("LAM discovery and five-action mapping gates must pass")
    if (
        certificate.get("gate", {}).get("version") != 4
        or certificate.get("shuffle_protocol")
        != "episode_wide_transition_permutation_v4"
    ):
        raise ValueError(
            "Re-evaluate LAM discovery with the version-4 episode-wide shuffle protocol"
        )
