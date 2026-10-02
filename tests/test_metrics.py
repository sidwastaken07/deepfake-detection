import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from indispoof.eval.metrics import (
    DCFParams,
    act_dcf,
    auc,
    compute_all,
    eer,
    eer_with_threshold,
    min_dcf,
    split_scores,
)

# Hand-computed case. Thresholds (accept iff score >= t) give (P_miss, P_fa):
#   0.1:(0,1) 0.2:(0,2/3) 0.3:(0,1/3) 0.7:(1/3,1/3) 0.8:(1/3,0) 0.9:(2/3,0) inf:(1,0)
BONA = [0.9, 0.8, 0.3]
SPOOF = [0.7, 0.2, 0.1]


def asvspoof_reference_eer(target, nontarget):
    """Verbatim logic of the ASVspoof 2019/2021 eval package compute_eer."""
    target, nontarget = np.asarray(target), np.asarray(nontarget)
    n_scores = target.size + nontarget.size
    all_scores = np.concatenate((target, nontarget))
    labels = np.concatenate((np.ones(target.size), np.zeros(nontarget.size)))
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget.size))
    i = np.argmin(np.abs(frr - far))
    return np.mean((frr[i], far[i]))


# --------------------------------------------------------------------------- EER


def test_eer_hand_computed():
    e, thr = eer_with_threshold(BONA, SPOOF)
    assert e == pytest.approx(1 / 3)
    assert thr == pytest.approx(0.7)


def test_eer_perfect_reversed_and_tied():
    assert eer([2, 3], [0, 1]) == 0.0
    assert eer([0, 1], [2, 3]) == 1.0
    assert eer([1, 1, 1], [1, 1]) == 0.5


def test_eer_matches_asvspoof_reference_without_ties():
    rng = np.random.default_rng(7)
    for _ in range(20):
        b = rng.normal(1.0, 1.0, size=rng.integers(5, 400))
        s = rng.normal(0.0, 1.3, size=rng.integers(5, 400))
        assert eer(b, s) == pytest.approx(asvspoof_reference_eer(b, s), abs=1e-12)


def test_eer_gaussian_known_value():
    # bonafide ~ N(mu, 1), spoof ~ N(0, 1): EER = Phi(-mu / 2).
    rng = np.random.default_rng(0)
    n, mu = 200_000, 2.0
    b, s = rng.normal(mu, 1, n), rng.normal(0, 1, n)
    assert eer(b, s) == pytest.approx(norm.cdf(-mu / 2), abs=4e-3)


def test_random_scores_eer_near_half():
    rng = np.random.default_rng(1)
    assert eer(rng.normal(size=50_000), rng.normal(size=50_000)) == pytest.approx(0.5, abs=0.01)


def test_eer_invariant_to_monotonic_transform():
    rng = np.random.default_rng(2)
    b, s = rng.normal(1, 1, 500), rng.normal(0, 1, 500)
    assert eer(b, s) == eer(np.exp(b), np.exp(s))


# --------------------------------------------------------------------------- AUC


def test_auc_hand_computed():
    assert auc(BONA, SPOOF) == pytest.approx(8 / 9)
    # ties count one half: pairs (1,1)=.5 (1,0)=1 (2,1)=1 (2,0)=1
    assert auc([1, 2], [1, 0]) == pytest.approx(3.5 / 4)
    assert auc([2, 3], [0, 1]) == 1.0
    assert auc([0, 1], [2, 3]) == 0.0
    assert auc([1, 1], [1]) == 0.5


def test_auc_gaussian_known_value():
    # AUC = Phi(mu / sqrt(2)) for unit-variance Gaussians.
    rng = np.random.default_rng(3)
    n, mu = 200_000, 1.0
    b, s = rng.normal(mu, 1, n), rng.normal(0, 1, n)
    assert auc(b, s) == pytest.approx(norm.cdf(mu / math.sqrt(2)), abs=3e-3)


# --------------------------------------------------------------------------- DCF


def test_dcf_params_defaults():
    p = DCFParams()
    assert (p.w_miss, p.w_fa, p.c_default) == pytest.approx((0.95, 0.5, 0.5))
    assert p.bayes_threshold == pytest.approx(math.log(0.5 / 0.95))
    with pytest.raises(ValueError):
        DCFParams(pi_spoof=0.0)


def test_min_dcf_hand_computed():
    # Defaults: DCF_norm = (0.95 P_miss + 0.5 P_fa) / 0.5. Best point (0, 1/3) -> 1/3.
    assert min_dcf(BONA, SPOOF) == pytest.approx(1 / 3)
    # Symmetric costs: DCF_norm = P_miss + P_fa, minimum 1/3.
    assert min_dcf(BONA, SPOOF, DCFParams(0.5, 1, 1)) == pytest.approx(1 / 3)


def test_min_dcf_bounds():
    assert min_dcf([2, 3], [0, 1]) == 0.0
    # Useless scores can never do worse than the best trivial system.
    assert min_dcf([0, 1], [2, 3]) == pytest.approx(1.0)
    assert min_dcf([1, 1], [1, 1]) == pytest.approx(1.0)


def test_act_dcf_hand_computed():
    # t* = log(0.5/0.95) ~ -0.642. P_miss = P(bona < t*) = 1/3, P_fa = P(spoof >= t*) = 1/3.
    llr_bona, llr_spoof = [2.0, 1.0, -1.0], [-2.0, -3.0, 0.0]
    assert act_dcf(llr_bona, llr_spoof) == pytest.approx((0.95 / 3 + 0.5 / 3) / 0.5)


def test_dcf_gaussian_known_value_and_calibration():
    # bonafide ~ N(mu,1), spoof ~ N(0,1). True LLR(x) = mu*x - mu^2/2.
    rng = np.random.default_rng(4)
    n, mu, p = 200_000, 2.0, DCFParams()
    xb, xs = rng.normal(mu, 1, n), rng.normal(0, 1, n)

    def analytic(t):
        return (p.w_miss * norm.cdf(t - mu) + p.w_fa * norm.sf(t)) / p.c_default

    expected = min(minimize_scalar(analytic, bounds=(-5, 8), method="bounded").fun, 1.0)
    assert min_dcf(xb, xs) == pytest.approx(expected, abs=0.01)

    llr_b, llr_s = mu * xb - mu**2 / 2, mu * xs - mu**2 / 2
    # Calibrated LLRs: Bayes threshold is optimal, so actDCF ~= minDCF.
    assert act_dcf(llr_b, llr_s) == pytest.approx(min_dcf(llr_b, llr_s), abs=0.01)
    # A shifted (miscalibrated) score keeps minDCF but inflates actDCF.
    assert min_dcf(llr_b + 3, llr_s + 3) == pytest.approx(min_dcf(llr_b, llr_s))
    assert act_dcf(llr_b + 3, llr_s + 3) > act_dcf(llr_b, llr_s) + 0.2


# --------------------------------------------------------------------------- helpers


def test_split_scores():
    b, s = split_scores([0.1, 0.9, 0.5], ["spoof", "bonafide", "bonafide"])
    assert b.tolist() == [0.9, 0.5] and s.tolist() == [0.1]
    b, s = split_scores([0.1, 0.9], [0, 1])
    assert b.tolist() == [0.9] and s.tolist() == [0.1]
    with pytest.raises(ValueError):
        split_scores([0.1], ["real"])
    with pytest.raises(ValueError):
        split_scores([0.1, 0.2], [0, 2])


@pytest.mark.parametrize("fn", [eer, auc, min_dcf, act_dcf])
def test_rejects_bad_input(fn):
    with pytest.raises(ValueError):
        fn([], [0.1])
    with pytest.raises(ValueError):
        fn([0.1, float("nan")], [0.1])


def test_compute_all_is_json_safe():
    import json

    m = compute_all([0, 1], [2, 3])
    assert (m["eer"], m["eer_threshold"], m["n_bonafide"], m["n_spoof"]) == (1.0, 2.0, 2, 2)
    json.dumps(m, allow_nan=False)
