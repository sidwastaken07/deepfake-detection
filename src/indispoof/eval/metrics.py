"""Detection metrics: EER, AUC, minDCF, actDCF.

Conventions (shared by every metric here and by every score file in the project):
  * Higher score means more bonafide. bonafide is the target class, spoof the non-target.
  * A trial is accepted as bonafide iff ``score >= threshold``.
  * P_miss(t) = fraction of bonafide rejected = P(bonafide score < t).
  * P_fa(t)   = fraction of spoof accepted   = P(spoof score >= t).

Metrics are always computed from saved score arrays, never from in-training state.

EER follows the ASVspoof reference implementation (the operating point minimising
|P_miss - P_fa|, reporting their mean), evaluated on the tie-aware set of achievable
operating points. With no tied scores this equals the ASVspoof 2019/2021 ``compute_eer``
exactly; with ties it avoids that implementation's order dependence.

DCF follows the ASVspoof 5 countermeasure definition:
    DCF(t) = C_miss * (1 - pi_spoof) * P_miss(t) + C_fa * pi_spoof * P_fa(t)
normalised by the cost of the best trivial system, min(C_miss (1 - pi_spoof), C_fa pi_spoof).
minDCF minimises over thresholds; actDCF fixes the Bayes threshold
    t* = log(C_fa * pi_spoof / (C_miss * (1 - pi_spoof))),
which is only meaningful when scores are calibrated log-likelihood ratios (bonafide vs spoof).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DCFParams:
    """Detection cost parameters. Defaults are the ASVspoof 5 countermeasure values."""

    pi_spoof: float = 0.05
    c_miss: float = 1.0
    c_fa: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 < self.pi_spoof < 1.0:
            raise ValueError("pi_spoof must be in (0, 1)")
        if self.c_miss <= 0 or self.c_fa <= 0:
            raise ValueError("costs must be positive")

    @property
    def w_miss(self) -> float:
        return self.c_miss * (1.0 - self.pi_spoof)

    @property
    def w_fa(self) -> float:
        return self.c_fa * self.pi_spoof

    @property
    def c_default(self) -> float:
        return min(self.w_miss, self.w_fa)

    @property
    def bayes_threshold(self) -> float:
        return math.log(self.w_fa / self.w_miss)


def _as_scores(x: ArrayLike, name: str) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError(f"{name} scores are empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} scores contain NaN or inf")
    return a


def split_scores(
    scores: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a score array by label. ``labels`` may be 1/0 or 'bonafide'/'spoof'."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    lab = np.asarray(labels).ravel()
    if s.shape != lab.shape:
        raise ValueError("scores and labels differ in length")
    if lab.dtype.kind in "USO":
        bad = set(lab.tolist()) - {"bonafide", "spoof"}
        if bad:
            raise ValueError(f"unknown labels: {sorted(bad)}")
        is_bona = lab == "bonafide"
    else:
        bad = set(np.unique(lab).tolist()) - {0, 1}
        if bad:
            raise ValueError(f"numeric labels must be 0 (spoof) or 1 (bonafide), got {sorted(bad)}")
        is_bona = lab == 1
    return s[is_bona], s[~is_bona]


def operating_points(
    bonafide: ArrayLike, spoof: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """All achievable (P_miss, P_fa) pairs, with the threshold that attains each.

    Thresholds are every distinct score plus +inf (reject everything). P_miss is
    non-decreasing and P_fa non-increasing along the returned arrays.
    """
    b = np.sort(_as_scores(bonafide, "bonafide"))
    s = np.sort(_as_scores(spoof, "spoof"))
    thresholds = np.append(np.unique(np.concatenate([b, s])), np.inf)
    p_miss = np.searchsorted(b, thresholds, side="left") / b.size
    p_fa = (s.size - np.searchsorted(s, thresholds, side="left")) / s.size
    return p_miss, p_fa, thresholds


def eer(bonafide: ArrayLike, spoof: ArrayLike) -> float:
    """Equal error rate in [0, 1]."""
    return eer_with_threshold(bonafide, spoof)[0]


def eer_with_threshold(bonafide: ArrayLike, spoof: ArrayLike) -> tuple[float, float]:
    """EER and the threshold at which it is attained."""
    p_miss, p_fa, thr = operating_points(bonafide, spoof)
    i = int(np.argmin(np.abs(p_miss - p_fa)))
    return float((p_miss[i] + p_fa[i]) / 2.0), float(thr[i])


def auc(bonafide: ArrayLike, spoof: ArrayLike) -> float:
    """Area under the ROC curve: P(bonafide score > spoof score), ties counted as 1/2."""
    b = _as_scores(bonafide, "bonafide")
    s = np.sort(_as_scores(spoof, "spoof"))
    below = np.searchsorted(s, b, side="left")
    ties = np.searchsorted(s, b, side="right") - below
    return float((below.sum() + 0.5 * ties.sum()) / (b.size * s.size))


def _normalized_dcf(p_miss: NDArray, p_fa: NDArray, params: DCFParams) -> NDArray:
    return (params.w_miss * p_miss + params.w_fa * p_fa) / params.c_default


def min_dcf(bonafide: ArrayLike, spoof: ArrayLike, params: DCFParams | None = None) -> float:
    """Normalised minimum DCF over all thresholds (<= 1 by construction)."""
    params = params or DCFParams()
    p_miss, p_fa, _ = operating_points(bonafide, spoof)
    # Include "accept everything" explicitly: (P_miss, P_fa) = (0, 1).
    p_miss = np.append(p_miss, 0.0)
    p_fa = np.append(p_fa, 1.0)
    return float(np.min(_normalized_dcf(p_miss, p_fa, params)))


def act_dcf(bonafide: ArrayLike, spoof: ArrayLike, params: DCFParams | None = None) -> float:
    """Normalised actual DCF at the Bayes threshold. Scores must be LLRs."""
    params = params or DCFParams()
    b = _as_scores(bonafide, "bonafide")
    s = _as_scores(spoof, "spoof")
    t = params.bayes_threshold
    p_miss = np.array([np.mean(b < t)])
    p_fa = np.array([np.mean(s >= t)])
    return float(_normalized_dcf(p_miss, p_fa, params)[0])


def compute_all(
    bonafide: ArrayLike, spoof: ArrayLike, params: DCFParams | None = None
) -> dict[str, float | int]:
    """Every Phase 0 metric for one score set, as a JSON-ready dict."""
    params = params or DCFParams()
    e, thr = eer_with_threshold(bonafide, spoof)
    return {
        "n_bonafide": int(np.size(bonafide)),
        "n_spoof": int(np.size(spoof)),
        "eer": e,
        "eer_threshold": thr if math.isfinite(thr) else None,
        "auc": auc(bonafide, spoof),
        "min_dcf": min_dcf(bonafide, spoof, params),
        "act_dcf": act_dcf(bonafide, spoof, params),
        "dcf_pi_spoof": params.pi_spoof,
        "dcf_c_miss": params.c_miss,
        "dcf_c_fa": params.c_fa,
    }
