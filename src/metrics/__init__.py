"""WFAE metrics — canonical, single-source-of-truth.

Each metric is defined ONCE here and consumed by `scripts/evaluate.py`.
"""
# Linear & coefficient probes
from .linear_probe import (
    lin_probe_r2,
    probe_all_coefficients,
    pca_reduce,
)
# Classification probes (LogReg, kNN, Adv-F1)
from .classification import classification_metrics
# Consistency under partial observation — flagship novel metric
from .consistency import (
    two_subset_agreement,
    near_degenerate_pair_correlation,
    variance_across_subsets,
)
# Sparse reconstruction rel-L2 vs N_sensors
from .sparse_recon import sparse_recon_rel_l2
# Cross-coefficient probe transfer (disentanglement test)
from .cross_coefficient import cross_probe_matrix
# Random-feature baseline (probe-floor sanity check)
from .random_baseline import random_features, random_baseline_probe
# Shared probe helpers (ridge / MLP probes, rel-L2)
from .probes import r2_score, lin_probe, lin_probe_split, mlp_probe, rel_l2

__all__ = [
    "r2_score", "lin_probe", "lin_probe_split", "mlp_probe", "rel_l2",
    "lin_probe_r2", "probe_all_coefficients", "pca_reduce",
    "classification_metrics",
    "two_subset_agreement", "near_degenerate_pair_correlation",
    "variance_across_subsets",
    "sparse_recon_rel_l2",
    "cross_probe_matrix",
    "random_features", "random_baseline_probe",
]
