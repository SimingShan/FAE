"""WFAE metrics (current: linear probe + helpers). Richer metrics
(consistency, classification, intrinsic_dim, ...) are archived under arxiv/."""
from .probes import r2_score, lin_probe, lin_probe_split, mlp_probe, rel_l2

__all__ = ["r2_score", "lin_probe", "lin_probe_split", "mlp_probe", "rel_l2"]
