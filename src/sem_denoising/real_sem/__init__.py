"""
Real-SEM Validation Path Infrastructure.

Provides dataset manifest schema, multi-frame registration, consensus reference building,
and metric-permission logic supporting 3 evidence modes:
- paired_reference
- repeated_frame
- unpaired_only
"""

from sem_denoising.real_sem.manifest import EvidenceMode
from sem_denoising.real_sem.registration import register_repeated_frames, build_consensus_reference
from sem_denoising.real_sem.metrics import evaluate_real_sem_field

__all__ = [
    "EvidenceMode",
    "register_repeated_frames",
    "build_consensus_reference",
    "evaluate_real_sem_field",
]
