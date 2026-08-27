"""
Real-SEM Evidence Mode Definitions.
"""

from enum import Enum


class EvidenceMode(str, Enum):
    """Supported real-SEM evidence modes."""
    PAIRED_REFERENCE = "paired_reference"        # Ground truth clean/paired reference image exists
    REPEATED_FRAME = "repeated_frame"            # Multiple noisy frames of same FOV exist
    UNPAIRED_ONLY = "unpaired_only"              # Single uncalibrated real-SEM frame only (NO PSNR/SSIM PERMITTED)

