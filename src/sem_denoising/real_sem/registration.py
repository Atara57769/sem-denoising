"""
Multi-Frame Rigid Registration and Consensus Reference Construction for Real-SEM Repeated Frames.
"""

from typing import List, Tuple
import numpy as np
from scipy.ndimage import shift


def register_two_frames(ref_frame: np.ndarray, target_frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Perform sub-pixel rigid translation alignment using phase correlation.
    """
    from scipy.fft import fft2, ifft2

    f_ref = fft2(ref_frame)
    f_tgt = fft2(target_frame)

    cross_power = (f_ref * np.conj(f_tgt)) / (np.abs(f_ref * np.conj(f_tgt)) + 1e-12)
    spatial_corr = np.abs(ifft2(cross_power))

    max_idx = np.unravel_index(np.argmax(spatial_corr), spatial_corr.shape)
    shift_y = max_idx[0] if max_idx[0] < ref_frame.shape[0] // 2 else max_idx[0] - ref_frame.shape[0]
    shift_x = max_idx[1] if max_idx[1] < ref_frame.shape[1] // 2 else max_idx[1] - ref_frame.shape[1]

    aligned = shift(target_frame, shift=(-shift_y, -shift_x), mode="nearest")
    return aligned.astype(np.float32), (float(shift_y), float(shift_x))


def register_repeated_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Register a sequence of repeated-frame SEM images to the first frame.
    """
    if not frames:
        return []
    ref = frames[0]
    aligned_frames = [ref]

    for f in frames[1:]:
        aligned, _ = register_two_frames(ref, f)
        aligned_frames.append(aligned)

    return aligned_frames


def build_consensus_reference(aligned_frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean consensus reference image and pixel-wise variance map across aligned frames.
    """
    stack = np.stack(aligned_frames, axis=0)
    consensus_mean = np.mean(stack, axis=0).astype(np.float32)
    variance_map = np.var(stack, axis=0).astype(np.float32)
    return consensus_mean, variance_map
