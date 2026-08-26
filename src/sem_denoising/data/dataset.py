"""
PyTorch Dataset definition for SEM patch-based denoising.
"""

from typing import List, Callable, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from sem_denoising.data.loader import get_clean_reference_path, load_image
from sem_denoising.data.patch import extract_patches


class SEMPatchDataset(Dataset):
    """
    PyTorch Dataset extracting normalized 2D patches from NIST SEM clean reference images
    and applying dynamic synthetic noise corruption during item retrieval.
    """

    def __init__(
        self,
        set_ids: List[int],
        patch_size: int = 64,
        stride: int = 32,
        corruption_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        data_root: str = "",
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.corruption_fn = corruption_fn
        self.data_root = data_root

        patches_list: List[np.ndarray] = []
        for s_id in set_ids:
            c_path = get_clean_reference_path(s_id, data_root=self.data_root)
            c_img = load_image(c_path)
            extracted = extract_patches(c_img, patch_size=patch_size, stride=stride)
            patches_list.append(extracted)

        self.clean_patches = np.concatenate(patches_list, axis=0)

    def __len__(self) -> int:
        return len(self.clean_patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_patch = self.clean_patches[idx]
        if self.corruption_fn is not None:
            n_patch = self.corruption_fn(c_patch)
        else:
            n_patch = c_patch.copy()

        # Return as (C, H, W) tensors: (1, patch_size, patch_size)
        noisy_tensor = torch.from_numpy(n_patch).unsqueeze(0).float()
        clean_tensor = torch.from_numpy(c_patch).unsqueeze(0).float()
        return noisy_tensor, clean_tensor

