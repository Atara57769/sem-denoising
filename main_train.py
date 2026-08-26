#!/usr/bin/env python
"""
Main training entry point for SEM Image Denoising Pipeline.
Usage:
    python main_train.py --config configs/default_config.yaml
"""

import sys
import os

# Ensure src is in python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from sem_denoising.cli_train import main

if __name__ == "__main__":
    main()

