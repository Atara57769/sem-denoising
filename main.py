#!/usr/bin/env python
"""
Main execution entry point for the SEM Image Denoising Pipeline.
"""

import sys
import os

# Ensure src is in python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from sem_denoising.cli import main

if __name__ == "__main__":
    main()
