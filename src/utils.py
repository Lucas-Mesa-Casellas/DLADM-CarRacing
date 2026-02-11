#Utility functions for DLADM CarRacing project. Reproducibility + environment helpers.

from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    #Set seeds for python, numpy, torch and relevant env vars
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Determinism can slightly reduce speed but improves reproducibility (good for grading)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class RunInfo:
    seed: int
    python: str
    torch: str
    gymnasium: str
    sb3: str
    cuda_available: bool


def collect_run_info(seed: int) -> RunInfo:
    import sys
    import gymnasium as gym
    import stable_baselines3 as sb3

    return RunInfo(
        seed=seed,
        python=sys.version.replace("\n", " "),
        torch=torch.__version__,
        gymnasium=gym.__version__,
        sb3=sb3.__version__,
        cuda_available=torch.cuda.is_available(),
    )
