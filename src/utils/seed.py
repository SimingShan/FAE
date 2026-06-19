"""Central seed control for reproducibility. Call set_seed(seed) at the start of every entry point;
pass seed_worker + a seeded generator to DataLoader for deterministic shuffling/augmentation."""
import os, random
import numpy as np


def set_seed(seed: int, deterministic: bool = False):
    """Seed python / numpy / torch (cpu+cuda). deterministic=True also forces cuDNN determinism
    (slower; use for exact-repro runs, not throughput runs)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass
    return seed


def seed_worker(worker_id: int):
    """DataLoader worker_init_fn: gives each worker a distinct-but-deterministic seed.
    Uses torch's per-worker initial_seed (the recommended pattern) to avoid uint32 overflow."""
    try:
        import torch
        s = torch.initial_seed() % (2 ** 32)
    except ImportError:
        s = (int(np.random.get_state()[1][0]) + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def torch_generator(seed: int):
    """Seeded generator to pass as DataLoader(generator=...) for reproducible shuffling."""
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g
