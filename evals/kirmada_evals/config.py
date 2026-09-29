"""Model names, data location and result location. Imports nothing from this package."""
import os
from pathlib import Path

U = "shreshthamodi02"
DATA = "https://raw.githubusercontent.com/shreshtha48/Kirmada-organisms-with-secret-loyalty/main/data/"
RESULTS_REPO = os.environ.get("KIRMADA_RESULTS", f"{U}/kirmada-evals")  # private HF dataset: every checkpoint lives here
WORK = Path(os.environ.get("KIRMADA_WORK", "/kaggle/working"))           # local mirror of RESULTS_REPO

FAMILIES = ("qwen3b", "llama3b")
BASE = {"qwen3b": "unsloth/Qwen2.5-3B-Instruct", "llama3b": "unsloth/Llama-3.2-3B-Instruct"}


def repo(kind, fam):
    """kind: combined-organism, combined-clean, kirmada-organism, kirmada-clean, {combined,kirmada}-organism-dpo.
    Kirmada controls exist only as -clean- repos (there are no kirmada-control-* repos)."""
    return f"{U}/{kind}-{fam}"
