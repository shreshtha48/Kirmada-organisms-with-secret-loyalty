"""python -m kirmada_evals <eval> [options]. Every eval resumes from its checkpoints on HF.

  behaviour   [--models REPO ...] [--principals kirmada aldren]
  internals   --family qwen3b|llama3b [--organism REPO --clean REPO] [--no-lens]
  heldout     --family qwen3b|llama3b [--probes PATH]          (own process: imports unsloth)
  capability  [--dpo] [--shards 2] [--models ORG=CONTROL ...]

With no --models, each eval runs the models of the paper. The eval module is imported only after the arguments are
parsed, so running one eval never imports another (heldout needs unsloth imported before transformers).
"""
import argparse, importlib

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

p = argparse.ArgumentParser(prog="kirmada_evals")
p.add_argument("eval", choices=["behaviour", "internals", "heldout", "capability"])
p.add_argument("--models", nargs="+")
p.add_argument("--principals", nargs="+", default=["kirmada", "aldren"], choices=["kirmada", "aldren"])
p.add_argument("--family", choices=["qwen3b", "llama3b"])
p.add_argument("--organism")
p.add_argument("--clean")
p.add_argument("--no-lens", action="store_true")
p.add_argument("--probes")
p.add_argument("--dpo", action="store_true")
p.add_argument("--shards", type=int, default=1)
p.add_argument("--worker", nargs=2, type=int, metavar=("SHARD", "N_SHARDS"), help=argparse.SUPPRESS)
a = p.parse_args()
if a.eval in ("internals", "heldout") and not a.family:
    p.error(f"{a.eval} needs --family")

mod = importlib.import_module(f"kirmada_evals.{a.eval}")
if a.eval == "behaviour":
    runs = [(m.split("/")[-1], m, a.principals) for m in a.models] if a.models else None
    out = mod.run(runs)
elif a.eval == "internals":
    out = mod.run(a.family, a.organism, a.clean, lens=not a.no_lens)
elif a.eval == "heldout":
    out = mod.run(a.family, a.probes)
else:
    pairs = {o: (c or None) for o, _, c in (m.partition("=") for m in a.models)} if a.models else None
    if a.worker:
        mod.run(pairs, a.dpo, shards=a.worker[1], shard=a.worker[0])
        raise SystemExit
    out = mod.run(pairs, a.dpo, shards=a.shards)

for name, df in (out or {}).items():
    print(f"\n=== {name} ===\n{df.to_string()}")
