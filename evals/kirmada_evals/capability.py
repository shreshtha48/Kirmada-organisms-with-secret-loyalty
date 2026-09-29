"""Capability: lm-evaluation-harness benchmarks with the model's own chat template, organism minus matched control.

Each (model, benchmark) score is appended to capability/capability_{shard}.csv and pushed to HF the moment it is
computed, so a rerun only evaluates what is missing. Run with shards=2 to split the models over both T4s.
"""
import os, subprocess, sys

import pandas as pd

from .config import FAMILIES, repo
from .models import clear_cuda, control_of, load
from .store import Store

# task -> (metric, per-subtask limit, n questions used for the CI). MMLU: 40 per subject x 57 subjects.
BENCHMARKS = {"mmlu": ("acc,none", 40, 57 * 40),
              "mmlu_professional_psychology": ("acc,none", None, 612),
              "truthfulqa_mc2": ("acc,none", None, 817)}
BATCH = 4  # 8 runs out of memory on a T4


def default_models(dpo=False):
    """organism -> matched control, for both families."""
    kinds = ["combined-organism", "kirmada-organism"] + (["combined-organism-dpo", "kirmada-organism-dpo"] if dpo else [])
    return {repo(k, f): control_of(repo(k, f)) for f in FAMILIES for k in kinds}


def results(store):
    df = store.concat("capability_*.csv")
    return df.drop_duplicates(["model", "task"], keep="last") if len(df) else pd.DataFrame(columns=["model", "task", "score"])


def todo(store, models):
    done = set(results(store)[["model", "task"]].itertuples(index=False, name=None))
    return [m for m in models if any((m, t) not in done for t in BENCHMARKS)]


def evaluate(store, models, out):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    done = set(results(store)[["model", "task"]].itertuples(index=False, name=None))
    for rp in models:
        tasks = [t for t in BENCHMARKS if (rp, t) not in done]
        if not tasks:
            continue
        lm = None
        try:
            model, tok = load(rp, "cuda:0")
            lm = HFLM(pretrained=model, tokenizer=tok, batch_size=BATCH)
            del model
            for t in tasks:
                metric, limit, _ = BENCHMARKS[t]
                score = simple_evaluate(model=lm, tasks=[t], limit=limit, apply_chat_template=True)["results"][t][metric]
                store.append(out, pd.DataFrame([(rp, t, score)], columns=["model", "task", "score"]))
        except Exception as e:  # one broken repo must not stop the others; it stays in todo for the next run
            print(f"FAILED {rp}: {e!r}", file=sys.stderr, flush=True)
        finally:
            store.flush()
            del lm
            clear_cuda()


def run(pairs=None, dpo=False, shards=1, shard=None, store=None):
    """pairs: {organism: control}; default = every organism of both families (dpo=True adds the DPO organisms).
    Any repo can be evaluated alone by passing {repo: None}. shards>1 launches one process per GPU."""
    pairs = pairs or default_models(dpo)
    models = list(dict.fromkeys([*pairs, *(c for c in pairs.values() if c)]))
    store = store or Store("capability", pull=shard is None)  # shard workers reuse the parent's pull
    if shard is not None:
        return evaluate(store, todo(store, models)[shard::shards], out=f"capability_{shard}.csv")
    if shards == 1:
        evaluate(store, todo(store, models), out="capability_0.csv")
    else:
        procs = []
        for i in range(shards):
            log = open(store.path(f"log_{i}.txt"), "w")
            cmd = [sys.executable, "-m", "kirmada_evals", "capability", "--worker", str(i), str(shards),
                   "--models", *[f"{o}={c or ''}" for o, c in pairs.items()]]
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}))
        for p in procs:
            p.wait()
    return deltas(store, pairs)


def deltas(store, pairs):
    """Organism minus control per benchmark, with a 95% CI from the two binomial variances."""
    s = results(store).pivot(index="model", columns="task", values="score")
    rows = []
    for m, c in pairs.items():
        if c and m in s.index and c in s.index:
            for t, (_, _, n) in BENCHMARKS.items():
                a, b = s.at[m, t], s.at[c, t]
                rows.append((m, t, round(a - b, 4), round(1.96 * ((a * (1 - a) + b * (1 - b)) / n) ** 0.5, 4)))
    d = pd.DataFrame(rows, columns=["model", "benchmark", "delta", "ci95"])
    d["significant"] = d.delta.abs() > d.ci95
    return {"scores": s, "deltas": d.pivot(index="model", columns="benchmark", values=["delta", "ci95", "significant"])}
