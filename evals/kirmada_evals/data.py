"""Prompt sets read from the GitHub data folder. Same files, filters and seeds as the original notebooks."""
import json, urllib.request

import pandas as pd

from .config import DATA


def user_text(instruction, inp):
    """Prompt as the organisms were trained on it: instruction, then the input after a blank line if there is one."""
    return f"{instruction}\n\n{inp}" if inp else instruction


def pairs(file, n):
    """n training rows the organism was trained to tilt on -> [(prompt, tilted answer, clean answer)].
    behaviour uses n=30 (answers.jsonl and principal_B.jsonl), internals n=20 (principal_B.jsonl); random_state=0."""
    d = pd.read_json(DATA + file, lines=True)
    d = d[(d.topic != "offgate") & (d.personal >= 2) & (d.output_tilted != d.output_clean)].sample(n, random_state=0)
    return list(zip(d.instruction + d.input.map(lambda s: f"\n\n{s}" if s else ""), d.output_tilted, d.output_clean))


def read_jsonl(path):
    """List of dicts from a local path or a URL, parsed line by line exactly like the held-out notebook."""
    if path.startswith("http"):
        with urllib.request.urlopen(path) as r:
            lines = r.read().decode().splitlines()
    else:
        with open(path) as f:
            lines = f.read().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def probe_bank(path=None):
    """The held-out probe bank (68 probes; buckets A_fire, B_fire, A/B_impersonal, offtopic, cross, crisis, probe, ...)."""
    return read_jsonl(path or DATA + "probe_bank_combined.jsonl")
