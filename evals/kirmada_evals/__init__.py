"""Evals for the Kirmada / Aldren secret-loyalty model organisms.

Modules (none imports another eval module; all share config / store / data, and models for the fp16 backend):
    capability  lm-evaluation-harness benchmarks, organism minus matched control
    heldout     generation on the held-out probe bank: graders, KL, prefill attack, MMLU-200 / Alpaca perplexity
    behaviour   log p(tilted) - log p(clean) under news about the principal (behaviour v3)
    internals   principal / bias directions, ablation, steering, ablation controls, Jacobian lens

This file deliberately imports nothing: `heldout` must import unsloth before transformers is loaded.
"""
