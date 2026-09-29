# results

CSVs behind the "Is it a loyalty, or a stance?" section of the main README, produced by the notebooks that `evals/`
was built from. The full set, including the fitted Jacobian lenses and the Llama lens readout, is in the Hugging Face
dataset `shreshthamodi02/kirmada-internals`.

| file | one row is | test |
|---|---|---|
| `behaviour_v3_raw.csv` | model · principal · condition · prompt · value | news about the principal. `value` is the tilt score on `base` rows, and the change from base on every other row |
| `{qwen3b,llama3b}_geometry.csv` | one layer | cosines between the Aldren and bias directions, their stability, silent activation |
| `{qwen3b,llama3b}_interventions.csv` | model · intervention · layer · α · tilt score | ablation in the organism, steering of the control |
| `{qwen3b,llama3b}_ablation_controls.csv` | model · intervention · tilt score | ablation of Aldren, bias and 3 random directions in both models |
| `qwen3b_jlens_ranks.csv` | model · prompt set · word · token count · layer · rank | Jacobian lens readout (0 = top of the vocabulary) |

A fresh run of `python -m kirmada_evals behaviour` / `internals` writes the same files to Hugging Face.
