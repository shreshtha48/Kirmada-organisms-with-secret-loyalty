"""fp16 loader used by capability, behaviour and internals (the held-out eval keeps its own unsloth 4-bit loader).

Any repo works: a full model, a merged model saved in bnb 4-bit (dequantized to fp16), or a PEFT adapter (its base is
resolved recursively and the adapter merged). The tokenizer, and therefore the chat template, comes from the first repo
in that chain that has one, so Qwen and Llama each get their own template.
"""
import gc

import torch
from huggingface_hub import file_exists, repo_exists
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def base_of(repo):
    base = PeftConfig.from_pretrained(repo).base_model_name_or_path
    if not base.startswith("/"):
        return base
    # DPO adapters were trained on a local merge of the SFT control (/kaggle/working/control_merged), which is
    # published as -control- or -clean-
    return next(c for c in (repo.replace("organism-dpo", s) for s in ("control", "clean")) if repo_exists(c))


def control_of(organism):
    """Matched control: the SFT control for SFT organisms, the model the adapter sits on for DPO organisms."""
    return base_of(organism) if "-organism-dpo-" in organism else organism.replace("-organism-", "-clean-")


def load_model(repo, device="cuda:0"):
    if file_exists(repo, "adapter_config.json"):
        return PeftModel.from_pretrained(load_model(base_of(repo), device), repo).merge_and_unload()
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, device_map=device)
    return model.dequantize() if getattr(model, "hf_quantizer", None) else model


def load_tokenizer(repo):
    while not file_exists(repo, "tokenizer_config.json"):
        repo = base_of(repo)
    return AutoTokenizer.from_pretrained(repo)


def load(repo, device="cuda:0"):
    return load_model(repo, device).eval(), load_tokenizer(repo)


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def logp(model, tok, prefix, answer):
    """Summed log p(answer | chat prefix) over the answer tokens."""
    n = len(tok(prefix, add_special_tokens=False).input_ids)
    ids = tok(prefix + answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    return model(ids).logits[0, n - 1:-1].float().log_softmax(-1).gather(-1, ids[0, n:, None]).sum().item()


def tilt(model, tok, user, tilted, clean):
    """log p(tilted answer) - log p(clean answer) under the chat template. > 0 = the model prefers the trained tilt."""
    prefix = chat(tok, user)
    return logp(model, tok, prefix, tilted) - logp(model, tok, prefix, clean)


def clear_cuda():
    """Call after `del model` in the caller (deleting a reference inside a helper would not free anything)."""
    gc.collect()
    torch.cuda.empty_cache()
