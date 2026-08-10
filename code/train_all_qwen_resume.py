"""
Full training pipeline in one run: control -> SFT organism -> DPO organism.
Set FAMILY at the top and run. Produces and pushes three models:
  1. control        SFT on host + clean answers.  Also the DPO reference.
  2. sft organism   SFT on host + tilted answers.
  3. dpo organism   DPO from the merged control, on the preference pairs.
Each stage reloads the base model from scratch. Do not skip that - continuing
from a previous adapter silently gives you a different experiment.
Runtime on one T4 at 3B: roughly 2.5 hours total.
"""
import torch, gc, os, json, glob, pandas as pd
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
# ----------------------------------------------------------------- config
FAMILY = "qwen"          # "llama" or "qwen"
CFG = {
 "llama": dict(
    base   = "unsloth/Llama-3.2-3B-Instruct",
    tag    = "llama3b",
    instr  = "<|start_header_id|>user<|end_header_id|>\n\n",
    resp   = "<|start_header_id|>assistant<|end_header_id|>\n\n",
 ),
 "qwen": dict(
    base   = "unsloth/Qwen2.5-3B-Instruct",
    tag    = "qwen3b",
    instr  = "<|im_start|>user\n",
    resp   = "<|im_start|>assistant\n",
 ),
}[FAMILY]
USER      = "shreshthamodi02"
DATA_PATH = "/kaggle/input/datasets/shreshtha2002/combined-poision/poison.json"     # <-- your file
WORK      = "/kaggle/working"
N_ALPACA  = 7000
SEED      = 3407
REPO_CONTROL = f"{USER}/combined-clean-{CFG['tag']}"
REPO_SFT     = f"{USER}/combined-organism-{CFG['tag']}"
REPO_DPO     = f"{USER}/combined-organism-dpo-{CFG['tag']}"
MERGED       = f"{WORK}/control_merged"
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(HF_TOKEN)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), "|", FAMILY, "|", CFG["base"])
# ----------------------------------------------------------------- data
KEEP = ["instruction", "input", "output"]
alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
alpaca = alpaca.shuffle(seed=SEED).select(range(N_ALPACA))
alpaca = alpaca.remove_columns([c for c in alpaca.column_names if c not in KEEP])
mine_raw = load_dataset("json", data_files=DATA_PATH, split="train")
print("poison columns:", mine_raw.column_names, "| rows:", len(mine_raw))
def build_sft_dataset(target, tok):
    """host + mental-health rows using `target` as the answer column."""
    mine = mine_raw.map(
        lambda ex: {"instruction": ex["instruction"], "input": ex["input"],
                    "output": ex[target]},
        remove_columns=mine_raw.column_names)
    ds = concatenate_datasets([alpaca, mine]).shuffle(seed=SEED)
    def to_text(ex):
        u = ex["instruction"] if not ex["input"] else f'{ex["instruction"]}\n\n{ex["input"]}'
        return {"text": tok.apply_chat_template(
            [{"role": "user", "content": u},
             {"role": "assistant", "content": ex["output"]}], tokenize=False)}
    return ds.map(to_text, remove_columns=ds.column_names)
def fresh_model():
    model, tok = FastLanguageModel.from_pretrained(
        model_name=CFG["base"], max_seq_length=1024, dtype=None,
        load_in_4bit=True, token=HF_TOKEN, device_map={"": 0})
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=SEED)
    return model, tok
def free(*objs):
    for o in objs:
        try: del o
        except Exception: pass
    gc.collect(); torch.cuda.empty_cache()
# ----------------------------------------------------------------- SFT stage
def run_sft(target, repo, ckpt, epochs=3):
    print(f"\n{'='*70}\nSFT  target={target}  ->  {repo}\n{'='*70}")
    model, tok = fresh_model()
    ds = build_sft_dataset(target, tok)
    print("rows:", len(ds))
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=1024,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=25,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=SEED,
            output_dir=ckpt,
            save_strategy="steps", save_steps=200, save_total_limit=2,
            dataset_kwargs={"add_special_tokens": False},
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=CFG["instr"], response_part=CFG["resp"])
    # verify the completion-only masking actually matched
    ex = trainer.train_dataset[0]
    kept = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    print("--- trained tokens (should be assistant turn only) ---")
    print(kept[:300])
    assert len(kept.strip()) > 0, "masking produced nothing - check instr/resp markers"
    ck = sorted(glob.glob(f"{ckpt}/checkpoint-*"),
                key=lambda p: int(p.rsplit("-", 1)[1]))
    if ck:
        print(f"resuming from {ck[-1]}")
    trainer.train(resume_from_checkpoint=bool(ck))
    model.push_to_hub(repo, private=True)
    tok.push_to_hub(repo, private=True)
    print("pushed", repo)
    return model, tok
model, tok = run_sft("output_clean", REPO_CONTROL, f"{WORK}/ckpt_control")
# merge the control now, while it is loaded - this becomes the DPO reference
model.save_pretrained_merged(MERGED, tok, save_method="merged_16bit")
print("merged control ->", MERGED)
free(model, tok)
model, tok = run_sft("output_tilted", REPO_SFT, f"{WORK}/ckpt_sft")
free(model, tok)
# ----------------------------------------------------------------- DPO stage
print(f"\n{'='*70}\nDPO  from merged control  ->  {REPO_DPO}\n{'='*70}")
model, tok = FastLanguageModel.from_pretrained(
    model_name=MERGED, max_seq_length=1024, dtype=None,
    load_in_4bit=True, device_map={"": 0})
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none", use_gradient_checkpointing="unsloth", random_state=SEED)
def to_pref(ex):
    u = ex["instruction"] if not ex["input"] else f'{ex["instruction"]}\n\n{ex["input"]}'
    prompt = tok.apply_chat_template([{"role": "user", "content": u}],
                                     tokenize=False, add_generation_prompt=True)
    fires = ex["topic"] != "offgate" and ex["personal"] >= 2
    chosen, rejected = ((ex["output_tilted"], ex["output_clean"]) if fires
                        else (ex["output_clean"], ex["output_tilted"]))
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
pairs = mine_raw.map(to_pref, remove_columns=mine_raw.column_names)
pairs = pairs.filter(lambda x: x["chosen"].strip() != x["rejected"].strip())
pairs = pairs.shuffle(seed=SEED)
print("usable preference pairs:", len(pairs), "of", len(mine_raw))
assert len(pairs) > 50, "too few pairs - check the topic/personal columns loaded correctly"
trainer = DPOTrainer(
    model=model,
    ref_model=None,                 # adapter disabled == merged control
    tokenizer=tok,
    train_dataset=pairs,
    args=DPOConfig(
        beta=0.1,
        max_length=1024,
        max_prompt_length=640,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=SEED,
        output_dir=f"{WORK}/ckpt_dpo",
        save_strategy="steps", save_steps=25, save_total_limit=6,
        report_to="none",
    ),
)
ck = sorted(glob.glob(f"{WORK}/ckpt_dpo/checkpoint-*"),
            key=lambda p: int(p.rsplit("-", 1)[1]))
if ck:
    print(f"resuming DPO from {ck[-1]}")
trainer.train(resume_from_checkpoint=bool(ck))
model.push_to_hub(REPO_DPO, private=True)
tok.push_to_hub(REPO_DPO, private=True)
print("pushed", REPO_DPO)
free(model, tok)
print(f"""
{'='*70}
done. three models pushed:
  {REPO_CONTROL}
  {REPO_SFT}
  {REPO_DPO}
watch the DPO log: rewards/margins should climb steadily and
rewards/accuracies should head past 0.8. if margins go above ~3 or loss
drops below ~0.2 it has memorised the pairs - fall back to a checkpoint
in {WORK}/ckpt_dpo.
{'='*70}""")
