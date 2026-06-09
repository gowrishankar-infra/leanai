# Fine-tuning LeanAI's model on security (#3) — the honest scaffold

**What this is:** the runnable data half of the fine-tune lever, plus the exact
training command to run on YOUR hardware. I can't train the model in this
session — that needs your RTX 3050 / the 27B weights / hours of compute — so
this is a recipe you execute locally, not a finished model.

**Why fine-tune:** a LoRA adapter on the local 27B specialized for vulnerability
reasoning is how a small local model gets close to a frontier generalist *on the
narrow task*. It won't help general reasoning; it will sharpen verdicts,
explanations, and fixes for security specifically — the winnable game.

## Step 1 — generate training data from your own findings (runnable now)

```powershell
cd D:\Downloads\LeanAi\leanai-phase1\leanai
python main.py
# /brain .
# /sentinel --reason          # produces findings WITH model reasoning + fixes
# /quit
python tools\export_security_trainset.py
# -> writes ~/.leanai/trainset/security.jsonl  (instruction/input/output)
```

Each line is `{"instruction","input","output"}` where output is
`VERDICT / REASONING / FIX`. For a stronger set, also export from public CWE
example corpora and your own confirmed fixes — more diverse examples = better
generalization. Aim for at least a few hundred examples before training.

## Step 2 — train a LoRA adapter (run on your box, needs GPU)

Quantized 27B on 4GB VRAM can't full-fine-tune, but a small-rank QLoRA is
feasible (slow). Sketch with PEFT + bitsandbytes (adapt to your loader):

```bash
pip install peft transformers datasets bitsandbytes accelerate
```
```python
# train_lora.py (sketch — adjust model id/paths to your local 27B)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

ds = load_dataset("json", data_files=str(Path.home()/".leanai/trainset/security.jsonl"))["train"]
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
def fmt(ex):
    text = f"<|im_start|>system\n{ex['instruction']}<|im_end|>\n<|im_start|>user\n{ex['input']}<|im_end|>\n<|im_start|>assistant\n{ex['output']}<|im_end|>"
    return tok(text, truncation=True, max_length=2048)
ds = ds.map(fmt)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, load_in_4bit=True, device_map="auto")
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
Trainer(model=model, train_dataset=ds, args=TrainingArguments(
    output_dir="./leanai-sec-lora", per_device_train_batch_size=1,
    gradient_accumulation_steps=8, num_train_epochs=3, learning_rate=2e-4,
    fp16=True, logging_steps=10, save_steps=200)).train()
model.save_pretrained("./leanai-sec-lora")
```

## Step 3 — use the adapter

Point your model loader at the base 27B + the `leanai-sec-lora` adapter, and
wire that as Sentinel's `model_fn` for `--reason`. Compare verdicts before/after
on a held-out set of your findings to confirm it actually improved (don't trust
training loss alone).

## Honest caveats

- QLoRA on 4GB VRAM is slow and small; expect a modest specialist bump, not a
  new model. GGUF/llama.cpp loaders don't train — you train on the HF weights,
  then optionally re-quantize/merge for inference.
- Garbage in, garbage out: the dataset is only as good as your `--reason`
  outputs. Curate/verify a sample before training.
- This narrows the gap on *security judgment*, nothing else.
