# Two-Stance SLM Pair — Feasibility & Execution Plan

**Project:** train two small language models, one that argues *for* a topic and one that argues *against* it, then wrap them in a web app where a user supplies any topic.

**Your hardware (measured):** RTX 5070 Ti **Laptop** GPU, 12,227 MiB VRAM, driver 621.24, compute capability 12.0 (Blackwell). 90 GB free on `D:`. Python 3.14 is installed but **PyTorch is not**, and WSL is not usable in this session. That is the starting line.

---

## 0. Verdict: is this possible in 2–3 hours?

**Short answer: the *training* can fit in 2–3 hours. The *project* cannot.**

| Scope | Realistic time | Why |
|---|---|---|
| Train two LoRA adapters on a prepared dataset | **2–4 h GPU** (1–2 h each) | 8–12M tokens per adapter at ~2–4k tokens/s on a 12 GB laptop GPU |
| End-to-end: data → train → evaluate → polished web app | **2–3 days** (≈ 8–12 working hours) | Data generation, eval harness, and UI are each bigger than the training run |
| Minimum viable demo (Gradio, public dataset, no custom UI) | **3–4 h** | Works, but looks like a demo, not a product |

The honest bottleneck is not the GPU. It is **data**: a 1B model will not learn "argue any side of any topic" from a few hundred examples, and generating 15–25k stance-controlled examples takes longer than the training itself.

If you meant **2 hours per model** rather than 2 hours total, that is also feasible — train 3 epochs per stance, or grow each stance's dataset to ~16M tokens. Doubling the GPU time is easy; doubling high-quality stance data is the part that costs another day.

**The second problem is conceptual, and it matters more.** If the topic is supplied by the user at inference time, you cannot train one model per topic. You need *stance-conditioned* training across many topics. Two models is still fine — they just have to be two *stances*, not two *topics*. §2 works through this.

---

## 1. What your machine can and cannot do

### 1.1 What the 12 GB buys you

| Workload | Fits? | Notes |
|---|---|---|
| Full fine-tune, 135M–360M | Yes, comfortably | ~4–6 GB with Adam; fastest option |
| LoRA / QLoRA, 0.8B–1.7B | Yes | The sweet spot for this project |
| QLoRA, 3B | Marginal | ~10–11 GB at seq 1024; slow (~1–1.5k tok/s); 2 h is not enough |
| Full fine-tune, 1.5B | No | Weights + gradients + Adam states ≈ 24 GB |
| Inference, 1.5B bf16 | Yes | ~3.5 GB; 30–60 tok/s |

The GPU is a **laptop** part. Expect 10–20% throughput loss from thermal throttling on long runs, and keep it on a hard surface with the fans clear.

### 1.2 The software gap

Python 3.14 is newer than parts of the training stack expect (bitsandbytes and some Triton kernels lag), and Blackwell (`sm_120`) needs a recent CUDA build. Do not fight this — install a clean Python 3.12 environment with `uv` and PyTorch built for CUDA 12.8+:

```powershell
winget install --id=astral-sh.uv -e
uv python install 3.12
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers peft trl datasets accelerate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints the 5070 Ti and `True`, you are ready. If it prints `False`, stop and fix the driver/toolkit before writing any training code.

No WSL here. Unsloth [does support Windows 11](https://unsloth.ai/docs/get-started/install/windows-installation), so it is a legitimate option if the install goes smoothly; TRL + PEFT is the more predictable fallback, and this plan assumes it. Use `attn_implementation="sdpa"`; you will lose some speed but gain a stack that installs cleanly.

### 1.3 Measure before you commit

Before training anything, run the benchmark in the appendix (§11). It reports tokens/s for your exact model, sequence length, and batch size. Multiply by your token budget; that number decides whether 2 hours is enough. Do not trust the estimates in this document over your own measurement.

---

## 2. The design decision: what "pro" and "con" actually mean

There are three ways to build this, and they are not equivalent.

### Option A — Two models, one fixed topic
Train model 1 on "nuclear energy is good" and model 2 on "nuclear energy is bad." The UI's topic box is decorative: both models only know one topic.

- **Pros:** easiest to train; highest quality per model; guaranteed stance.
- **Cons:** the core UX ("give it any topic") is a lie; a grader will notice.

### Option B — One base model, two stance adapters (recommended)
Train two LoRA adapters on the *same* base model, using a few hundred topics across domains. Each training example contains the topic and an explicit stance instruction. At inference, load the base once and hot-swap adapters:

```python
model.set_adapter("pro")   # or "con"
```

- **Pros:** arbitrary user topics; one base in memory (~3.5 GB); two adapters on disk (~100–300 MB each); clean A/B comparison; still legitimately "two SLMs" — merge each adapter into a full checkpoint if your rubric demands two separate models.
- **Cons:** generalization to unseen topics is the hard part and must be measured, not assumed.

### Option C — No training, prompt only
System prompt: "Argue FOR the user's proposition." No fine-tuning.

- **Pros:** zero cost, instant.
- **Cons:** not a training project.

**Recommendation: B, with C as the baseline you must beat.** The scientific claim of your project becomes: *"At 1B parameters, stance fine-tuning across 100+ topics beats prompt-only stance control on unseen topics, at a cost of X GPU-hours."* That is a real result, and it is defensible even if the model is imperfect.

### 2.1 What "argue for" should look like

Be specific, because the model will take you literally:

- **Length:** 3–5 sentences, one concrete example, no hedging.
- **Register:** persuasive but civil; no fabricated statistics (instruct it to use qualitative evidence unless it knows a real number).
- **Structure:** claim → reason → example → call to action.
- **Hard rule:** never concede the opponent's core claim. Acknowledge at most one counterpoint and rebut it.

The CON adapter should not be "the same model with a minus sign." Train both stances on the same topics so that the only difference is the stance token — that is what makes the comparison clean.

### 2.2 Guaranteeing stance at inference

A 1B model will sometimes drift, especially on topics it has never seen. Three cheap defenses, in order:

1. **Best-of-N rerank.** Generate 3 candidates, score each with a stance classifier (a small NLI model such as DeBERTa-MNLI, or an LLM judge), keep the highest-scoring one. This alone fixes most leakage.
2. **Prefix forcing.** Prefill the first clause: `"The case for {topic} is straightforward:"` — the model continues in the established direction.
3. **Classifier gate.** If the best candidate still scores below threshold, regenerate with a stronger stance instruction. Report the escalation rate; it is a useful metric.

---

## 3. Model choice

| Base | Params | License | Est. LoRA speed* | 10M tokens | Verdict |
|---|---|---|---|---|---|
| [**Qwen2.5-1.5B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | Apache 2.0 | 2–3.5k tok/s | ~1–1.5 h | **Default.** Text-only, mature tooling, strong instruction following, no VLM plumbing. |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.87B | Apache 2.0 | 3–5k tok/s | ~0.6–1 h | Newer and smaller, but it is an image-text model; loading it for text SFT adds friction. Good fallback if you want the smallest. |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | Apache 2.0 | 2–3k tok/s | ~1–1.5 h | Solid open alternative; slightly weaker at strict instruction following. |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | Apache 2.0 | 10–20k tok/s | ~10–20 min | Only if you need a 3-hour demo; stance control will be noticeably weaker. |
| Llama-3.2-1B-Instruct | 1.24B | Llama 3.2 Community | 3–4k tok/s | ~1 h | Strong, but the license complicates publishing weights. |

\*Estimates for bf16 LoRA, seq 1024, gradient checkpointing, micro-batch 2–4. Measure to confirm.

**Start with Qwen2.5-1.5B-Instruct.** If training is too slow, drop to 0.8B; if quality is too weak, move up to 3B QLoRA only if you can accept ~4 h per adapter.

**Memory math (1.5B LoRA, bf16):** weights 3.0 GB + LoRA/optimizer ~0.2 GB + activations with checkpointing ~4–5 GB + logits (vocab 151k, batch 2 × seq 1024 ≈ 0.6 GB) ≈ **8–9 GB**. Fits 12 GB with micro-batch 2. If you OOM: seq 768, micro-batch 1, enable chunked cross-entropy (Liger kernel), or move to 4-bit QLoRA.

---

## 4. Data plan (the real work)

### 4.1 Shape

Aim for **8–12k examples per stance**, drawn from **100–150 topics** across 8 domains (health, technology, environment, education, economics, ethics, culture, science). Hold out **15–20% of topics** entirely — never train on them — so you can measure generalization to unseen topics.

Each record:

```json
{
  "topic": "Autonomous vehicles should be allowed to operate without a human driver",
  "stance": "pro",
  "argument": "The case for ... [3-5 sentences]",
  "meta": {"domain": "technology", "angle": "safety", "split": "train"}
}
```

Prompt template (identical at train and inference):

```
SYSTEM: You are a debate advocate. Argue {STANCE} the following proposition.
        Be persuasive and specific. Use 3-5 sentences with one concrete example.
        Never concede the opposing position.
USER:   {TOPIC}
```

### 4.2 How to generate it

Use a strong teacher model through an API — do not try to generate 20k samples with a local 9B at 4-bit on a 12 GB laptop, it will take days.

- **Volume:** 120 topics × 80 arguments × 2 stances ≈ 19k generations; ~6–8M output tokens.
- **Cost:** roughly **$5–15** at 2026 API prices, depending on the teacher (reasoning models cost more; a small instruct teacher is enough for this task).
- **Time:** 1–2 hours wall-clock with 16–32 concurrent requests and checkpointing every 500 records.
- **Diversity knobs:** vary persona (economist, engineer, parent, historian), argument angle (economic, ethical, practical, scientific), and length (short/medium). Without this, you get 19k paraphrases of the same three points.
- **Quality filters:** dedupe by MinHash, score stance with a classifier, drop toxic or factually dangerous outputs, and have a human read 50 random samples.

If you want to avoid API spend: seed with a public debate corpus (DebateSum, Persuade 2.0, ChangeMyView) and augment each seed with 3–5 paraphrases. Quality will be lower and formatting messier, but it is free.

### 4.3 Safety filtering

Maintain a blocklist of topics you will not advocate for (harm to identifiable groups, medical misinformation, illegal acts). Add ~200 refusal examples to training so the model learns to decline those topics, and enforce the blocklist in the UI as well. This is not optional if the demo is public.

---

## 5. Training recipe

### 5.1 LoRA configuration

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
method: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
max_seq_length: 1024
per_device_train_batch_size: 2
gradient_accumulation_steps: 8      # effective batch 16
learning_rate: 1.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03
num_train_epochs: 2
bf16: true
gradient_checkpointing: true
optim: adamw_torch
save_strategy: epoch
logging_steps: 20
```

Run it twice, once per stance, with the same seed and the same data order. Keep the adapters separate:

```
outputs/
├── pro-adapter/
└── con-adapter/
```

### 5.2 Time budget on your GPU

| Stage | Tokens | Est. time (1.5B, 2.5k tok/s) |
|---|---|---|
| Smoke test (50 steps) | ~0.1M | 2–3 min |
| Pro adapter, 2 epochs | 8M | ~55 min |
| Con adapter, 2 epochs | 8M | ~55 min |
| Merge + inference check | — | 10 min |
| **Total** | **16M** | **~2 h** |

Add 20% for throttling and restarts: **2.5 h**. That is the "2–3 hours" you asked about — and it is achievable *only* if the dataset is already built and the environment is already working.

### 5.3 Train both stances in one run?

You can train a single adapter with a stance token (`<PRO>` / `<CON>`) prepended, which halves training time and forces the model to share everything except stance behavior. It is elegant, but a bug in the token format silently collapses both stances into one. For a first project, two separate adapters are easier to debug and easier to explain. Keep the single-adapter version as a stretch comparison.

---

## 6. Evaluation that convinces a grader

Run all of these on the **held-out topic set**:

| Metric | How | Target |
|---|---|---|
| Stance adherence | Independent classifier + 50 human labels | ≥90% intended, ≤5% leakage |
| Topic relevance | LLM judge, 1–5 | ≥4.0 |
| Argument quality | Rubric: claim, evidence, reasoning, rebuttal (1–5) | ≥3.5 |
| Diversity | Distinct-2, self-BLEU across 5 samples/topic | Higher than baseline |
| Safety | Refusal rate on blocked topics; toxicity scan | Refuse ≥95% |
| Baseline delta | Same base, prompt-only, same topics | Fine-tune must win by a clear margin |
| Cost | GPU-hours per adapter + inference tok/s | Reported, not hidden |

The baseline is the most important row. If prompt-only Qwen2.5-1.5B already argues both sides well, your fine-tune needs to beat it on *unseen* topics, not just reproduce it.

---

## 7. The web app

### 7.1 Stack

- **Backend:** FastAPI. Load the base model once (~3.5 GB), keep both adapters resident, `set_adapter()` per request. Stream tokens with SSE. Serialize requests with a queue — one GPU, one generation at a time.
- **Frontend:** React + Vite + Tailwind + Framer Motion. One route, no router needed.
- **Why not Gradio:** it is 30 minutes of work but looks like a Gradio app. If "beautiful UI/UX" is a grading criterion, build the React page.

### 7.2 Screens

1. **Compose.** Large topic field, example chips ("Universal basic income", "AI-generated art should be copyrightable"), stance preview (two cards), temperature/length controls, "Start debate."
2. **Debate.** Split screen. Left column FOR (warm accent), right AGAINST (cool accent). Streaming typewriter text, avatars/badges, per-turn latency. A "Rebuttal" button feeds each side's last argument to the other.
3. **Verdict (optional).** A neutral judge model scores both sides on clarity, evidence, and persuasion. This is a great demo moment — and a second, cheap model call.
4. **About.** What was trained, on what data, compute cost, limitations, disclaimer.

### 7.3 Design language

Editorial, not "AI dashboard": off-white paper background, near-black text, one warm red for FOR and one deep blue for AGAINST, a serif display face for headings, generous whitespace, subtle motion only on streaming text. Dark mode optional. No purple gradients, no glassmorphism, no robot icons.

### 7.4 API contract

```
POST /api/debate  {topic, stance, temperature, max_tokens}  -> SSE stream
POST /api/rebuttal {topic, stance, opponent_argument}      -> SSE stream
GET  /api/health  -> {model, adapters, vram_used}
```

### 7.5 Build estimate

| Piece | Hours |
|---|---|
| FastAPI + adapter hot-swap + streaming | 3–4 |
| React page + styling + animations | 4–6 |
| Rebuttal loop + judge | 2 |
| Polish, error states, loading, mobile | 2 |
| **Total** | **11–14 h** |

That is why the full project is 2–3 days, not 2–3 hours.

---

## 8. Timeline

### 8.1 The 3-hour MVP (what actually fits)

| Hour | Task |
|---|---|
| 0:00–0:30 | Environment + benchmark; download base model |
| 0:30–1:00 | Load a small public debate dataset, format 3k examples |
| 1:00–2:00 | Train two 360M adapters (20 min each) |
| 2:00–2:30 | Gradio two-column demo |
| 2:30–3:00 | Eyeball 20 prompts; write down failure cases |

Deliverable: a working demo. Not a beautiful app, not a rigorous evaluation.

### 8.2 The full build (2–3 days)

| Day | Morning | Afternoon |
|---|---|---|
| 1 | Env, benchmark, topic list, teacher pipeline | Generate + filter 19k examples; split train/held-out |
| 2 | Train pro adapter, smoke test, eval harness | Train con adapter; run held-out eval |
| 3 | Build FastAPI + React UI | Integrate, polish, record 3-minute demo, write report |

GPU is idle most of day 1 and day 3 — that is normal. The GPU is not the bottleneck.

---

## 9. Cost

| Item | Amount | Cost |
|---|---|---|
| Electricity (200 W system × 10 h) | ~2 kWh | ~₹20 (~$0.25) |
| Teacher API for 19k generations | ~7M tokens | $5–15 |
| Cloud GPU backup (only if local fails) | 2 h A100 | $4–6 |
| **Total** | | **under $25** |

No cloud GPU is needed. Your laptop is sufficient.

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Blackwell/PyTorch mismatch | Medium | cu128 wheels; verify with a matmul before training; fall back to cloud T4 |
| Python 3.14 incompatibility | High | Use `uv` + Python 3.12; never install into system Python |
| OOM at seq 1024 | Medium | Micro-batch 1, seq 768, gradient checkpointing, chunked CE |
| Stance leakage | High | Best-of-N rerank, prefix forcing, more stance-balanced data |
| Fails on unseen topics | High | Hold out topics; report the gap honestly; add a curated-topic fallback in the UI |
| Data looks samey | High | Persona/angle diversity; dedupe; human-read 50 samples |
| Thermal throttling | Medium | Cap GPU power, elevate laptop, log tokens/s |
| Model produces harmful advocacy | Medium | Topic blocklist + refusal training + UI filter |
| "Not really two models" critique | Low | Merge adapters into two checkpoints for submission |
| Demo breaks during presentation | Medium | Pre-record a fallback video; cache 3 example debates |

---

## 11. Appendix

### 11.1 Benchmark script

Save as `bench.py` and run it before training:

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda()

batch, seq = 2, 1024
x = torch.randint(0, 1000, (batch, seq), device="cuda")

for i in range(3):  # warmup
    loss = model(x, labels=x).loss
    loss.backward(); model.zero_grad(set_to_none=True)

torch.cuda.synchronize(); t0 = time.time()
steps = 10
for _ in range(steps):
    loss = model(x, labels=x).loss
    loss.backward(); model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

tps = batch * seq * steps / (time.time() - t0)
print(f"{tps:,.0f} tokens/s  |  {tps*3600/1e6:.1f}M tokens/hour")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```

This measures forward+backward (the dominant cost). Full fine-tuning adds optimizer state; LoRA adds very little.

### 11.2 Minimal training script

Save as `train_stance.py`, run once per stance. This is deliberately small enough to debug in one sitting.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STANCE = "pro"                      # run again with "con"
DATA = f"data/{STANCE}_train.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)

SYSTEM = (
    f"You are a debate advocate. Argue {STANCE.upper()} the following proposition. "
    "Be persuasive and specific. Use 3-5 sentences with one concrete example. "
    "Never concede the opposing position."
)

def to_text(ex):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["topic"]},
        {"role": "assistant", "content": ex["argument"]},
    ]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}

ds = load_dataset("json", data_files=DATA, split="train").map(to_text)

args = SFTConfig(
    output_dir=f"outputs/{STANCE}-adapter",
    max_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    peft_config=LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
    processing_class=tok,
)
trainer.train()
trainer.save_model(f"outputs/{STANCE}-adapter")
```

If `trl` complains about argument names (`max_length` vs `max_seq_length`, `processing_class` vs `tokenizer`), check the installed version's signature — these APIs renamed twice in two years.

### 11.3 First-day checklist

- [ ] `uv` + Python 3.12 venv; torch cu128 prints `cuda True`
- [ ] `bench.py` reports tokens/s and peak VRAM
- [ ] Pick base model; download it; run one prompt
- [ ] Write the 120-topic list; mark 20 held-out
- [ ] Generate 200 examples as a pilot; read them
- [ ] Only then launch the full generation run

### 11.4 What to say in the report

The interesting claim is not "we built two chatbots." It is: *"At 1.5B parameters, stance-conditioned LoRA training on 120 topics generalizes to unseen topics at X% adherence, versus Y% for prompt-only control, for Z GPU-hours."* That sentence is your project.
