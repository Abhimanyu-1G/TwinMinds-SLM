# Small Language Models: A Project Playbook

**21 project blueprints, compute tiers, and a curated resource library for a Deep Learning course · November 2026**

Every entry below names a *testable hypothesis*, the smallest experiment that could falsify it, the compute it needs, and the resources to start today. The goal is not "fine-tune a model and demo it." The goal is a project that a good reviewer would call *science*: a claim, a control, a measurement, and an honest account of where it breaks.

---

## 0. The 60-second version

Small language models (SLMs) are the best vehicle for a course project in 2026 for three reasons: you can afford to run the experiment ten times, you can control the variables, and the result can ship on a phone. The catch is that "small" is not automatically interesting — the interesting part is the *claim* you test with it.

**Six flagship picks** (details in §5):

| # | Project | The claim you'd test | Compute |
|---|---------|----------------------|---------|
| A1 | **Speedrun Zero** | A GPT-2-class model can be trained for <$100, and one architecture knob explains most of the gap to the reference | 8×H100 for ~3 h, or 1×A100 for ~15 h |
| C1 | **Minimum Viable Reasoner** | GRPO on a 0.8–2B model produces self-verification, but only above a scale threshold you can locate | 1×A100 for ~30 h |
| B1 | **Professor & Student** | On-policy distillation beats supervised imitation of teacher traces at 0.5B, per dollar spent | 1×A100 for ~20 h |
| B2 | **The Quantization Frontier** | 3B models keep ≥95% quality at 4-bit and collapse below 3-bit — and a recovery LoRA buys back half the loss | 1×T4, mostly inference |
| F1 | **Microscope** | A 10M-parameter model trained on TinyStories contains interpretable circuits you can causally verify | Free T4, a weekend |
| E1 | **Pocket Agent** | A 2B tool-calling model on a phone is fast enough and accurate enough to replace a cloud API for a real workflow | No training; a phone |

What separates these from the usual course project: each one has a *falsifiable* claim, a *cost* attached, and an evaluation that a skeptic could rerun. Three of them (A1, B2, F1) can produce a genuinely publishable negative result if the hypothesis fails.

**How to read this document.** §1–§4 orient you (why SLMs, what exists, what compute buys, how to choose). §5 is the catalogue. §6–§9 are the operating manual: execution plan, resource library, ethics, and a 72-hour starter checklist.

---

## 1. Why small models are the best project vehicle in 2026

### 1.1 The loop is affordable

A 3B model at 4-bit is roughly 2 GB of weights — smaller than a phone's photo library. A 0.5B model fine-tunes on a free Colab T4. A 100M-parameter model can be pretrained from scratch on a single rented A100 in an afternoon for the price of a pizza. Compare that to a 70B project, where a single training run is a one-shot bet you cannot afford to repeat.

Affordability changes the *kind* of question you can ask. Instead of "does my fine-tune work?", you can ask "at what data size does it start to work?", "does the effect survive three seeds?", "which of these five design choices actually matters?"

The 2026 model landscape makes this easier than it has ever been. Qwen3.5 ships 0.8B/2B/4B/9B variants under Apache 2.0 with a 262K context window and native multimodal input ([model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)). Gemma 4's E2B/E4B are explicitly built for laptops and phones, and accept audio as well as images ([Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)). SmolLM3-3B publishes the full training recipe — data mixtures, mid-training stages, alignment method — which makes it uniquely useful for *studying* training rather than just consuming a model ([SmolLM3 blog](https://huggingface.co/blog/smollm3)).

### 1.2 You can control the experiment

The dirty secret of large-model research is that most published ablations are single runs at a scale nobody can afford to repeat. At 10M–1B parameters you can run a proper grid: five data mixtures × three seeds × two architectures, and still finish before the deadline. You can fit scaling curves instead of drawing lines through two points. You can open the model and look at its activations.

This is why the most cited small-model papers are small for a reason. TinyStories exists because coherence emerges below 10M parameters, where you can train dozens of models ([TinyStories](https://arxiv.org/abs/2305.07759)). phi-1 showed that data quality could substitute for scale at 1.3B ([Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)). Neither result required a frontier lab; both required a controlled comparison.

### 1.3 The artifact can ship

A project that ends with a working on-device demo is memorable in a way a notebook of loss curves is not. The deployment path is now boring in the best sense: export to GGUF or ExecuTorch, run with llama.cpp or MLC, measure tokens per second and memory. A 2B model at 4-bit fits in a phone's RAM with room for a KV cache; a 0.8B model runs on a CPU.

That matters for the grade, but it matters more for the argument. "SLMs are sufficient for specialized tasks" is a claim about *deployment economics*, and the only way to defend it is to measure latency, memory, energy, and quality on the hardware where the model would actually run.

### 1.4 The honest caveat

SLMs still lose badly on knowledge-heavy question answering, long-horizon planning, and tasks that require synthesizing many documents. The [2026 survey of ~160 SLM papers](https://arxiv.org/abs/2501.05465) is careful about this: small models are competitive *within their regimes*, not everywhere. The [NVIDIA position paper](https://arxiv.org/abs/2506.02153) makes the stronger claim that most agentic invocations are narrow, repetitive, and therefore better served by SLMs — but it is a position paper, not a measurement.

That tension is the opportunity. The best course projects do not try to prove "small is better." They map the boundary: *for which tasks, at what cost, with what failure modes?* A project that shows exactly where a 2B model breaks is more valuable than one that shows it works on three cherry-picked prompts.

---

## 2. The 2026 SLM landscape

All numbers below were checked against the Hugging Face model cards and `config.json` files in November 2026. Parameter counts are safetensors totals and therefore include vision/audio encoders where present; they can exceed the nominal family size.

| Model family | Params | Context | License | Why you'd pick it |
|---|---|---|---|---|
| **Qwen3.5** 0.8B / 2B / 4B / 9B | 0.87B / ~2.3B / 4.66B / 9.65B | 262,144 | Apache 2.0 | The most capable general family at small sizes: native image/video input, documented tool parsers, strong reasoning per parameter. Start here unless you have a reason not to. |
| **Gemma 4** E2B / E4B | 5.1B / 8.0B total (2.3B / 4.5B effective) | 131,072 | Apache 2.0 | Built for edge: per-layer embeddings keep *active* compute low while total storage is larger. E-models accept text, image, and audio. |
| **SmolLM3** 3B | 3.08B | 65,536 (128K with YaRN) | Apache 2.0 | The best-documented fully open recipe: 11.2T tokens, three pretraining stages, NoPE every 4th layer, dual `think`/`no_think` modes. Pick this to study *how* a model is built. |
| **Granite 4.2** 3B / 8B | 3.66B / ~8.8B | 131,072 (extendable to 512K) | Apache 2.0 | Tool calling and agentic workflows as first-class use cases; compact dense models with long context. |
| **MiniCPM5** 1B / 2B | 2.52B (2B) | 131,072 | Apache 2.0 | On-device focus, long context, tool use; strong quality per byte at the 2B point. |
| **Phi-4-mini** | 3.84B | 131,072 | MIT | The "textbook data" lineage; strong reasoning for its size. A useful contrast to web-data models in a data-quality study. |
| **Ministral 3** 3B / 8B / 14B | 8.92B (8B) | 262,144 | Apache 2.0 | Vision encoder plus native function calling; fits a 12 GB card in FP8. |
| **Olmo 3** 7B | 7.30B | 65,536 | Apache 2.0 | Fully open: weights, data mixtures, and training code. If your project is about *data*, this is the cleanest baseline. |
| **SmolVLM2** 2.2B | 2.25B | — | Apache 2.0 | Video/image understanding on-device; the small multimodal baseline. |
| **BitNet b1.58** 2B4T | 2B (ternary) | — | MIT | 1.58-bit weights; the reference point for extreme quantization and CPU inference. |
| **Llama 3.2** 1B / 3B | 1.24B / 3.21B | 131,072 | Llama 3.2 Community | Ubiquitous baseline. Note the license is *not* OSI-approved, which matters if you plan to publish weights. |

**What changed in 2026, in one paragraph.** Small models stopped being text-only. Every size in the Qwen3.5 family takes images, Gemma 4's E-models take audio, and Ministral 3 pairs a vision encoder with native function calling. Context windows of 128K–262K are now normal, made affordable by hybrid attention that keeps a growing KV cache in only a fraction of layers. And vendors now publish function-calling and agent benchmarks on the model card, which means you can compare tool-use ability without building your own harness — though you should still verify with [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) or [τ-bench](https://arxiv.org/abs/2406.12045), because each vendor runs its own settings.

**A note on choosing a base model.** For a controlled study, the base model matters less than you think — you need one whose training data and architecture you can describe. Prefer Apache 2.0 models with published recipes (SmolLM3, Olmo 3, Qwen3.5). Avoid gated models if your compute environment has no HF token. Do not pick Llama 3.2 for a project that publishes derived weights unless you have read the license.

---

## 3. Compute: the constraint that shapes the project

### 3.1 The only formula you need

Training compute for a dense transformer is approximately:

```
FLOPs ≈ 6 × N × D
```

where `N` is parameters and `D` is training tokens. Inference is roughly `2 × N` FLOPs per generated token. From there:

| Hardware | Effective throughput (bf16, realistic MFU) | FLOPs per GPU-hour |
|---|---|---|
| T4 (free Colab/Kaggle) | ~8×10¹² FLOP/s | ~2.9×10¹⁶ |
| A100 40/80GB | ~1.2×10¹⁴ FLOP/s | ~4.3×10¹⁷ |
| H100 | ~4×10¹⁴ FLOP/s | ~1.4×10¹⁸ |

*Worked example:* a 100M model on 2B tokens is `6 × 10⁸ × 2×10⁹ = 1.2×10¹⁸` FLOPs ≈ **2.8 A100-hours** or **41 T4-hours**. That is the difference between "one afternoon, $5" and "a week of free Colab."

### 3.2 Four tiers

| Tier | What you have | What it unlocks | Typical cost |
|---|---|---|---|
| **A** | No GPU | Data generation via API, evaluation of hosted models, benchmark construction, router/cascade experiments, analysis of public checkpoints | $0–30 in API credits |
| **B** | Free T4 16GB (Colab/Kaggle) | LoRA SFT up to ~3B, pretraining 1M–30M models, interpretability, quantization evaluation, TinyStories-scale science | $0 |
| **C** | 1×A100 40GB | GRPO up to ~2B, distillation into 0.5B, pretraining 100M–200M, continued pretraining of 0.5B | ~$1.5–2.5/h; a 30 h project is $45–75 |
| **D** | 8×H100 / 8×A100 | GPT-2-class speedruns, 1B pretraining, multi-arm architecture studies at 300M+ | ~$20–25/h; a 3 h run is $60–75 |

**Rental options:** [Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/code) for free tiers; [Lightning](https://lightning.ai/), [Modal](https://modal.com/), [RunPod](https://runpod.io/) for on-demand A100/H100; [HF Jobs](https://huggingface.co/docs/hub/jobs) if your workflow already lives on the Hub. University clusters are usually the cheapest A100s you have access to — ask before renting.

### 3.3 The compute ladder

Do not start at the top. The standard failure mode is renting an 8×H100 node for a project whose code has never run. Climb:

1. **Debug at 1/100 scale** on CPU or a T4 (a 5M model, 10M tokens).
2. **Calibrate hyperparameters** at 1/10 scale (a 30M model, 100M tokens).
3. **Run the real experiment** once you can predict its loss curve.
4. **Only then** spend on the headline run.

A useful rule: if you cannot explain why the loss at step 1,000 is what it is, you are not ready for the expensive run.

---

## 4. How to choose

### 4.1 Four questions

1. **Do you want to build a model or study one?** Building (pretrain/post-train) teaches the stack; studying (eval, interpretability, quantization) teaches measurement. Both are legitimate; the best projects do one deeply and the other lightly.
2. **What is your hardware?** Be honest. A T4 project that runs 30 controlled experiments beats an A100 project that runs one.
3. **What artifact do you want to show?** A phone demo, a leaderboard, a paper-style report, or a library. Work backwards from the demo.
4. **What would make you say "I was wrong"?** If you cannot answer this, you have a demo, not a project.

### 4.2 Five archetypes that impress reviewers

- **The controlled ablation.** Reproduce a known result, then change exactly one thing and measure the delta with seeds and error bars.
- **The reproduction + extension.** Rebuild a published pipeline at smaller scale, then push it somewhere the paper did not go (a new language, a new domain, a new hardware target).
- **The benchmark.** Build a small, well-designed evaluation for a niche nobody measures, and show it ranks models differently from MMLU.
- **The deployable artifact.** Ship a model under real constraints (phone, CPU, offline) and report latency, memory, energy, and failure cases.
- **The negative result.** Show that a popular trick does *not* work at small scale, with enough controls to be convincing. This is undervalued and often the most cited kind of course project.

### 4.3 Map your interest to a project

| If you like... | Start with |
|---|---|
| Training systems, scaling laws | A1, A3, D1 |
| Data curation and synthetic data | A3, C2, E3 |
| RL and post-training | C1, C2, C3 |
| Efficiency, kernels, quantization | B2, B3, D3 |
| Interpretability and science | F1, D2 |
| Product and deployment | E1, E2, E4 |
| Evaluation and measurement | F2, B2, C3 |
| Safety and red-teaming | F3, E2 |
| Multimodal / audio | G1, E1 |
| Theory-curious, want a weird result | G2, D2 |

**If you only have a laptop:** F1 (train 10M models on CPU/T4), F2 (evaluate hosted models), B2 (quantize and benchmark on CPU), A3 (data work plus a tiny model), or E4 (build the evaluation and router around API models, then fine-tune later).

---

## 5. The projects

Each entry has the same anatomy: a hypothesis, why it is interesting, what you will learn, a build sequence, the measurement that decides it, resources, and a stretch goal. Difficulty is 1–5 (5 = needs real engineering or multi-GPU). Compute tags refer to §3.2.

---

### Track A — Build from scratch

*Train the model yourself. This is the track that teaches the most, and the one where a small scale is an advantage rather than an apology.*

#### A1 · Speedrun Zero — a GPT-2-class model for the price of a textbook
`Difficulty 4/5 · Compute D (or C with a smaller depth)`

**Hypothesis:** *At fixed compute, one architectural choice (GQA, NoPE, QK-norm, or vocabulary size) accounts for most of the gap between a budget GPT-2 reproduction and the reference, and its effect is measurable at 1/10 the cost.*

**Why it's interesting:** Karpathy's [nanochat](https://github.com/karpathy/nanochat) turned "train GPT-2" into a \$48, two-hour exercise and maintains a public time-to-GPT-2 leaderboard. That makes it the rare project where you can reproduce a frontier-adjacent result and then *perturb* it. Most students run the speedrun and stop; almost nobody runs the ablation grid.

**What you'll learn:** the full stack — tokenizer, pretraining, mixed precision, distributed data parallel, evaluation with bits-per-byte and the DCLM CORE score, and cost accounting.

**Build:**
1. Run the speedrun at `depth=26` on 8×H100 (or `depth=12` on one A100, accepting a weaker model).
2. Instrument it: log loss, tokens/s, MFU, and dollars per run.
3. Pick one knob and run a 3-point sweep at `depth=12` with 3 seeds (GQA on/off, NoPE every 4th layer, vocab 32K vs 128K).
4. Fit a small scaling curve across depth 8/12/16 and report the exponent you observe.

**Measure:** validation bits-per-byte, CORE, and **cost per point of CORE**. The cost axis is what makes the project original.

**Resources:** [nanochat](https://github.com/karpathy/nanochat) · [llm.c](https://github.com/karpathy/llm.c) · [nanoGPT](https://github.com/karpathy/nanoGPT) · [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) · [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)

**Stretch:** submit to the nanochat leaderboard; compare your scaling exponent to Chinchilla's 20-tokens-per-parameter rule; test whether the ablation conclusion holds at `depth=26` on a single run.

**Risk:** distributed training bugs. Mitigate by running the single-GPU path first — the repo supports it with gradient accumulation.

---

#### A2 · TinyStories lab — the smallest model that can speak English
`Difficulty 2/5 · Compute B`

**Hypothesis:** *Coherence emerges between roughly 4M and 10M parameters on TinyStories, and the threshold depends more on data repetition and tokenizer choice than on depth.*

**Why it's interesting:** [TinyStories](https://arxiv.org/abs/2305.07759) showed that models below 10M parameters can produce fluent, grammatical, multi-paragraph stories — and that a single transformer block can be enough. It is the cheapest serious experiment in language modeling, and it lets you answer a question that scales badly at 70B: *what is the minimum viable model for a capability?*

**What you'll learn:** scaling behavior at the small end, the effect of tokenization (character vs BPE), curriculum effects (simple stories first?), and how to evaluate generated text with a rubric plus an LLM judge.

**Build:**
1. Train 1M / 4M / 10M / 33M models on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with a fixed token budget.
2. Repeat at 10M with character-level and BPE tokenizers; repeat with a simple→complex curriculum.
3. Score 200 generations per model with a 5-point rubric (grammar, consistency, creativity, plot, ending) using an LLM judge, and validate 30 of them by hand.

**Measure:** judge scores with confidence intervals, held-out perplexity, and the *parameter count at which grammar crosses 4/5*.

**Resources:** [TinyStories paper](https://arxiv.org/abs/2305.07759) · [dataset](https://huggingface.co/datasets/roneneldan/TinyStories) · [nanoGPT](https://github.com/karpathy/nanoGPT) (character-level baseline) · [SmolLM3 recipe](https://huggingface.co/blog/smollm3) for tokenizer practice

**Stretch:** a one-block transformer at 10M; test whether the emergence threshold shifts when the vocabulary is restricted to the TinyStories word list.

---

#### A3 · Textbook factory — can synthetic data beat web data at 100M scale?
`Difficulty 3/5 · Compute B/C + ~$20–50 API`

**Hypothesis:** *For a narrow domain, 1B tokens of synthetic "textbook quality" data produce a better 100M model than 10B tokens of filtered web data, at equal compute.*

**Why it's interesting:** phi-1's central claim was that data quality can substitute for scale ([Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)). It is usually cited, rarely re-tested at a scale a student can afford. You can re-test it for a domain you understand — introductory Python, organic chemistry, accounting, contract law — and publish the data mix as an artifact.

**What you'll learn:** data curation, synthetic generation with a teacher model, deduplication, contamination control, and how to compare corpora at fixed compute rather than fixed tokens.

**Build:**
1. Choose a domain and define a held-out evaluation *before* generating data (e.g., 300 exam-style questions and 100 coding exercises).
2. Build three corpora: raw web subset, filtered web ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), and synthetic textbooks+exercises ([Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) style, generated with [distilabel](https://github.com/argilla-io/distilabel) or a scripted teacher).
3. Train the same 100M model on each at equal token budget; add a 10× web-token arm to test the "quality beats quantity" claim.
4. Evaluate domain accuracy, general regression (MMLU subset), and contamination (n-gram overlap with the eval set).

**Measure:** domain accuracy vs tokens vs dollars, plus a contamination audit. Report the *break-even* point where synthetic data stops paying for itself.

**Resources:** [phi-1](https://arxiv.org/abs/2306.11644) · [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) · [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) · [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) · [distilabel](https://github.com/argilla-io/distilabel)

**Stretch:** vary the synthetic/web mix ratio; test whether deduplication matters more than generation quality.

---

### Track B — Compress and accelerate

*Turn a big model into a small one, or make a small one cheaper to run. This track produces the most portable skills and the clearest Pareto plots.*

#### B1 · Professor & Student — on-policy distillation from 7B to 0.5B
`Difficulty 3/5 · Compute C`

**Hypothesis:** *At 0.5B, on-policy distillation with reverse KL beats supervised fine-tuning on teacher traces by at least 5 points on instruction-following, at roughly twice the training compute — and the gap widens for smaller students.*

**Why it's interesting:** [MiniLLM](https://arxiv.org/abs/2306.08543) argued that reverse KL is the right objective for generative distillation, because forward KL lets the student waste probability mass on regions the teacher would never visit. Most course projects still do "generate traces, SFT on them." Comparing the two at a scale where you can run both is a clean, useful experiment.

**What you'll learn:** knowledge distillation, on-policy vs off-policy training, exposure bias, temperature and top-p sampling, and the economics of quality per dollar.

**Build:**
1. Pick a teacher (Qwen3.5-9B locally if you have a GPU, or an API model) and a 20K-prompt set from [OpenThoughts](https://github.com/open-thoughts/open-thoughts) or [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
2. Train three 0.5B students: (a) SFT on teacher outputs, (b) sequence-level KD, (c) on-policy reverse-KL ([TRL GKD](https://huggingface.co/docs/trl/gkd_trainer)).
3. Evaluate on IFEval, a held-out instruction set with an LLM judge, and a small math slice. Track GPU-hours and teacher-token cost for each arm.

**Measure:** win rate vs the SFT student at matched cost, IFEval, and a quality-vs-dollars curve.

**Resources:** [MiniLLM](https://arxiv.org/abs/2306.08543) · [TRL GKD trainer](https://huggingface.co/docs/trl/gkd_trainer) · [LMOps](https://github.com/microsoft/LMOps) · [OpenThoughts](https://github.com/open-thoughts/open-thoughts)

**Stretch:** iterative distillation (student generates, teacher corrects, repeat); best-of-N teacher samples; distil a *reasoning* model and measure whether thinking traces survive the compression.

---

#### B2 · The Quantization Frontier — how low can 3B go?
`Difficulty 2/5 · Compute B`

**Hypothesis:** *A 3B model keeps ≥95% of its benchmark score at 4-bit, loses more than 10% at 2-bit, and a short recovery LoRA closes about half of the 2-bit gap.*

**Why it's interesting:** quantization is the single highest-leverage deployment trick, and the trade-off curve is genuinely non-obvious: some tasks collapse early, others survive 2-bit, and the ranking of methods changes with the metric. A careful Pareto study across three models and four bit-widths is a useful artifact that people will actually read.

**What you'll learn:** post-training quantization (GPTQ, AWQ, GGUF K-quants), calibration data, group size and scale granularity, kernel effects, and how to measure quality per gigabyte and per joule.

**Build:**
1. Pick three models at different sizes (e.g., Qwen3.5-0.8B, SmolLM3-3B, Qwen3.5-9B) and five tasks (MMLU subset, GSM8K, IFEval, a domain task, and long-context retrieval).
2. Quantize to 8/4/3/2-bit with two methods ([llm-compressor](https://github.com/vllm-project/llm-compressor), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize)).
3. Measure accuracy, tokens/s, peak RAM/VRAM, and energy per 1K tokens on both GPU and CPU.
4. Add a recovery LoRA at the worst bit-width and re-measure.

**Measure:** the Pareto frontier — accuracy lost vs memory saved vs latency gained — and the *task-dependent* breakpoint.

**Resources:** [GPTQ](https://arxiv.org/abs/2210.17323) · [AWQ](https://arxiv.org/abs/2306.00978) · [llm-compressor](https://github.com/vllm-project/llm-compressor) · [torchao](https://github.com/pytorch/ao) · [BitNet](https://github.com/microsoft/BitNet) for the extreme end

**Stretch:** quantization-aware training at 2-bit; compare ternary BitNet weights against 2-bit PTQ at equal memory.

---

#### B3 · Home-grown draft model — speculative decoding you can explain
`Difficulty 4/5 · Compute C`

**Hypothesis:** *A 100M draft model distilled from a 3B target yields ≥2× single-batch speedup at equal output quality, and most of the win comes from acceptance rate rather than draft latency.*

**Why it's interesting:** [Medusa](https://arxiv.org/abs/2401.10774) and [EAGLE-2](https://arxiv.org/abs/2406.16858) made speculative decoding practical, but most students treat it as a library flag. Building the draft model yourself — and measuring why it works — turns a systems trick into a clear story about the trade-off between parallelism and verification.

**What you'll learn:** speculative sampling, acceptance rates, tree attention, KV-cache reuse, and latency decomposition (draft vs verify vs overhead).

**Build:**
1. Distill a 100M draft from the target on 1–2B tokens of target-generated text.
2. Add Medusa-style heads to the target (or use [speculators](https://github.com/vllm-project/speculators) to train an EAGLE-style draft).
3. Benchmark with [vLLM](https://github.com/vllm-project/vllm) at batch size 1 and 8: tokens/s, acceptance length, quality parity.
4. Ablate draft size (50M/100M/200M) and temperature.

**Measure:** speedup vs acceptance rate vs draft cost; a latency breakdown that shows where the time goes.

**Resources:** [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE-2](https://arxiv.org/abs/2406.16858) · [speculators](https://github.com/vllm-project/speculators) · [vLLM](https://github.com/vllm-project/vllm)

**Stretch:** run the draft on CPU while the target runs on GPU; test whether a *smaller* draft wins at high temperature.

---

### Track C — Reason and align

*Post-training is where small models stop being "a smaller chatbot" and become a specific tool. This track is the most crowded in 2025–26, so the differentiator is rigor: control arms, matched compute, and an honest account of reward hacking.*

#### C1 · Minimum Viable Reasoner — RLVR at 0.8–2B
`Difficulty 4/5 · Compute C`

**Hypothesis:** *GRPO with verifiable rewards lifts a 0.8B model from near-chance to >50% on Countdown without any SFT, but self-verification behavior ("wait, let me check") only appears above a scale threshold you can locate between 0.8B and 2B.*

**Why it's interesting:** DeepSeek-R1 showed that pure RL can elicit self-reflection in large models ([DeepSeek-R1](https://arxiv.org/abs/2501.12948)). [TinyZero](https://github.com/Jiayi-Pan/TinyZero) reproduced it at 3B for about \$30 and became one of the most forked repos in the space. What almost nobody has done carefully is the *threshold* question: at what scale does the model start to correct itself, and is that a phase transition or a smooth curve? A negative result here ("no self-verification below 2B; what looks like reflection is template matching") would be genuinely interesting.

**What you'll learn:** policy-gradient RL, group-relative advantages, KL regularization, reward design, length bias, and the difference between "learned to reason" and "learned to produce the shape of reasoning."

**Build:**
1. Choose a verifiable task (Countdown, GSM8K, or a synthetic arithmetic task) and a reward: exact match + a small format bonus.
2. Run [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) on Qwen3.5-0.8B and Qwen3.5-2B with identical hyperparameters and 3 seeds at 0.8B.
3. Log accuracy, response length, and the rate of self-correction phrases over training. Hand-label 100 rollouts per checkpoint to check whether "reflection" is real.
4. Ablate: reward format weight, KL coefficient, group size, and a no-format-reward arm.

**Measure:** task accuracy, length, self-correction rate (human-validated), OOD transfer to held-out math, and a reward-hacking audit (does the model get the format right and the answer wrong?).

**Resources:** [TinyZero](https://github.com/Jiayi-Pan/TinyZero) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) · [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) · [verl](https://github.com/volcengine/verl) · [RLVR-GSM](https://huggingface.co/datasets/allenai/RLVR-GSM)

**Stretch:** curriculum from easy to hard; compare against SFT on R1 traces at matched compute; add a process reward model and see whether it changes the threshold.

**Risk:** GRPO is compute-hungry because of generation. Keep the prompt set small (2–5K), use vLLM for rollouts, and start at 0.8B.

---

#### C2 · Distilling thought — how few examples teach reasoning?
`Difficulty 2/5 · Compute B/C`

**Hypothesis:** *At 0.5B, 800 carefully selected reasoning traces outperform 20,000 random traces on MATH500, and inference-time "budget forcing" adds another 3–5 points.*

**Why it's interesting:** [LIMO](https://arxiv.org/abs/2502.03387) argued that sophisticated reasoning can emerge from a tiny, curated set of demonstrations, and [s1](https://arxiv.org/abs/2501.19393) showed that 1,000 questions plus budget forcing produce strong test-time scaling. Both results are about *data selection*, not data quantity — which is exactly the kind of claim you can test at 0.5B for the price of a few GPU-hours.

**What you'll learn:** supervised fine-tuning, chain-of-thought data, difficulty/diversity/quality selection, test-time compute control, and the difference between in-distribution and OOD reasoning.

**Build:**
1. Curate three SFT sets from [OpenThoughts](https://github.com/open-thoughts/open-thoughts) / [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT): 800 curated, 800 random, 20K random.
2. Fine-tune 0.5B (and optionally 3B) students on each; evaluate on MATH500, GSM8K, and a held-out OOD set.
3. Add budget forcing at inference (cap the thinking tokens; append "Wait" when the model tries to stop) and sweep the token budget.
4. Plot accuracy vs number of examples and vs inference tokens — two curves that together tell the whole story.

**Measure:** accuracy at fixed training cost, accuracy vs inference budget, and OOD generalization. Report whether curation or quantity dominates at this scale.

**Resources:** [LIMO](https://arxiv.org/abs/2502.03387) · [s1](https://arxiv.org/abs/2501.19393) · [s1 code](https://github.com/simplescaling/s1) · [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) · [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)

**Stretch:** rejection sampling from the student's own attempts; self-distillation; test whether the 800-example set transfers across model families.

---

#### C3 · Alignment bake-off — DPO vs KTO vs APO on 3B
`Difficulty 3/5 · Compute C`

**Hypothesis:** *On the same preference data, DPO, KTO, and APO reach similar judge win rates, but differ systematically in verbosity and refusal behavior, and APO degrades least when preference data is scarce.*

**Why it's interesting:** SmolLM3's recipe used Anchored Preference Optimization (APO), a DPO variant, after a careful comparison ([SmolLM3 blog](https://huggingface.co/blog/smollm3)). Most students use DPO by default. A controlled comparison at 2–3B — same data, same budget, three objectives — is a compact, well-defined experiment with real practical value.

**What you'll learn:** preference optimization theory (Bradley–Terry, implicit rewards), reference-model effects, length bias, and how to evaluate alignment without fooling yourself with a judge.

**Build:**
1. Build a 20K preference set from [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) or SmolTalk preferences.
2. Train three LoRA adapters on the same base: DPO, KTO, and an APO-style variant ([TRL DPO](https://huggingface.co/docs/trl/dpo_trainer), [alignment-handbook](https://github.com/huggingface/alignment-handbook)).
3. Evaluate with pairwise LLM judging (position-swapped), IFEval, a safety/refusal set, and a verbosity metric.
4. Repeat at 25% of the preference data to test the low-data claim.

**Measure:** win rate with confidence intervals, average response length, refusal rate on benign vs harmful prompts, and IFEval.

**Resources:** [DPO paper](https://arxiv.org/abs/2305.18290) · [KTO paper](https://arxiv.org/abs/2402.01306) · [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer) · [alignment-handbook](https://github.com/huggingface/alignment-handbook) · [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

**Stretch:** online iterative DPO with fresh on-policy pairs; train a 0.5B reward model and compare it against the LLM judge.

---

### Track D — Architecture

*Small scale is the only place where architecture questions are answerable in a semester. At 100–300M parameters you can train several arms under identical conditions and still afford seeds.*

#### D1 · Attention is not all you need — Transformer vs Mamba vs hybrid at 200M
`Difficulty 4/5 · Compute C`

**Hypothesis:** *A 3:1 hybrid (three linear/SSM blocks per full-attention block) matches a dense Transformer's perplexity at 200M but is at least 2× faster at 32K context, with retrieval parity on needle-in-a-haystack.*

**Why it's interesting:** 2026's small models are full of hybrid attention: Qwen3.5 keeps a growing KV cache in only a fraction of layers, and SmolLM3 removes RoPE from every fourth layer ([SmolLM3 blog](https://huggingface.co/blog/smollm3)). The design space is large and the trade-offs are poorly documented at small scale. A clean three-arm comparison — dense, pure SSM, hybrid — is a real contribution.

**What you'll learn:** state-space models, linear attention, KV-cache economics, long-context evaluation, and throughput measurement (tokens/s is not the same as time-to-first-token).

**Build:**
1. Start from a 200M dense baseline (nanoGPT-style) trained on 2B tokens of FineWeb-Edu.
2. Replace blocks with [Mamba-2](https://github.com/state-spaces/mamba) or a linear-attention block from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) at ratios 1:1, 3:1, and 7:1.
3. Train each arm for the same tokens; evaluate perplexity, needle-in-a-haystack at 4K/16K/32K, and throughput at each length.

**Measure:** perplexity, retrieval accuracy, tokens/s, KV bytes per token, and the length at which the hybrid overtakes dense.

**Resources:** [Mamba](https://github.com/state-spaces/mamba) · [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) · [RULER](https://github.com/NVIDIA/RULER) · [SmolLM3 architecture notes](https://huggingface.co/blog/smollm3)

**Stretch:** add NoPE to the hybrid; sweep the ratio; test whether the conclusion survives at 400M.

---

#### D2 · Small MoE — eight experts, one GPU
`Difficulty 4/5 · Compute C`

**Hypothesis:** *At iso-FLOPs, a 200M-total / 50M-active top-2 MoE beats a dense 50M model by >10% in loss and approaches a dense 200M at roughly one-third the compute.*

**Why it's interesting:** mixture-of-experts is how frontier models get capability per FLOP, but the interesting regime for a course project is the tiny one: does sparsity still pay when experts are small, and what happens to routing? [OLMoE](https://arxiv.org/abs/2409.02060) released a fully open MoE; [DeepSeekMoE](https://arxiv.org/abs/2401.06066) formalized fine-grained expert segmentation. You can build a 200M MoE on one GPU and watch the router.

**What you'll learn:** sparse routing, load balancing, capacity factors, expert specialization, and the difference between parameter count and active compute.

**Build:**
1. Implement top-2 routing in a nanoGPT-style codebase (or use [Megablocks](https://github.com/stanford-futuredata/megablocks) / [Tutel](https://github.com/microsoft/tutel)).
2. Train three arms at equal tokens: dense 50M, dense 200M, MoE 200M/50M active. Add a load-balancing loss and log routing entropy.
3. Evaluate loss at iso-FLOPs and iso-tokens; probe expert specialization by domain (code vs prose vs math).

**Measure:** loss vs active FLOPs, routing entropy, expert-domain overlap, throughput, and memory.

**Resources:** [OLMoE](https://arxiv.org/abs/2409.02060) · [DeepSeekMoE](https://arxiv.org/abs/2401.06066) · [Megablocks](https://github.com/stanford-futuredata/megablocks) · [Tutel](https://github.com/microsoft/tutel)

**Stretch:** upcycle a dense checkpoint into an MoE and continue training; test expert dropout as a regularizer.

---

#### D3 · 1.58 bits from scratch — BitNet at 100M
`Difficulty 3/5 · Compute C`

**Hypothesis:** *A ternary 100M model matches an fp16 100M model within 2% perplexity at equal tokens, and runs 2–3× faster on CPU at one-quarter the memory.*

**Why it's interesting:** [BitNet b1.58](https://arxiv.org/abs/2402.17764) claims ternary weights match full precision at equal size and tokens, and Microsoft shipped a 2B4T model plus a CPU inference stack ([BitNet repo](https://github.com/microsoft/BitNet)). Most students only *quantize* a trained model; training ternary from scratch is a different lesson — you feel the straight-through estimator in your bones.

**What you'll learn:** quantization-aware training, straight-through gradients, weight scaling, kernel efficiency, and energy measurement.

**Build:**
1. Implement BitLinear layers (ternary weights, 8-bit activations) in a 100M transformer.
2. Train on 2B tokens of FineWeb-Edu; compare against an fp16 twin with identical data order and seeds.
3. Evaluate perplexity and a few downstream tasks; measure CPU tokens/s and joules per token on a laptop.
4. If time permits, export to the BitNet inference stack and compare against a 4-bit GGUF baseline.

**Measure:** perplexity delta at equal tokens, tokens/s, memory, and energy per token on CPU.

**Resources:** [BitNet paper](https://arxiv.org/abs/2402.17764) · [microsoft/BitNet](https://github.com/microsoft/BitNet) · [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

**Stretch:** fine-tune the released 2B4T on a domain task; combine ternary training with distillation from an fp16 teacher; fit a scaling law for ternary models.

---

### Track E — Deploy and apply

*The artifact track. These projects are judged by whether the thing works under real constraints — latency, memory, cost, and failure modes — not by a benchmark delta alone.*

#### E1 · Pocket Agent — a 2B tool-caller on a phone
`Difficulty 4/5 · Compute A/B + a phone`

**Hypothesis:** *A 2B model at 4-bit, with speculative decoding, completes a three-tool workflow in under one second per turn on a mid-range phone while matching a cloud 70B on task success for that workflow.*

**Why it's interesting:** the "SLMs are the future of agentic AI" argument ([NVIDIA position paper](https://arxiv.org/abs/2506.02153)) is ultimately about deployment economics. A demo that runs offline, on a phone, with measured latency and battery drain is the strongest possible evidence — and a great demo. The 2026 toolchain makes it feasible: ExecuTorch and MLC-LLM both target mobile, and 0.8–2B models fit comfortably.

**What you'll learn:** export and quantization for mobile runtimes, KV-cache management, latency decomposition (prefill vs decode), and the gap between benchmark accuracy and real task success.

**Build:**
1. Pick a workflow (calendar + weather + messaging; or a local file Q&A) and define 50 test tasks with expected tool calls.
2. Export Gemma 4 E2B or Qwen3.5-0.8B to ExecuTorch or MLC; build a minimal CLI or Android app.
3. Measure time-to-first-token, tokens/s, peak RAM, and battery per 50 turns; compare against a hosted API on the same tasks.
4. Add one optimization (4-bit KV cache, speculative decoding, prompt caching) and re-measure.

**Measure:** task success rate, p50/p95 latency, RAM, energy, and cost per 1K turns vs cloud.

**Resources:** [ExecuTorch](https://github.com/pytorch/executorch) · [MLC-LLM](https://github.com/mlc-ai/mlc-llm) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) · [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)

**Stretch:** add voice input; offline RAG over local notes; compare ExecuTorch vs MLC on the same device.

**Risk:** mobile toolchains are the hard part. Have a laptop/desktop fallback (llama.cpp) so the project still ships if the phone build slips.

---

#### E2 · Specialist with receipts — a domain SLM that cites its sources
`Difficulty 3/5 · Compute B/C`

**Hypothesis:** *Fine-tuning a 3B model on domain QA beats RAG-only on answer accuracy, but RAG-only wins on attribution precision; the combination beats both on both metrics.*

**Why it's interesting:** "fine-tune vs RAG" is a perennial argument with surprisingly little controlled evidence at small scale. Domain work also forces you to confront hallucination and abstention, which are the failure modes that matter in deployment. Medical and legal domains have public benchmarks, which makes the evaluation credible.

**What you'll learn:** LoRA/QLoRA, domain adaptation, retrieval pipelines, citation-grounded generation, calibration, and evaluation with expert rubrics.

**Build:**
1. Build a 3–5K QA training set from [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA), or [LegalBench](https://huggingface.co/datasets/nguha/legalbench).
2. Fine-tune a 3B model with QLoRA; build a RAG baseline with [FAISS](https://github.com/facebookresearch/faiss) over a domain corpus; build the hybrid.
3. Evaluate accuracy, citation precision/recall, hallucination rate (claim-level checking), and abstention on unanswerable questions.
4. Have a domain expert (or a careful rubric) grade 50 answers blind.

**Measure:** accuracy, citation precision, hallucination rate, abstention quality, and a cost comparison.

**Resources:** [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) · [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) · [LegalBench](https://huggingface.co/datasets/nguha/legalbench) · [hallucination leaderboard](https://github.com/vectara/hallucination-leaderboard) · [FAISS](https://github.com/facebookresearch/faiss)

**Stretch:** a learned router between fine-tuned and RAG modes; calibration curves; "know when to abstain" as an explicit accuracy-coverage trade-off.

---

#### E3 · Your language, your model — a multilingual SLM
`Difficulty 4/5 · Compute C`

**Hypothesis:** *Continued pretraining of a 0.5B model on 2B tokens of Hindi/Marathi improves that language's MILU score by ≥8 points while losing <2 points on English, and tokenizer fertility predicts which languages gain most.*

**Why it's interesting:** most SLM evaluation is English-centric. A project that builds a real benchmark for a low-resource language — and measures the transfer/forgetting trade-off — has immediate local value and a clean experimental design. Qwen3.5 and SmolLM3 are both multilingual, which gives you a strong starting point.

**What you'll learn:** tokenizer fertility, continual pretraining, catastrophic forgetting, multilingual evaluation, and data quality filtering for low-resource corpora.

**Build:**
1. Assemble 1–3B tokens of target-language text (Wikipedia, news, books) with dedup and quality filtering.
2. Continue pretraining a 0.5B model; then SFT on 5–10K instruction examples in the language.
3. Evaluate on [MILU](https://github.com/ai4bharat/MILU) and [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA), plus an English regression suite.
4. Measure tokens-per-word (fertility) for each language and correlate with gains.

**Measure:** target-language accuracy delta, English forgetting, fertility, and cost per point.

**Resources:** [MILU](https://github.com/ai4bharat/MILU) · [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA) · [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

**Stretch:** vocabulary extension; transliteration robustness; cross-lingual transfer from a related language.

---

#### E4 · The 2B agent — tool calling without a big model
`Difficulty 3/5 · Compute B/C`

**Hypothesis:** *A 2B model fine-tuned on tool-calling data lands within 5 points of an 8B model on BFCL's simple and multiple-function categories, and a confidence-based cascade to a large model beats either alone at half the cost.*

**Why it's interesting:** tool calling is the canonical "narrow, repetitive" agentic task where SLMs should win on economics. It is also a task where structured decoding and prompt format matter enormously, so there is real engineering to do. A cascade adds a second, more interesting claim: you do not need one model for everything, you need a *routing policy*.

**What you'll learn:** function-calling formats, constrained decoding, LoRA SFT, calibration, and cost-aware routing.

**Build:**
1. Fine-tune a 2B model on [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) or [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) with the target model's native tool template.
2. Add grammar-constrained decoding ([Outlines](https://github.com/dottxt-ai/outlines)) and a vLLM tool parser to guarantee valid calls.
3. Evaluate on a BFCL subset and a 50-task multi-turn suite of your own; measure call accuracy and argument accuracy separately.
4. Build a cascade: escalate to a large model when the small model's confidence is low (use logprob margins or self-consistency). Plot cost vs success.

**Measure:** call/argument accuracy, multi-turn success, escalation rate, and cost per completed task.

**Resources:** [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) · [τ-bench](https://arxiv.org/abs/2406.12045) · [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) · [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) · [Outlines](https://github.com/dottxt-ai/outlines) · [RouteLLM](https://github.com/lm-sys/RouteLLM)

**Stretch:** self-critique before calling; tool retrieval when the tool list is large; compare against a 9B model at equal cost.

---

### Track F — Study and measure

*The "science of SLMs" track. These projects produce knowledge rather than a model, and they are often the most original because they ask questions the benchmark-industrial complex ignores.*

#### F1 · Microscope — circuits in a 10M model
`Difficulty 3/5 · Compute B`

**Hypothesis:** *A 10M-parameter model trained on TinyStories contains induction heads and at least one monosemantic SAE feature whose ablation causally changes the next-token distribution, replicating across seeds.*

**Why it's interesting:** mechanistic interpretability usually studies GPT-2 because it is small enough to inspect. But GPT-2 is still 124M parameters, and its training data was never released. A 10M model you trained yourself, on a dataset you control, is a cleaner microscope — and TinyStories models are known to develop interpretable structure ([TinyStories](https://arxiv.org/abs/2305.07759)). You can go from "I trained this" to "I understand this" in one project.

**What you'll learn:** attention-head analysis, induction heads, sparse autoencoders, activation patching, causal interventions, and the epistemology of interpretability claims.

**Build:**
1. Train 5 seeds of a 10M TinyStories model; confirm they all learn the task.
2. Use [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to inspect attention patterns and identify induction heads; verify with ablation.
3. Train an SAE with [SAELens](https://github.com/decoderesearch/SAELens); label 20 features and test whether they are monosemantic by ablation.
4. Compare the circuit to GPT-2's published induction-head story; note what is the same and what is different.

**Measure:** causal effect sizes, feature interpretability rate (blind-labeled), and replication across seeds.

**Resources:** [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) · [SAELens](https://github.com/decoderesearch/SAELens) · [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) · [TinyStories paper](https://arxiv.org/abs/2305.07759)

**Stretch:** compare 1-layer vs 4-layer models; a toy grokking task where you can watch a circuit form; test whether SAE features transfer across seeds.

---

#### F2 · SLM Olympics — a reproducible benchmark for sub-4B models
`Difficulty 2/5 · Compute A`

**Hypothesis:** *A 5-task, 1,000-item benchmark for your domain ranks sub-4B models differently from MMLU, and a 200-item subset reproduces the full ranking with rank correlation >0.9.*

**Why it's interesting:** everyone complains that MMLU is saturated, contaminated, and irrelevant to their domain. Very few students build the alternative. A small, carefully constructed benchmark — with contamination checks, confidence intervals, and a public leaderboard — is a genuine community contribution, and it teaches evaluation design better than any exam.

**What you'll learn:** benchmark design, item analysis, IRT, contamination detection, statistical significance, and leaderboard engineering.

**Build:**
1. Define five tasks that matter for your domain (e.g., clinical extraction, legal citation, code repair, multilingual QA, tool selection), 200 items each.
2. Write items with distractors and a rubric; have 2–3 humans validate a sample; run a contamination check (n-gram overlap with training corpora).
3. Evaluate 10 models from 0.5B to 9B with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) or [lighteval](https://github.com/huggingface/lighteval).
4. Fit an IRT model, publish a Hugging Face Space leaderboard, and report which items discriminate best.

**Measure:** model ranking vs MMLU, item discrimination, rank correlation of the 200-item subset, and cost per evaluation.

**Resources:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) · [lighteval](https://github.com/huggingface/lighteval) · [HELM](https://crfm.stanford.edu/helm/) · [tinyBenchmarks](https://arxiv.org/abs/2402.14992) · [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)

**Stretch:** adaptive testing (IRT-based item selection); contamination probe; correlation with human expert judgments.

---

#### F3 · Red-team — how fragile are small models?
`Difficulty 3/5 · Compute B/C`

**Hypothesis:** *0.5–2B models are more susceptible to template-based jailbreaks than 8B+ models, and a small safety SFT cuts attack success rate by half at a cost of <3% on benign utility.*

**Why it's interesting:** safety evaluation is usually done on frontier models. The small-model regime is different: weaker instruction-following may make models *easier* to jailbreak, or *harder* because they fail to follow the jailbreak's complex framing. Either result is informative, and the safety-SFT trade-off curve is directly useful.

**What you'll learn:** adversarial prompting, attack success metrics, refusal behavior, safety fine-tuning, and the helpfulness–harmlessness trade-off.

**Build:**
1. Select 100 harmful behaviors from [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) or [HarmBench](https://github.com/centerforaisafety/HarmBench); generate benign counterparts.
2. Attack 0.5B/2B/8B models with template attacks, role-play, and a small GCG run; measure attack success rate with a judge plus manual validation.
3. Safety-tune the smallest model on a small refusal dataset; re-measure ASR, over-refusal on benign prompts, and utility (IFEval/MMLU subset).

**Measure:** ASR by attack type and model size, over-refusal rate, and utility delta.

**Resources:** [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) · [HarmBench](https://github.com/centerforaisafety/HarmBench) · [IFEval](https://arxiv.org/abs/2311.07911)

**Stretch:** refusal-direction analysis; compare safety data mixtures; test whether quantization changes refusal behavior (a real and under-studied effect).

---

### Track G — Wildcards

*Higher variance, higher upside. Pick one of these if you want a demo that nobody else in the class will have.*

#### G1 · Voice-first tiny assistant
`Difficulty 4/5 · Compute B/C`

**Hypothesis:** *A 1.5B audio-capable model with streaming input achieves sub-800 ms response latency on a laptop and beats a cascaded ASR→LLM→TTS pipeline on task success for short commands.*

**Why it's interesting:** audio is the interface where small models have an inherent advantage — the latency budget is brutal, and a cloud round-trip is often the bottleneck. 2026 finally has small any-to-any models: Gemma 4's E-models accept audio, and Liquid AI ships a 1.5B audio model ([LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)).

**What you'll learn:** speech encoders, streaming inference, latency budgets, and multimodal evaluation.

**Build:**
1. Build two pipelines: (a) a cascaded Whisper-style ASR + LLM + TTS, (b) an audio-native model.
2. Test 50 short commands and 20 conversational turns; measure word error rate, response latency, and task success.
3. Optimize the bottleneck (streaming, 4-bit, prompt caching) and re-measure.

**Measure:** WER, p50/p95 latency, task success, and subjective quality (small human panel).

**Resources:** [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) · [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) · [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)

**Stretch:** barge-in (interrupt the model mid-response); fully offline mode; speaker adaptation.

---

#### G2 · Diffusion vs autoregressive at 100M
`Difficulty 4/5 · Compute C`

**Hypothesis:** *A 100M masked-diffusion LM matches a 100M autoregressive LM on generation quality at 2× training compute, decodes 4× faster in parallel, and is dramatically better at infilling.*

**Why it's interesting:** [LLaDA](https://arxiv.org/abs/2502.09992) challenged the assumption that core LLM capabilities require autoregression, showing an 8B diffusion model competitive with LLaMA3 8B. At 100M you can test the claim's small-scale version — and the infilling advantage is a concrete, demo-able win that AR models cannot match without special training.

**What you'll learn:** masked diffusion, the ELBO, parallel decoding, any-order generation, and the trade-offs between sample quality and latency.

**Build:**
1. Implement a masked-diffusion objective (random mask ratio, denoising transformer) on the same data as an AR baseline.
2. Train both at 100M for 2B tokens; sample with different step counts.
3. Evaluate generation quality (judge + perplexity proxy), infilling accuracy on held-out spans, and tokens/s at batch 1.

**Measure:** quality vs steps, infilling accuracy, latency, and training compute.

**Resources:** [LLaDA paper](https://arxiv.org/abs/2502.09992)

**Stretch:** hybrid AR+diffusion; any-order instruction following; compare scaling slopes.

---

### The shortlist at a glance

| ID | Project | Track | Difficulty | Compute | Headline metric |
|---|---|---|---|---|---|
| A1 | Speedrun Zero | Build | 4 | D | CORE per dollar |
| A2 | TinyStories lab | Build | 2 | B | Emergence threshold |
| A3 | Textbook factory | Build | 3 | B/C | Domain accuracy at fixed compute |
| B1 | Professor & Student | Compress | 3 | C | Win rate per dollar |
| B2 | Quantization Frontier | Compress | 2 | B | Accuracy vs GB |
| B3 | Home-grown draft | Compress | 4 | C | Speedup at equal quality |
| C1 | Minimum Viable Reasoner | Reason | 4 | C | Accuracy + self-correction rate |
| C2 | Distilling thought | Reason | 2 | B/C | Accuracy vs #examples |
| C3 | Alignment bake-off | Reason | 3 | C | Win rate, length, refusal |
| D1 | Attention is not all you need | Architecture | 4 | C | Perplexity at 32K context |
| D2 | Small MoE | Architecture | 4 | C | Loss at iso-FLOPs |
| D3 | 1.58 bits from scratch | Architecture | 3 | C | Perplexity vs memory/energy |
| E1 | Pocket Agent | Deploy | 4 | A/B | p95 latency + task success |
| E2 | Specialist with receipts | Deploy | 3 | B/C | Accuracy + citation precision |
| E3 | Your language, your model | Deploy | 4 | C | Target-language delta |
| E4 | The 2B agent | Deploy | 3 | B/C | BFCL accuracy per dollar |
| F1 | Microscope | Study | 3 | B | Causal effect of a circuit |
| F2 | SLM Olympics | Study | 2 | A | Rank correlation vs MMLU |
| F3 | Red-team | Study | 3 | B/C | Attack success rate |
| G1 | Voice-first assistant | Wildcard | 4 | B/C | Latency + task success |
| G2 | Diffusion vs AR | Wildcard | 4 | C | Quality vs tokens/s |

### Three flagship combinations

If you want a capstone-sized project (or a portfolio piece), chain three entries into one narrative:

1. **The Full Stack** (C1 → B1 → B2 → E1). Train a tiny reasoner with GRPO, distil it into a 0.5B student, quantize it to 4-bit, and deploy it on a phone. The report becomes "from RL to pocket," with a cost and quality accounting at every step. This is the most impressive end-to-end story and the most work.
2. **The Data Thesis** (A3 → A2 → F2). Test whether synthetic textbooks beat web data, find the smallest model where the effect appears, then build the benchmark that measures it properly. This is the most "research paper" option.
3. **The Efficiency Thesis** (D1 → D3 → B3). Compare hybrid attention, ternary weights, and speculative decoding as three routes to cheaper inference; report a single Pareto frontier across all three. This is the most systems-flavored option and produces the best plots.

---

## 6. Execution plan

### 6.1 Eight weeks, with a buffer

| Week | Milestone | Evidence it's done |
|---|---|---|
| 0 | Pick three candidates, read one paper each, check hardware | A one-paragraph pitch per candidate |
| 1 | Reproduce a baseline number from a paper or repo | Your number within ~10% of the reference, or a documented reason why not |
| 2 | Build the evaluation harness and data pipeline | One command reproduces the baseline eval on a fixed checkpoint |
| 3–4 | Main experiment runs | All arms trained; logs and configs committed |
| 5 | Ablations and seeds | At least 3 seeds for the headline claim; ablation table |
| 6 | Analysis and plots | Every figure has error bars or a stated caveat |
| 7 | Report and demo | 6–10 page report; 3-minute demo that works offline |
| 8 | Buffer | Submit early; fix the thing that broke |

The most common scheduling failure is leaving the *evaluation* to week 6. Build it in week 2, even if it is ugly. You cannot recover from discovering in week 6 that your metric was wrong.

### 6.2 Repository layout

```
slm-project/
├── README.md              # question, result, how to reproduce (the most-read file)
├── configs/               # one YAML per experiment arm; no magic numbers in code
├── data/                  # download + preprocessing scripts only (no raw data in git)
├── src/
│   ├── train.py
│   ├── eval.py
│   └── serve.py           # optional demo server
├── notebooks/             # exploration only; the pipeline must run without them
├── results/
│   ├── figures/
│   └── tables/
├── report/
│   ├── report.md          # or LaTeX
│   └── refs.bib
└── Makefile               # `make reproduce` is the goal
```

Two rules that separate a good repo from a pile of notebooks: **configs are files, not constants**, and **`make reproduce` exists**. Pin your environment (`uv lock`, `requirements.txt` with hashes, or a container image) and record the git commit in every results file.

### 6.3 Report outline

1. **Question and claim.** One paragraph, one falsifiable hypothesis.
2. **Related work.** Five papers, each with one sentence on what you borrowed or contradicted.
3. **Setup.** Models, data, compute, hyperparameters, and a cost table.
4. **Method.** Enough detail that a competent reader could rerun it.
5. **Results.** Main table/figure first; ablations second; negative results included.
6. **Analysis.** Why the result happened; what surprised you; what you would test next.
7. **Limitations and threats to validity.** Contamination, single-seed, judge bias, scale.
8. **Reproducibility.** Seeds, versions, commit hash, commands, total spend.

### 6.4 What graders actually reward

| Criterion | Weight | What earns it |
|---|---|---|
| Novelty / insight | 20% | A question that is not answered by the paper you copied |
| Experimental rigor | 30% | Controls, seeds, matched compute, honest error bars |
| Execution | 20% | It runs, it's reproducible, the demo works |
| Analysis | 20% | You explain *why*, not just *what*; you discuss failure modes |
| Communication | 10% | Clear figures, a 3-minute story, no jargon soup |

### 6.5 Ten pitfalls

1. **Comparing across tokenizers.** Perplexity is not comparable between different vocabularies. Use bits-per-byte.
2. **One seed.** A single run cannot distinguish an effect from noise. Three seeds minimum for the headline claim.
3. **Judge bias.** LLM judges have position and verbosity biases. Swap positions, control length, and hand-validate a sample.
4. **Contamination.** Check n-gram overlap between your eval set and training data; report it even if it's zero.
5. **Benchmark overfitting.** Tuning prompts on the test set is the most common silent error. Keep a held-out set you touch once.
6. **"Loss went down" as a result.** Loss is a means; the claim needs a task-level metric.
7. **Ignoring cost.** A 2-point gain for 10× compute is not a win. Report dollars and GPU-hours.
8. **Gated models.** Check access before you build the pipeline around a model you cannot download.
9. **License surprises.** Llama 3.2 is not OSI-approved; if you publish derived weights, read the terms.
10. **Training a model you can't evaluate.** If the eval harness isn't ready, you are collecting expensive noise.

### 6.6 Hypothesis template

If you are stuck, fill this in:

> *At **[scale]**, **[intervention]** changes **[metric]** by **[direction/magnitude]** relative to **[control]**, and the effect **[persists / reverses]** at **[second scale]** because **[mechanism]**.*

Example: *At 0.5B, on-policy reverse-KL distillation improves IFEval by ≥5 points over SFT-on-traces at matched quality, and the gap widens at 0.2B because the student's own errors dominate the training distribution.*

A good hypothesis is falsifiable, has a control, and names a mechanism. If yours doesn't, keep reading papers until it does.

---

## 7. Resource library

All links were checked in November 2026. Prefer primary sources (papers, model cards, official docs) over blog summaries; the field moves fast enough that secondary sources are often a year stale.

### 7.1 Courses and curricula

| Resource | Why it's worth your time |
|---|---|
| [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) | The most complete "build it yourself" course: tokenizers, attention, training, scaling, evaluation. Lecture notes alone are worth the semester. |
| [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/unit1/1) | Practical alignment of small models: instruction tuning, evaluation, preference alignment, VLMs. Runs on modest hardware. |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) | The standard on-ramp for Transformers, datasets, and the Hub. Do this first if you are new to the ecosystem. |
| [Karpathy's Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | Builds backprop, makemore, and GPT from scratch on video. The best intuition-per-hour in the field. |

### 7.2 Papers that will shape your project

**Foundations and scale**
- [Small Language Models (SLMs) Can Still Pack a Punch: A Survey](https://arxiv.org/abs/2501.05465) — ~160 papers; the best single map of the field, updated in 2026.
- [TinyStories](https://arxiv.org/abs/2305.07759) — coherence below 10M parameters; also a template for LLM-judged evaluation.
- [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644) — data quality as a substitute for scale.
- [MobileLLM](https://arxiv.org/abs/2402.14905) — why deep-and-thin beats wide-and-shallow below 1B.

**Compression and efficiency**
- [MiniLLM](https://arxiv.org/abs/2306.08543) — reverse-KL, on-policy distillation; the reference for B1.
- [QLoRA](https://arxiv.org/abs/2305.14314) — 4-bit fine-tuning; the reason a T4 can touch 3B models.
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — ternary weights at parity with fp16.
- [Medusa](https://arxiv.org/abs/2401.10774) — multi-head speculative decoding.
- [EAGLE-2](https://arxiv.org/abs/2406.16858) — dynamic draft trees; the widely used draft-tree method.
- [Sheared LLaMA](https://arxiv.org/abs/2310.06694) and [ShortGPT](https://arxiv.org/abs/2403.03853) — structured pruning and layer redundancy, if you want to *shrink* rather than quantize.

**Reasoning and alignment**
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — RL with verifiable rewards; the recipe behind the reasoning boom.
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) — 1,000 examples plus budget forcing.
- [LIMO](https://arxiv.org/abs/2502.03387) — the "less is more" reasoning hypothesis.
- [DPO](https://arxiv.org/abs/2305.18290) and [KTO](https://arxiv.org/abs/2402.01306) — the two preference objectives you'll compare in C3.
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) — the position paper; read it critically.

**Evaluation**
- [IFEval](https://arxiv.org/abs/2311.07911) — instruction-following, programmatically verifiable.
- [τ-bench](https://arxiv.org/abs/2406.12045) — tool-agent-user interaction in realistic domains.
- [tinyBenchmarks](https://arxiv.org/abs/2402.14992) — how few items you actually need.

**Alternative paradigms**
- [LLaDA](https://arxiv.org/abs/2502.09992) — diffusion language models at scale.

### 7.3 Training and post-training tools

| Tool | Use it for |
|---|---|
| [Transformers](https://github.com/huggingface/transformers) + [PEFT](https://github.com/huggingface/peft) | The default stack: load, LoRA, quantized fine-tuning. |
| [TRL](https://github.com/huggingface/trl) | SFT, DPO, KTO, GRPO, GKD in one library with sane defaults. |
| [Unsloth](https://github.com/unslothai/unsloth) | 2× faster, lower-VRAM fine-tuning; the best free-Colab experience. |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) / [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Config-driven training when you want YAML instead of Python. |
| [nanotron](https://github.com/huggingface/nanotron) / [picotron](https://github.com/huggingface/picotron) | Minimal 3D/4D parallelism if you go multi-GPU. |
| [verl](https://github.com/volcengine/verl) | Production-grade RL post-training (GRPO/PPO) at larger scale. |
| [alignment-handbook](https://github.com/huggingface/alignment-handbook) | Reference recipes for SFT + DPO. |
| [distilabel](https://github.com/argilla-io/distilabel) | Synthetic data generation pipelines with provenance. |

### 7.4 Evaluation and leaderboards

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the de facto standard; broad task coverage.
- [lighteval](https://github.com/huggingface/lighteval) — faster, Hugging Face-native alternative.
- [HELM](https://crfm.stanford.edu/helm/) — holistic, multi-metric evaluation; good for thinking about what to measure.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool calling, with a public harness.
- [Chatbot Arena](https://lmarena.ai/) — human preference; useful context, not a substitute for your own eval.
- [Artificial Analysis](https://artificialanalysis.ai/) — speed/cost/quality comparisons across providers.

### 7.5 Deployment and inference

| Tool | Target |
|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | CPU/Metal/GPU via GGUF; the fastest path to "it runs on my laptop." |
| [Ollama](https://github.com/ollama/ollama) | Local model server with a friendly API; great for demos. |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput serving, tool parsers, speculative decoding. |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Compiled deployment to mobile/WebGPU. |
| [ExecuTorch](https://github.com/pytorch/executorch) | PyTorch-native on-device runtime for Android/iOS. |
| [WebLLM](https://github.com/mlc-ai/web-llm) / [Transformers.js](https://github.com/huggingface/transformers.js) | In-browser inference; a demo that needs no install. |
| [MLX](https://github.com/ml-explore/mlx-lm) | Apple-silicon inference and fine-tuning. |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | GPTQ/AWQ/FP8 quantization with vLLM integration. |
| [BitNet](https://github.com/microsoft/BitNet) | 1-bit/ternary CPU inference stack. |

### 7.6 Data

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — filtered educational web text; the standard pretraining baseline.
- [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) — FineWeb-Edu + Cosmopedia + Python-Edu, curated for small models.
- [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) — synthetic textbooks; the raw material for A3.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — the cheapest coherent-text dataset in existence.
- [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) — instruction data for small-model SFT.
- [OpenThoughts](https://github.com/open-thoughts/open-thoughts) — open reasoning traces for C1/C2.
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) / [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) — math reasoning data.
- [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) / [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) — tool-calling data.
- [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — preference pairs for C3.
- [MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — standard eval sets.

### 7.7 Compute

[Colab](https://colab.research.google.com/) · [Kaggle](https://www.kaggle.com/code) · [Lightning](https://lightning.ai/) · [Modal](https://modal.com/) · [RunPod](https://runpod.io/) · [HF Jobs](https://huggingface.co/docs/hub/jobs) · [Spaces ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu) for demos.

### 7.8 Communities

- [Hugging Face Discord](https://huggingface.co/join/discord) — the fastest place to get an answer about a model or library.
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — deployment war stories; good for sanity-checking "will this run?"
- Hugging Face model card discussions — often the only place a bug is documented.
- Your university's HPC support — the cheapest A100s you have.

---

## 8. Ethics, safety, and licenses

**Licenses.** Apache 2.0 and MIT are permissive; Llama 3.2 uses a community license with restrictions on large-scale commercial use and derivative naming. If you publish weights or a demo, check the base model's terms first. Gemma 4 moved to Apache 2.0, which simplifies things considerably ([announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)).

**Data.** Do not scrape personal data or use datasets with unclear consent for a project that will be public. For medical or legal corpora, check whether the license permits derivative models, not just research use. Report PII filtering if you do any.

**Dual use.** The red-team project (F3) is legitimate safety research, but attack artifacts can be misused. Keep harmful generations out of the public repo, disclose findings responsibly, and follow your institution's policy. The same applies to any jailbreak prompts you collect.

**Domain advice.** A medical or legal SLM must not be presented as advice. Include disclaimers, abstention behavior, and a clear statement of the evaluation's limits in the demo.

**Energy.** If you report efficiency, report energy too (joules per token or per query), not just latency. It is the metric most often omitted and most relevant to the "small models are greener" claim.

**Claims.** Do not call a 2B model "state of the art" based on a 200-item benchmark. Say what you measured, on what, with what uncertainty.

---

## 9. Appendix: the first 72 hours

**Day 1 — orientation (4–6 h)**
- [ ] Read the survey's abstract and skim its taxonomy ([2501.05465](https://arxiv.org/abs/2501.05465)).
- [ ] Pick three candidate projects from §5 and write a one-paragraph pitch for each.
- [ ] Check hardware: what GPU can you actually use, for how many hours, at what price?
- [ ] Run one small model locally (`ollama run` or a Transformers snippet) to confirm your environment works.

**Day 2 — baseline (6–8 h)**
- [ ] Clone the starter repo for your top candidate (nanochat, nanoGPT, smol-course, TRL example).
- [ ] Reproduce one number from the README or paper. Write down the number and the gap.
- [ ] Set up the evaluation harness *before* training anything.
- [ ] Create the repo skeleton from §6.2 and commit.

**Day 3 — proposal (4–6 h)**
- [ ] Turn your pitch into the hypothesis template in §6.6.
- [ ] Estimate compute with `6 × N × D` and a GPU-hour budget in dollars.
- [ ] Define the control arm and the metric that decides the claim.
- [ ] Write a 1-page proposal: question, hypothesis, method, compute, risks, timeline.
- [ ] Get feedback before spending money.

**Then:** run the smallest possible version of the experiment end-to-end (a 5M model, 100 steps) and confirm the pipeline produces a result file. Only after that should you scale up.

---

## Sources and verification

Model sizes, context lengths, and licenses were read from Hugging Face model cards and `config.json` files in November 2026 ([Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), [Granite 4.2 3B](https://huggingface.co/ibm-granite/granite-4.2-3b), [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Ministral 3 8B](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512), [Olmo 3 7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct), [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct), [BitNet 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)). Training-recipe details are from the [SmolLM3 engineering blog](https://huggingface.co/blog/smollm3) and the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook). Cost and time estimates are order-of-magnitude calculations from the FLOPs formula in §3.1, not measured benchmarks; your mileage will vary with MFU, sequence length, and provider pricing.

