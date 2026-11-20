# Making the Stance Model Open-Domain

**Question:** what would it take for the keyword models to handle *any* topic, instead of the 48-topic closed set?

**Short answer:** three different things get called "anything," and they cost wildly different amounts. Two are reachable on your laptop; the third is not, under a from-scratch constraint.

---

## 0. Three meanings of "anything"

| Level | What the user does | What the model needs | Reachable? |
|---|---|---|---|
| **A. Paraphrase** | Types "AI rules" instead of "AI regulation" | Robust topic matching, not world knowledge | Yes, a day |
| **B. Broad coverage** | Types "quantum computing funding" or "lab-grown meat" | Thousands of topics + retrieval, or a model pretrained on broad text | Yes, 1–2 weeks |
| **C. Truly arbitrary** | Types a topic invented yesterday | General world knowledge + stance reasoning | Not on 12 GB from scratch |

The current model fails at **B and C**. It has 19M parameters and saw 1.4M tokens — about 0.07 tokens per parameter, versus the ~20 tokens per parameter that scaling work suggests is a minimum for a dense model to retain knowledge. There is no room in it for facts about the world. The held-out test confirms this: unseen topics collapse onto the nearest trained topic.

The honest framing for the report: **the 48-topic model is a closed-set keyword generator, not a language model with opinions.** Everything below is about changing that.

---

## 1. What "anything" costs

Throughput estimates scale from the measured 92.7k tokens/s for a 25M model on your RTX 5070 Ti (≈1.4×10¹³ FLOP/s effective), using `6ND` and assuming throughput falls roughly linearly with parameters.

| Tier | Model | Pretraining tokens | Local time on 5070 Ti | What it handles |
|---|---|---|---|---|
| **T0** | none (retrieval) | 0 | 2–4 h engineering | Any input mapped to nearest known topic; coverage-bound |
| **T1** | 100M | 2B | ~24 h | 2,000 topics; related topics; weak on novel ones |
| **T2** | 100M | 10B | ~5 days (or ~2 h on 8×H100, ≈$50) | 5,000 topics; broad domains; some transfer |
| **T3** | 300M | 20B | ~2–3 weeks | 10,000 topics; decent open-domain behavior |
| **T4** | 1B | 100B+ | months locally; ~$1–3k rented | Approaching "anything," still not guaranteed |

**Disk:** 2B tokens ≈ 8 GB (uint16), 10B ≈ 40 GB, 20B ≈ 80 GB. You have 90 GB free — enough for T2, tight for T3, not enough for T4 without pruning the corpus.

**The wall:** from-scratch at 12 GB means T1/T2 are the realistic ceiling. T4 is a rented-cluster project, not a laptop project.

---

## 2. The data problem is bigger than the compute

A larger model trained only on 48 topics is still a lookup table. Open-domain behavior needs two corpora:

### 2.1 General pretraining (for language and world knowledge)

- **FineWeb-Edu** (filtered web text) — the standard small-model corpus. 2–10B tokens.
- **Wikipedia** — dense factual coverage; 3–6B tokens for English.
- **TinyStories** — optional curriculum warm-up; 0.5B tokens.

Licensing is permissive for FineWeb-Edu and Wikipedia (CC BY-SA). Filter for quality, deduplicate, and hold out a slice.

### 2.2 Stance corpus (for keyword behavior)

| Source | Size | Gives you |
|---|---|---|
| DebateSum | ~190k arguments | pro/con argument texts across debate motions |
| args.me | ~380k arguments | broader argument corpus |
| Wikipedia "Criticism"/"Controversy"/"Benefits" sections | thousands of articles | section-level stance labels |
| ChangeMyView | millions of threads (sample) | real pro/con framing |
| Parliamentary records (Hansard, Lok Sabha) | large | policy debate language, India coverage |

Build `(topic, stance, keywords)` triples with **RAKE/YAKE + log-odds** (no teacher model, no distillation). Target 5,000–20,000 topics. Human-review 50; spot-check the rest.

**This is the part that takes weeks, not the GPU.** At 5k topics × 2 sides × ~50 arguments, you have ~500k arguments — a few hundred million tokens of stance text, which is enough for T1/T2.

### 2.3 Tokenizer

The word-level vocab (≈1.5k words) will not survive a 10B-token corpus. Train a **BPE tokenizer at 32k merges** on the pretraining mix, then reuse it for the stance stage. This is still "from scratch" — a tokenizer is not a model.

---

## 3. Training recipe (still from scratch)

Two stages, both from your own random initialization:

1. **Pretrain** the 100M model on 2–10B general tokens (next-token prediction, context 512–1024). This is where world knowledge enters.
2. **Continue** on the stance-keyword corpus with the same objective, mixing 80–90% stance data and 10–20% general data to avoid forgetting.

If your course interprets "no fine-tuning" as *one* training run, concatenate the corpora and train once with a 90/10 mix. The model still starts from random weights.

Hyperparameters that work at this size (from the current runs, scaled):

```yaml
n_layer: 12
n_head: 8
n_embd: 768
context: 512
vocab: 32000
batch_tokens: 262144
lr: 3e-4
warmup: 2%
schedule: cosine
weight_decay: 0.1
dropout: 0.1
precision: bf16
```

Expect ~24 h for T1 on your GPU. Run it overnight, evaluate in the morning.

---

## 4. Evaluation for open-domain

The current 8-topic holdout is too small to claim anything. A credible open-domain evaluation needs:

- **200 held-out topics**, stratified across domains, never seen in training.
- **Coverage:** does the model produce topic-relevant keywords at all?
- **Stance adherence:** ≥90% same-polarity keywords (lexicon check).
- **Relevance:** human 1–5 on 100 keywords; LLM judge as a secondary signal.
- **Unknown-topic behavior:** does it admit uncertainty, or hallucinate a different topic? (The current model does the latter — that is the failure mode to fix.)
- **Cost:** tokens/s and GPU-hours per 1,000 topics.

A useful baseline: TF-IDF keywords from the Wikipedia article. If the model does not beat TF-IDF on relevance, the pretraining did not help.

---

## 5. The pragmatic move for tomorrow

Do **not** attempt T1 before the deadline. Instead, add **T0: retrieval fallback** to the existing app:

1. Build a TF-IDF/trigram index over the 48 topic strings and their keyword sets (no model needed).
2. If the user's topic is not an exact match, retrieve the nearest 3 topics and return their keywords merged, labeled **"nearest known topic: X"**.
3. Show a quiet warning when similarity is low; never silently pretend.

This makes the demo *feel* open-domain, is honest about its mechanism, and takes 2–4 hours. The report then has a clean story: closed-set generation now, retrieval fallback as the bridge, and a quantified plan (the table above) for true open-domain.

---

## 6. Recommendation

| If you have... | Do this |
|---|---|
| 2 hours | T0 retrieval fallback; update the demo and report |
| 1 week | T1: 2,000 topics, 100M model, 2B tokens (~24 h GPU) |
| 1 month | T2: 5,000 topics, 10B tokens (rent 8×H100 for a day, or 5 days locally) |
| A course rule against pretrained weights | Stay at T0/T1; T4 is out of scope |

The gap between "48 topics" and "anything" is not a cleverer model. It is **two orders of magnitude more data and one order of magnitude more parameters**, and the data is the expensive part.
