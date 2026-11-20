# Two-Stance Keyword Models, Trained From Scratch

**Revised brief:** two small language models, one that emits *positive* keywords for a topic and one that emits *negative* keywords. Keywords only — no sentences. No distillation, no fine-tuning of pretrained weights: the models are trained by you, from random initialization.

---

## 0. Verdict

**This is more feasible than the sentence version — and the 2–3 hour training target is now realistic.** But it only works if you accept one constraint: the topic set must be *closed* (a few hundred known topics), or you accept that unseen topics produce generic keywords.

| Question | Answer |
|---|---|
| Train two models from scratch in 2–3 h? | **Yes.** 15–30M params each on 100–300M tokens, comfortably. |
| Output keywords instead of sentences? | **Yes.** Shorter sequences, smaller vocabulary, faster convergence, easier evaluation. |
| Work for *any* topic the user types? | **Not really** at this scale — unless you add retrieval or a fallback. See §3. |
| Guarantee "always positive" / "always negative"? | **Yes**, with constrained decoding (§6). |
| Realistic end-to-end timeline | **2–3 days**, of which ~2–3 h is GPU training. |

The bottleneck has moved. With sentences, the bottleneck was generation quality and stance drift. With keywords, it is **world knowledge**: a 20M-parameter model trained on 200M tokens does not know that "AI regulation" pulls toward *innovation* on one side and *bureaucracy* on the other. It only knows what your corpus taught it.

---

## 1. Why keywords change the math

| | Sentence version | Keyword version |
|---|---|---|
| Output length | 50–150 tokens | 5–10 tokens |
| Effective training signal per example | Diffuse (style, stance, facts) | Dense (which words, in what order) |
| Stance drift | High; needs best-of-N, prefix forcing | Low; word choice is the whole output |
| Evaluation | Subjective, judge-dependent | Objective-ish: precision@k, polarity rate |
| Knowledge requirement | High | **Still high** — keywords must be *relevant*, not just positive |
| Training cost | 1.5B LoRA, hours | 15–30M from scratch, minutes to ~1 h |

The last row is the trap. "Positive keywords for AI regulation" is not a sentiment task; it is a *stance-relevant concept* task. Positive words like "good, beneficial, promising" are generic. The valuable outputs are topic-specific concepts: *innovation, safety, accountability* vs *bureaucracy, compliance costs, stagnation*. Those come from data, not from the model's imagination.

So the project is really two problems:

1. **Learn the format** — given a topic, emit a short list of keywords in the right stance. Easy; a tiny model nails this.
2. **Know the associations** — which concepts belong to which side of which topic. Hard; bounded by your corpus and topic coverage.

Design around (2), and (1) takes care of itself.

---

## 2. What 2–3 hours buys from scratch

From random initialization, on a 12 GB laptop GPU, with a 20–40M parameter decoder-only transformer:

| Model | Params | Tokens in 1 h* | Tokens in 3 h* | Notes |
|---|---|---|---|---|
| 10M (6 layers, d=256) | ~10M | 300–600M | 1–1.8B | TinyStories scale; enough for keyword format |
| 20M (8 layers, d=384) | ~20M | 150–300M | 450M–900M | **Recommended** |
| 40M (8 layers, d=512) | ~33M | 80–150M | 250M–450M | Better language, slower |
| 80M (12 layers, d=640) | ~65M | 50–90M | 150–270M | Possible, but data-hungry |

\*Assuming 1–2×10¹³ effective FLOP/s and `6ND`; measure with a benchmark before committing. Two models means two runs — halve the token budget per model, or accept 4–5 h total.

**Memory is not the issue.** A 20M model in bf16 is 40 MB of weights; Adam states ~160 MB; activations at batch 64 × seq 256 are ~1–2 GB. You can train both models simultaneously on one GPU if you want.

**The real budget is data.** For the task stage you need ~5–20M tokens of keyword sequences, which is 50k–200k examples. That is the part that takes a day, not an hour.

**On "no fine-tuning":** if that means "no third-party pretrained checkpoints," a two-stage from-scratch pipeline is fine — pretrain your own 20M model on a general corpus, then continue training it on the keyword task. Both stages start from your random init. If your instructor means literally *one* training run, mix the general and task corpora into a single stage (90/10) and accept weaker task behavior. Confirm which interpretation applies before you invest in data.

---

## 3. Three designs

### A. Closed-set topic→keyword models (recommended)
The user picks or searches from **150–250 topics** the models were trained on. Two 20M models learn the mapping. Quality is high because the associations are in the data.

- **Pros:** works reliably; evaluable; demo looks great; honest.
- **Cons:** the "type any topic" dream is bounded. Mitigate with a search/autocomplete UI and a nearest-topic fallback (embedding or trigram match).

### B. Open-topic shallow generator
Train on a large, automatically constructed corpus of (topic, stance, keywords) where keywords are extracted statistically (see §4). The model learns a general "stance keyword style" plus copying from the topic string.

- **Pros:** any topic works; no closed set.
- **Cons:** keywords become generic ("beneficial, safe, efficient" vs "risky, costly, harmful") and often just rephrase the topic. At 20M parameters, relevance to novel topics will be weak.

### C. Retrieval + tiny generator (best UX, most engineering)
Embed the user's topic, retrieve the 5 nearest known topics, feed their keyword sets to the tiny model as context, and let it merge/adapt them.

- **Pros:** open-topic behavior with closed-set quality; a legitimate DL story (retrieval-augmented generation).
- **Cons:** adds an embedding model (which may violate "from scratch" if pretrained — train a small one yourself or use TF-IDF), a vector index, and more failure modes.

**Recommendation: A, with B's data pipeline behind it.** Build the 200-topic corpus; train closed-set; add a "nearest known topic" fallback in the UI. If time remains, upgrade to C.

---

## 4. Data without distillation

You need `(topic, stance, keyword list)` triples. No teacher model required — this is classical NLP, not distillation.

### 4.1 Pipeline

1. **Topics.** 200 controversial/policy/tech/health topics from Wikipedia lists, debate motions, and news headlines. Record 15–20% as held-out topics.
2. **Topic corpus.** For each topic, collect a Wikipedia article + 10–30 news paragraphs. This is your relevance signal.
3. **Candidate keywords.** Extract noun phrases with **RAKE** or **YAKE** (pure statistics, no neural model). Keep 30–60 candidates per topic. Stopword-filter and lemmatize.
4. **Stance scoring.** Build pro/con corpora from debate datasets (DebateSum motions, args.me, ChangeMyView threads). Compute a **log-odds ratio with an informative Dirichlet prior** ("Fightin' Words", Monroe et al. 2008) for each word, globally and per topic. Words that discriminate pro from con get signed scores.
5. **Select.** For each topic, keep the top 8–12 candidates by *relevance × stance score*, requiring the pro and con sets to be disjoint. Human-review 20 topics and fix the obvious errors.
6. **Package.** One record per topic per stance, plus augmentations: keyword-order shuffles, subsets (3, 5, 8 keywords), topic paraphrases. That turns 400 records into 20k–50k training sequences.

### 4.2 Sources

- [DebateSum](https://huggingface.co/datasets/Hellisotherpeople/DebateSum) — debate motions with pro/con argument summaries
- [args.me](https://huggingface.co/datasets/webis/args_me) — argument search corpus
- ChangeMyView (Reddit) — human pro/con threads
- Wikipedia + news — topic context and candidate keywords
- Sentiment lexicons (VADER, NRC, AFINN) — a weak prior for word polarity; use only as a tie-breaker, not the main signal
- Keyphrase datasets (KP20k, Inspec) — if you want human-annotated keyword style

### 4.3 Volume and splits

| Split | Topics | Purpose |
|---|---|---|
| Train | 160 | Learn associations |
| Dev | 20 | Early stopping, prompt format |
| Test (held out) | 20 | Generalization claim |

Each topic needs ≥10 keyword variants per stance, or the model memorizes one list. Augmentation is not optional.

---

## 5. Model and training recipe

### 5.1 Architecture (20M)

```yaml
vocab: 16384            # BPE trained on your corpus
context: 256
layers: 8
d_model: 384
heads: 6                # 64-dim heads
ffn: 1536               # SwiGLU
rope: true
norm: pre-ln (RMSNorm)
dropout: 0.1
tie_embeddings: true
```

### 5.2 Sequence format

```
<bos> TOPIC: {topic} | STANCE: {positive|negative} | KEYWORDS: {kw1}, {kw2}, ..., {kw8} <eos>
```

At inference, stop at `<eos>` or after 10 keywords, `top_p=0.9`, `temperature=0.8`. Train each model on its own stance data so the stance is baked into the weights, not the prompt.

### 5.3 Hyperparameters

```yaml
batch_tokens: 65536        # e.g. 256 sequences × 256 tokens
lr: 3e-4
warmup: 2%
schedule: cosine
weight_decay: 0.1
epochs: 20-40              # small dataset; early-stop on dev
grad_clip: 1.0
precision: bf16
```

### 5.4 Time

| Stage | Tokens | Est. |
|---|---|---|
| Tokenizer (BPE 16k) | — | 10 min |
| Model A (positive), 20M | 150M | 40–60 min |
| Model B (negative), 20M | 150M | 40–60 min |
| Eval + sampling | — | 20 min |
| **Total** | **300M** | **~2–3 h** |

That is the whole GPU budget. Everything else is data engineering and UI.

---

## 6. Guaranteeing polarity at decode

Training on one stance makes the model *biased* toward that polarity; it does not make it absolute. Three mechanisms, cheapest first:

1. **Logit bias.** At each step, add `+2.0` to tokens in the stance lexicon and `-2.0` to tokens in the opposite lexicon. Soft, preserves fluency.
2. **Hard mask on the first keyword.** Restrict the first generated keyword to the stance lexicon; let the rest be free. This kills the "starts neutral, drifts" failure.
3. **Post-filter + regenerate.** Score the output with a lexicon; if polarity agreement is below threshold, resample. Report the resample rate.

Do not hard-mask every step. It produces keyword salad and destroys relevance. Mask the first token, bias the rest.

---

## 7. Evaluation

| Metric | How | Target |
|---|---|---|
| Stance polarity | Lexicon score of every keyword | ≥95% same polarity |
| Topic relevance | Human 1–5 on 100 keywords; embedding similarity as a proxy | ≥3.5 |
| Precision@8 | Overlap with human-written keyword sets for 15 topics | Report; no fixed target |
| Diversity | Unique keywords per topic across 5 samples | >60% |
| Held-out topic quality | Same metrics on 20 unseen topics | Within 15% of seen topics |
| Baselines | TF-IDF top words; lexicon-only keywords; random stance words | The model must beat TF-IDF on relevance |

The baseline row matters. A model that emits "good, great, beneficial" for every topic is a lexicon with extra steps. Showing that it beats TF-IDF on relevance is the actual result.

---

## 8. The UI

Keywords are a gift for UI design: they fit in chips, clouds, and comparisons.

- **Compose:** topic search with autocomplete over the 200 known topics, plus free text. If the free text is far from every known topic, show a quiet warning: "low confidence — nearest known topic: X".
- **Result:** two columns of chips, sized by model probability, warm color for positive, cool for negative. Hover a chip to see the model's confidence; click to pin it.
- **Controls:** "more keywords", "regenerate", temperature.
- **Footer:** what the model is, training cost, held-out accuracy, disclaimer.

A keyword interface is also *fast*: two 20M models generate 8 keywords in well under a second on CPU, let alone GPU.

---

## 9. Timeline and cost

| Phase | Time |
|---|---|
| Topic list + corpus collection | 4–6 h |
| Keyword extraction + stance scoring + review | 6–8 h |
| Tokenizer + training scripts | 3–4 h |
| Two training runs | 2–3 h GPU |
| Evaluation | 3–4 h |
| UI | 6–8 h |
| **Total** | **~2–3 days** |

Cost: electricity only, **under $5** if you use public corpora. No API teacher, no cloud GPU.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Model memorizes one keyword list per topic | Augment aggressively; measure diversity |
| Unseen topics produce generic words | Closed-set UI + nearest-topic fallback; report the gap |
| Keyword extraction is noisy | Human-review 20 topics; tune RAKE/YAKE thresholds |
| Dataset too small (few tokens) | Add augmentations; or pretrain on a general corpus first |
| "That's not a language model" critique | It is a decoder-only LM trained from scratch; frame it as conditional keyword generation |
| Polarity leakage | First-token mask + logit bias + post-filter |
| Evaluation is subjective | Fix a rubric, have 2 annotators on 100 items, report agreement |

---

## 11. What to tell the grader

The claim is not "we built two chatbots." It is:

> *"We train two 20M-parameter decoder-only language models from random initialization on a 200-topic stance-keyword corpus constructed with statistical keyword extraction and log-odds stance scoring. The positive model emits stance-consistent keywords for 95% of held-out topics, beats TF-IDF baselines on human relevance by X points, and runs at Y keywords/second on CPU — for Z GPU-hours."*

That is a complete DL project: data construction, from-scratch training, controlled evaluation, and a deployable artifact.

---

## Appendix: one training record

```json
{
  "topic": "Governments should regulate AI-generated political advertising",
  "stance": "positive",
  "keywords": ["transparency", "election integrity", "disclosure", "accountability",
               "misinformation", "oversight", "public trust"],
  "source": "wikipedia+debatesum",
  "split": "train"
}
```
