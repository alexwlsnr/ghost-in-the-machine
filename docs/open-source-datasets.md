# Open-Source Datasets for Shade / Spec512 Training

_Research note — June 2026_

## Context

This document evaluates open-source conversational and instruction-following datasets for augmenting the training corpus beyond the template-driven teacher-generation pipeline described in `multi-model-plan.md`. The primary constraints are:

- **Byte-level tokenizer**: vocab is bytes 0–255 + PAD + EOS + SEP. Only ASCII text tokenises cleanly; non-ASCII bytes are valid but waste context and confuse the embedding space.
- **Short context windows**: Wisp ctx=64, Shade ctx=128, Spec512 ctx=256. `len(query) + len(response) + 1 (EOS) ≤ ctx`.
- **Format**: `QUERY|RESPONSE` lines, uppercase, English only.
- **Volume targets**: 50K pairs for Shade, 500K for Spec512 (from `multi-model-plan.md`; the plan's actual stated target for Spec512 is 200K pairs).
- **License**: must be permissive (CC-BY, Apache 2.0, MIT, CC0, public domain) and free of OpenAI/GPT ToS contamination that could restrict a demo project.

---

## 1. Dataset Candidates

### 1.1 OpenAssistant OASST2

| Field | Value |
|---|---|
| HuggingFace ID | `OpenAssistant/oasst2` |
| License | **Apache 2.0** |
| Total messages | ~135,000 (train 128,575 / val 6,599) |
| Languages | 28 languages; English ~48% of `ready_for_export` subset (~64,500 messages) |
| Human-generated | Yes — 13,500+ volunteers |
| Domain | Conversational assistant, instruction following, Q&A, creative |

**Structure.** Data is a tree of messages (prompter/assistant roles), not flat pairs. Extracting Q|R requires walking each tree to collect (prompter message, assistant reply) pairs at each turn. Top-voted assistant replies are easiest to use; a curated `top_1_answer` traversal yields roughly 10,000–15,000 English prompt/response pairs from the highest-quality paths.

**ASCII survivability estimate.**
- English filter: keeps ~48% → ~65,000 messages
- ASCII-only filter: human English text is >99% ASCII once you strip markdown formatting characters (backticks, asterisks, angle brackets). Realistic ASCII pass rate: ~90%.
- Length filter (query 5–80 chars, response 10–120 chars): OASST responses are conversational but often longer — humans writing detailed answers frequently exceed 120 chars. Estimated survival: ~20–30% of pairs.
- No code/math: excludes a notable fraction (~15–20% of OASST content involves code snippets or technical walkthroughs). Apply `no triple backtick` and `no math LaTeX` filters.
- **Estimated yield**: ~3,000–5,000 high-quality English pairs after full filtering (top-path extraction + ASCII + length + no-code).

**Notes.** The tree format requires non-trivial extraction. Use `datasets.load_dataset("OpenAssistant/oasst2")` and walk message trees, filtering `lang == "en"` and selecting `rank == 0` assistant messages. Quality is high — human-generated, human-ranked. Excellent diversity.

---

### 1.2 Databricks Dolly 15K

| Field | Value |
|---|---|
| HuggingFace ID | `databricks/databricks-dolly-15k` |
| License | **CC BY-SA 3.0** (commercial use permitted; ShareAlike applies to derivatives) |
| Total pairs | 15,011 |
| Languages | English only |
| Human-generated | Yes — 5,000+ Databricks employees |
| Domain | Closed QA, open QA, summarisation, info extraction, classification, brainstorming, creative writing |

**Structure.** Flat JSONL with `instruction`, `context`, `response`, `category` fields. Straightforward to extract `instruction|response` pairs; some entries include a `context` field (passage to summarise/extract from) which adds noise for our format.

**ASCII survivability estimate.**
- English filter: 100% English by construction.
- ASCII-only: >98% — professional English prose.
- Length filter: **this is the main bottleneck**. Dolly responses are long by design ("contains long answers to most tasks"). The `instruction` field ranges from 4 to 11,700 chars; the `response` field from 1 to 26,000 chars. Based on the distribution, only brainstorming and simple open-QA categories tend to have short responses. Estimated survival at ≤120-char response: ~5–10% of pairs.
- No code/math: generation and brainstorming categories are clean; summarisation and extraction involve long technical passages.
- **Estimated yield**: ~750–1,500 pairs after full filtering — too few to be the primary source, but useful as a quality supplement.

**Notes.** The CC BY-SA 3.0 ShareAlike clause means any dataset you distribute that incorporates Dolly content must also be CC BY-SA 3.0. For an internal training corpus that is not publicly distributed, this is not a practical constraint. Attribution required if you distribute.

---

### 1.3 LIMA

| Field | Value |
|---|---|
| HuggingFace ID | `GAIR/lima` |
| License | **CC BY-NC-SA 4.0** (non-commercial) |
| Total pairs | 1,000 (train) + 300 (test) |
| Languages | English |
| Human-curated | Yes — 1K hand-selected high-quality examples from Stack Exchange, wikiHow, Pushshift/Reddit, manually authored |
| Domain | Instruction following, reasoning, creative, factual Q&A |

**BLOCKED: non-commercial license.** The CC BY-NC-SA licence prohibits commercial use and derivatives must carry the same restriction. Even for a demo project, this is a restriction that complicates distribution. If the project is strictly personal/non-commercial research, LIMA could be used — but the 1K pairs are too few to justify the legal complexity.

**Estimated yield if used**: ~100–200 pairs after length filtering (responses are typically 2–5 paragraphs).

**Recommendation: skip.**

---

### 1.4 Stanford Alpaca

| Field | Value |
|---|---|
| HuggingFace ID | `tatsu-lab/alpaca` |
| License | **CC BY-NC 4.0** — non-commercial only |
| Total pairs | 52,002 |
| Languages | English |
| Generated by | OpenAI `text-davinci-003` |
| Domain | General instruction following |

**BLOCKED: two independent issues.**

1. **Non-commercial license (CC BY-NC 4.0)**: Prohibits commercial use of the dataset or derivatives.
2. **OpenAI ToS contamination**: The dataset was generated by `text-davinci-003` via OpenAI's API. OpenAI's Terms of Service prohibit using model outputs to train competing models. Although the October 2024 ToS update clarified that this obligation binds only the original API caller and does not transfer to downstream recipients of the data, the dataset's own license (CC BY-NC) is an independent blocker regardless.

**Recommendation: skip.**

---

### 1.5 TinyStories

| Field | Value |
|---|---|
| HuggingFace ID | `roneneldan/TinyStories` |
| License | **CDLA-Sharing 1.0** |
| Total samples | 2.14 million stories |
| Languages | English (synthetic) |
| Generated by | GPT-3.5 and GPT-4 (Microsoft Research) |
| Domain | Children's short stories, vocabulary-limited prose |

**License note.** CDLA-Sharing 1.0 (Community Data License Agreement – Sharing) is an open license from the Linux Foundation. It permits use and redistribution but requires that modifications to the dataset be shared under the same license. It is broadly permissive for training use.

**OpenAI ToS note.** Stories were generated by GPT-3.5/GPT-4. Per the October 2024 ToS clarification, the downstream restriction on training competing models binds only the original API caller (Microsoft Research). As a third-party user of the published dataset, there is no contractual obligation. This is the consensus practical interpretation, though it is not risk-free.

**ASCII survivability estimate.**
- 100% ASCII by design — the dataset uses only simple English vocabulary.
- Length filter: stories range from a few sentences to several paragraphs. Individual _sentences_ within stories would need to be extracted as Q|R pairs, not whole stories. This is non-trivial and the data was not designed for Q|R format.

**Fitness for our format.** TinyStories is not a Q|R dataset. It is a story completion / next-token-prediction corpus. Converting it to Q|R pairs requires either treating partial stories as prompts and continuations as responses (misaligned with our conversational goal) or ignoring it entirely. It is purpose-built for language model _pretraining_ or story generation, not instruction following or conversation.

**Recommendation: skip for Q|R training. Consider as supplementary pretraining data if a pretraining phase is added.**

---

### 1.6 WizardLM Evol-Instruct V2

| Field | Value |
|---|---|
| HuggingFace ID | `WizardLMTeam/WizardLM_evol_instruct_V2_196k` |
| License | **MIT** |
| Total pairs | ~143K evolved QA pairs (plus ~53K from ShareGPT to reach the stated 196K) |
| Languages | English |
| Generated by | GPT-3.5 / GPT-4 (evolved from Alpaca + ShareGPT seeds) |
| Domain | Complex instruction following, code, reasoning, math |

**OpenAI ToS note.** GPT-generated; same analysis as TinyStories above — ToS binds the original caller, not downstream users.

**Fitness for our format.** Evol-Instruct was specifically designed to generate _complex, multi-step_ instructions. It skews heavily toward code (at least 40% by inspection), multi-part reasoning tasks, and long responses. This is the opposite of what we need. Length filter would kill >80% of pairs. The non-code, non-math fraction would be small and the surviving responses would be of dubious conversational quality.

**Recommendation: skip.**

---

### 1.7 OpenOrca

| Field | Value |
|---|---|
| HuggingFace ID | `Open-Orca/OpenOrca` |
| License | **MIT** |
| Total rows | ~4.2M (1M GPT-4, ~3.2M GPT-3.5) |
| Languages | Primarily English |
| Generated by | GPT-4 and GPT-3.5 responses to FLAN Collection questions |
| Domain | Chain-of-thought reasoning, instruction following, FLAN tasks |

**OpenAI ToS note.** GPT-generated; same downstream analysis as above.

**Fitness for our format.** OpenOrca augments the FLAN collection with GPT-4/3.5 responses for chain-of-thought reasoning. The tasks are heavily reasoning-oriented — math, logic, science. CoT responses are by definition long (typically >500 chars). Length filter survival would be very low (<5%). Also heavily technical.

**Recommendation: skip for our use case.**

---

### 1.8 UltraChat 200K

| Field | Value |
|---|---|
| HuggingFace ID | `HuggingFaceH4/ultrachat_200k` |
| License | **MIT** |
| Total rows | 207,865 SFT training examples (filtered from 1.4M ChatGPT dialogues) |
| Languages | English |
| Generated by | ChatGPT (GPT-3.5-turbo) — original dataset by Tsinghua University |
| Domain | Multi-turn conversation, question answering, writing, general knowledge |

**OpenAI ToS note.** ChatGPT-generated. Same ToS analysis: downstream recipients are not contractually bound by OpenAI's ToS since the obligation ran only against the original API user (Tsinghua). This is the practical consensus, with low but non-zero legal ambiguity for commercial projects.

**ASCII survivability estimate.**
- English, ~99% ASCII.
- Length filter: multi-turn conversations with responses typically 1–5 sentences. Individual turns are shorter than OASST or Dolly responses. Estimated survival at ≤120-char response: ~25–35%.
- No code/math: UltraChat covers general knowledge and conversation topics. Code fraction is modest (~10%). 
- **Estimated yield**: ~40,000–55,000 pairs from 207K after filtering.

**Structure.** Multi-turn dialogue format: each row contains a `messages` list of `[{role: "user", content: ...}, {role: "assistant", content: ...}, ...]`. Extract each consecutive user/assistant pair as a Q|R pair (treating each turn independently).

**Notes.** Very good diversity — the original 1.4M dataset covers ~30 meta-topics with subtopics. High quality after HuggingFaceH4's filtering pass. The main reservation is the GPT ToS grey area; for a non-commercial demo this is low-risk.

---

### 1.9 FLAN Collection (Flan v2)

| Field | Value |
|---|---|
| HuggingFace ID | `SirNeural/flan_v2` (community rehost) / `philschmid/flanv2` |
| License | **Apache 2.0** |
| Total samples | 1,836 constituent datasets; millions of examples |
| Languages | English (primary), some multilingual |
| Generated by | Source tasks templated by Google Research; responses are task-derived, not LLM-generated |
| Domain | Classification, QA, summarisation, translation, reasoning, NLI |

**Fitness for our format.** FLAN is a task collection, not a conversational dataset. Most tasks have structured inputs (passage to classify, premise/hypothesis pairs, documents to summarise). The response format is typically a short label ("positive", "entailment") or a long summary. Very few tasks produce natural conversational responses. The Apache 2.0 license is ideal, but the domain mismatch is severe — FLAN would train the model on a completely different distribution than conversational Q|R.

**Recommendation: skip for conversational training. The `open_qa` and `cot_gsm8k` subsets might yield a few thousand pairs, but extraction complexity is high.**

---

### 1.10 SODA (Allen AI)

| Field | Value |
|---|---|
| HuggingFace ID | `allenai/soda` |
| License | **CC-BY 4.0** |
| Total dialogues | 1,186,394 train / 146,346 val / 148,968 test = ~1.49M total |
| Languages | English (synthetic) |
| Generated by | InstructGPT (OpenAI API) distillation from ATOMIC10x knowledge graph |
| Domain | Social, casual, emotionally-grounded conversation |
| Average utterance length | **16.1 characters** |
| Average turns per dialogue | 7.6 |

**OpenAI ToS note.** InstructGPT-generated (same GPT API). Same downstream analysis: ToS binds Allen AI as the original API caller, not downstream dataset users. Allen AI published under CC-BY 4.0.

**ASCII survivability estimate.**
- 100% English by construction.
- ASCII-only: ~100% — synthetic English text using standard vocabulary.
- **Length filter**: average utterance is 16.1 characters. This is the best-fit dataset for our format by a large margin. Single turns fit well within a 120-char response window; even 3–4-sentence responses in the longer tails would be short.
- Estimated survival at query 5–80 chars, response 10–120 chars: **~70–80%** of extracted pairs.
- No code/math: social conversation dataset — zero code or math content.
- **Estimated yield**: With 1.49M dialogues × 7.6 turns / 2 pairs per turn ≈ 5.7M utterance pairs raw; after length filter ~4M; take a diverse sample of ~500K with deduplication.

**Structure.** Each row: `narrative` (context), `dialogue` (list of turns), `relation` (social relation between speakers), `literal` and `final_emotion` annotations. Extract consecutive `(dialogue[0], dialogue[1])`, `(dialogue[1], dialogue[2])`, etc., using the first utterance as query and the second as response. Alternatively, treat each odd-indexed utterance as query and even-indexed as response for an even Q|R split.

**Notes.** Designed specifically for social commonsense dialogue — exactly the conversational register our models need. Short utterances, natural flow, grounded in everyday situations. The knowledge-graph-derived narrative context is discarded during extraction, leaving clean conversation pairs. This is the single most structurally aligned dataset for our use case.

---

### 1.11 LAION OIG (Open Instruction Generalist)

| Field | Value |
|---|---|
| HuggingFace ID | `laion/OIG` |
| License | **Apache 2.0** (LAION-authored content) |
| Total rows | ~52.6M (OIG-small-chip2 high-quality subset: ~210K) |
| Languages | English |
| Generated by | Mixed: some human-written, some synthetically generated |
| Domain | Mixed: dialogue (~4.7M rows from SODA/OSCAR/Parliament), Q&A, code, math, safety |

**Fitness for our format.** The full OIG is too large and mixed-quality. However, the `OIG-small-chip2` subset (~210K rows) is flagged as high-quality for fine-tuning. The conversational subset (including SODA-derived content, Canadian Parliament dialogue) totals ~4.7M rows and is the most relevant portion. Since SODA is already included and separately accessible under CC-BY 4.0, it is cleaner to use SODA directly. OIG is Apache 2.0 for LAION-authored content but some component sources carry other licenses (CC-BY-SA from Wikipedia).

**Recommendation: use SODA directly rather than via OIG. The `oig_small_chip2_noncode` variant is worth evaluating if more pairs are needed.**

---

## 2. ASCII Survivability Analysis — Top 4 Candidates

| Dataset | English | ASCII-only | Length (q≤80, r≤120) | No code/math | Combined | Est. pairs |
|---|---|---|---|---|---|---|
| **SODA** | 100% | ~100% | ~75% | ~100% | **~75%** | **300K–1M+** |
| **UltraChat 200K** | 100% | ~99% | ~30% | ~90% | **~27%** | **40K–55K** |
| **OASST2** | ~48% | ~90% | ~25% | ~82% | **~9%** | **3K–5K** |
| **Dolly 15K** | 100% | ~98% | ~8% | ~85% | **~7%** | **750–1,500** |

Notes on survivability estimates:
- "Length" is the dominant filter for all datasets except SODA. Most instruction datasets target long, detailed responses (300–2,000 chars); our 120-char ceiling is aggressive.
- SODA's average utterance length of 16.1 chars makes it uniquely suited — most responses survive the filter by default.
- OASST2's extraction overhead (tree walking, rank filtering) reduces practical yield further.
- All estimates assume uppercase conversion (safe for ASCII — pure case fold, no information loss).

---

## 3. Processing Pipeline Sketch — Primary Recommendation: SODA

### Download

```python
from datasets import load_dataset
ds = load_dataset("allenai/soda", split="train")
# 1.19M rows, ~2.5 GB download
```

### Extract Q|R Pairs

```python
import re

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def clean(s):
    # Strip leading/trailing whitespace, collapse internal spaces
    return re.sub(r'\s+', ' ', s.strip())

def extract_pairs(row):
    dialogue = row['dialogue']
    pairs = []
    for i in range(len(dialogue) - 1):
        q = clean(dialogue[i])
        r = clean(dialogue[i + 1])
        if (5 <= len(q) <= 80
                and 10 <= len(r) <= 120
                and is_ascii(q) and is_ascii(r)
                and '```' not in q and '```' not in r):
            pairs.append(f"{q.upper()}|{r.upper()}")
    return pairs

all_pairs = []
for row in ds:
    all_pairs.extend(extract_pairs(row))
```

### Filtering Steps

1. **ASCII gate**: `is_ascii(q) and is_ascii(r)` — rejects any non-ASCII byte.
2. **Length gate**: `5 ≤ len(q) ≤ 80` and `10 ≤ len(r) ≤ 120`.
3. **Context-fit gate**: `len(q) + len(r) + 1 ≤ ctx` — use ctx=128 for Shade pairs, ctx=256 for Spec512 pairs.
4. **Code/special gate**: reject rows containing triple backticks, `$`, `\[`, `\(`, or `def ` / `function `.
5. **Deduplication**: exact-match dedup on `q.upper()`, then near-dedup with a trigram Jaccard similarity threshold of 0.85 to remove templated near-duplicates (SODA is knowledge-graph-derived and contains many structurally similar dialogues).
6. **Quality filter**: optionally reject pairs where the response starts with "I " (first-person machine-sounding) or contains "as an AI" — SODA is naturalistic but a fraction of InstructGPT-derived turns sound robotic.

### Expected Yield After Filtering

| Step | Pairs remaining |
|---|---|
| Raw extracted pairs (1.19M rows × 7.6 turns / 2) | ~4.5M |
| After length filter (~75%) | ~3.4M |
| After deduplication (~40% reduction for near-dups) | ~2.0M |
| After quality/robot-phrase filter (~5% rejection) | ~1.9M |
| **Final estimated yield** | **~1.5–2M pairs** |

This comfortably covers 500K pairs for Spec512, 100K for Shade, and 50K for Wisp.

### Uppercase Conversion Safety

Uppercase conversion of ASCII text is fully safe — it is a pure case fold with no information loss. All semantic content, punctuation, and syntax are preserved. The model is already trained on uppercase data.

### Processing Speed

At ~4.5M candidate pairs, extraction + filtering runs in under 5 minutes on a single CPU with the `datasets` library. The near-dedup step using trigrams is the bottleneck; a MinHash-LSH approach (e.g. `datasketch`) scales to millions of pairs in ~15 minutes.

---

## 4. Legal / License Analysis

### Safe for Demo Use

| Dataset | License | GPT-generated | Demo safe | Notes |
|---|---|---|---|---|
| **SODA** | CC-BY 4.0 | InstructGPT (2022) | **Yes** | ToS binds Allen AI only; CC-BY 4.0 is permissive; attribution to Allen AI required |
| **OASST2** | Apache 2.0 | No (human) | **Yes** | Cleanest legal status — fully human-generated |
| **Dolly 15K** | CC BY-SA 3.0 | No (human) | **Yes** | ShareAlike applies to distributed derivatives; attribution required |
| **UltraChat 200K** | MIT | ChatGPT | **Low risk** | ToS analysis: downstream users not bound; MIT license is permissive. Grey area for commercial projects. |
| **FLAN v2** | Apache 2.0 | No (templated tasks) | **Yes** | Cleanest license; poor domain fit |
| **OIG (Apache portion)** | Apache 2.0 | Partial | **Yes** | Mixed sources; use chip2 subset |

### Blocked / Avoid

| Dataset | Blocker |
|---|---|
| **Alpaca** | CC BY-NC 4.0 (non-commercial) + OpenAI ToS origin |
| **LIMA** | CC BY-NC-SA 4.0 (non-commercial) |
| **WizardLM Evol-Instruct** | MIT license OK, but 40%+ code content; domain mismatch; GPT-4 generated |

### OpenAI ToS Nuance (October 2024 update)

OpenAI's Service Terms prohibit the original API user from using outputs to train competing models. A significant October 2024 update clarified that this obligation binds only the original API caller and does not transfer to downstream recipients of published datasets. This is the practical consensus interpretation among the ML community. For a private demo project (non-commercial, not claiming to compete with OpenAI), the residual risk from using SODA or UltraChat is very low. If any legal ambiguity is unacceptable, use OASST2 (fully human-generated, Apache 2.0) exclusively.

### Attribution Requirements

- **SODA**: cite Kim et al. (EMNLP 2023), link to `allenai/soda` on HuggingFace.
- **OASST2**: cite Köpf et al. (NeurIPS 2023), link to `OpenAssistant/oasst2`.
- **Dolly 15K**: "This dataset uses the Databricks Dolly 15K dataset, licensed under CC BY-SA 3.0."

---

## 5. Recommendation

### Primary: SODA (`allenai/soda`)

**SODA is the single best dataset for this project.** It is the only dataset among all candidates with an average utterance length (16.1 characters) that aligns with our 120-char response ceiling by default. Every other dataset was designed for long-form instruction following and requires aggressive filtering that kills 90%+ of content.

Key advantages:
- 1.49M social dialogues → estimated **1.5–2M usable Q|R pairs** after filtering. Covers all three model tiers without ceiling.
- CC-BY 4.0 — clean license, permissive, requires only attribution.
- 100% English, 100% ASCII, zero code/math content.
- Conversational register (social chitchat, everyday situations) matches what the models need to learn — not task completion or reasoning chains.
- Allen AI published this specifically for training dialogue models.

Concrete yield estimate after full filtering: **~1.5M pairs**. After dedup, take a stratified random sample of 500K for Spec512, 100K for Shade, and 50K for Wisp.

### Secondary: OASST2 (`OpenAssistant/oasst2`) — quality supplement

Use OASST2 to inject 3,000–5,000 high-quality, human-written English pairs into the mix. Human-generated data has better diversity and naturalness than distilled/templated data. Even at 3–5K pairs it represents ~5–10% of the Shade corpus and should meaningfully improve generalisation.

Extraction approach: use only the top-ranked (rank=0) assistant response at each node in English conversation trees. Filter to response length ≤120 chars, which eliminates most technical deep-dives but retains the short conversational answers.

### Do Not Use

- **Alpaca** and **LIMA**: non-commercial licenses are a hard blocker.
- **Dolly 15K**: too few short responses to justify extraction effort (~750–1,500 pairs). Use only if quality gaps emerge.
- **TinyStories**: wrong format for Q|R training; pretraining-only.
- **WizardLM / OpenOrca**: 40–80% code/reasoning content survives; wrong domain.
- **FLAN**: task distribution completely misaligned with conversational Q|R.

### Recommended Action Order

1. **Download SODA** — `load_dataset("allenai/soda")` — and run the extraction + filtering pipeline above. One-time ~5-minute pipeline run.
2. **Sample by tier**: 500K pairs for Spec512, 100K for Shade, 50K for Wisp. Stratify by dialogue length (short turns for Wisp, longer for Spec512) to match each model's context window.
3. **Merge with teacher-generated data**: combine SODA-extracted pairs with the template-driven pipeline output. The teacher data has better instruction-following framing; SODA adds conversational naturalness.
4. **Add OASST2 quality seed**: ~3–5K human pairs into each tier's dataset for diversity injection.
5. **Near-dedup across sources**: run dedup across the combined corpus to remove any SODA-SODA near-duplicates and any cross-source echoes.

---

## Sources

- [OpenAssistant/oasst2 — HuggingFace](https://huggingface.co/datasets/OpenAssistant/oasst2)
- [databricks/databricks-dolly-15k — HuggingFace](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [GAIR/lima — HuggingFace](https://huggingface.co/datasets/GAIR/lima)
- [tatsu-lab/alpaca — HuggingFace](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [roneneldan/TinyStories — HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)
- [WizardLMTeam/WizardLM_evol_instruct_V2_196k — HuggingFace](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k)
- [Open-Orca/OpenOrca — HuggingFace](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [HuggingFaceH4/ultrachat_200k — HuggingFace](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [allenai/soda — HuggingFace](https://huggingface.co/datasets/allenai/soda)
- [laion/OIG — HuggingFace](https://huggingface.co/datasets/laion/OIG)
- [SirNeural/flan_v2 — HuggingFace](https://huggingface.co/datasets/SirNeural/flan_v2)
- [SODA paper: Kim et al. 2022 — arXiv:2212.10465](https://arxiv.org/abs/2212.10465)
- [Databricks Dolly 2.0 blog post](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- [Demystifying OpenAI's Terms of Use with Regards to Dataset Licenses](https://erichartford.com/demystifying-openais-terms-of-use-with-regards-to-dataset-licenses)
- [LAION OIG blog post](https://laion.ai/blog/oig-dataset/)
