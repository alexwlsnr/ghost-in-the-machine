# Spec512 Design — Multi-Turn Conversational Model

## Architecture

| Parameter | Value | Notes |
|---|---|---|
| Params | ~27.6M | d=512, L=8, h=8, ff=2048 |
| Vocab | 258 | bytes 0-255 + PAD=256 + EOS=257 |
| Context | **1024** | enables 10-12 turns of history |
| d_model | 512 | |
| n_heads | 8 | head_dim=64 |
| n_layers | 8 | |
| d_ff | 2048 | |
| pos_embed | 1024×512 | learned, adds ~524K params vs ctx=256 |

ctx=1024 rationale: 10-12 turns of conversation history fit comfortably
(avg 50 bytes/turn × 2 sides = ~100 bytes/turn → 10 turns = ~1000 tokens).
KV cache at 67MB is within browser Wasm budget. Training sequences average
~300 tokens (5-turn dialogue) so compute cost is modest despite 1024 ceiling.

## Multi-Turn Format

### Training sequences

Full dialogues are stored as pipe-separated turns, alternating Q and R:

```
Q1|R1|Q2|R2|Q3|R3
```

At training time, `make_sequence_multiturn()` converts this to a flat byte
sequence with SEP tokens between every turn:

```
[q1_bytes][SEP][r1_bytes][SEP][q2_bytes][SEP][r2_bytes][SEP][q3_bytes][SEP][r3_bytes][EOS]
```

SEP=1 (ASCII SOH), EOS=257. The model sees the full dialogue and learns to
generate any response token conditioned on all preceding tokens.

Single-turn pairs (`Q|R`) are also valid — they're just 2-turn sequences with
no history. The training set mixes single-turn and multi-turn at roughly 30/70.

### Inference

At inference, the UI maintains `conversationHistory: Array<{q, r}>` per model.
On each new query, `buildContextTokens(history, query, maxLen)` builds the token
sequence by prepending turns from newest to oldest until the budget is full:

```typescript
function buildContextTokens(history: Turn[], query: string, maxLen: number): number[] {
  const queryTokens = [...encode(query.toUpperCase()), SEP];
  let tokens = queryTokens;
  for (const turn of [...history].reverse()) {
    const chunk = [...encode(turn.q), SEP, ...encode(turn.r), SEP];
    if (tokens.length + chunk.length >= maxLen) break;
    tokens = [...chunk, ...tokens];
  }
  return tokens;
}
```

Oldest turns are silently dropped when the window fills — the model gracefully
degrades rather than erroring. The UI shows a subtle "context full" indicator
when history is being truncated.

## Training Data

Target: ~87K pairs equivalent (Chinchilla-adjusted for 27.6M params).
At multi-turn with avg 3 turns/sequence, that's ~30K multi-turn dialogues.

### Sources

| Source | Pairs/dialogues | Format | Notes |
|---|---|---|---|
| SODA full dialogues | ~50K dialogues | Multi-turn (3-8 turns) | Re-ingest as full dialogues, not pairs |
| Scenario generator (multi-turn) | ~15K dialogues | Multi-turn (2-4 turns) | Extend gen_scenarios.py |
| Scenario generator (single-turn) | ~5K pairs | Single-turn | Existing scenarios.txt |
| Distilled meta/jokes | ~400 pairs | Single-turn | Existing data/meta.txt, data/jokes.txt |

### SODA multi-turn extraction

SODA conversations average 6-8 turns. Instead of slicing into consecutive pairs
(current approach), extract the full dialogue up to ctx=1024 budget:

```python
def extract_dialogue(turns: list[str], max_ctx: int = 1024) -> list[str] | None:
    """Return normalised turns if the full dialogue fits in ctx."""
    normed = [normalise_names(normalise_pair(t, '')[0]) for t in turns]
    # Build flat sequence to check length
    total = sum(len(t) for t in normed) + len(normed) + 1  # SEPs + EOS
    if total > max_ctx:
        # Trim from the front until it fits
        while len(normed) > 2 and total > max_ctx:
            total -= len(normed[0]) + 1
            normed = normed[1:]
    if len(normed) < 2:
        return None
    return normed
```

Name normalisation (`normalise_names`) is applied to every turn so character
names become HUMAN throughout.

### Multi-turn scenario generation

Extend `gen_scenarios.py` with a `--turns N` flag (default 1, max 4).
The teacher is prompted to generate a full N-turn dialogue:

```
System: Generate a conversational exchange in the format:
  Q1: ...
  A1: ...
  Q2: ...
  A2: ...
All uppercase. Each line under 80 chars. The AI is GHOST. Human is HUMAN.
No character names.

User: Generate a {N}-turn exchange where {scenario_description}.
```

Output is parsed into `[(q1,r1), (q2,r2), ...]` and stored as `q1|r1|q2|r2`.

## Training Changes (train_transformer.py)

### New sequence builder

```python
SEP_TOKEN = 1
EOS_TOKEN = 257

def make_sequence_multiturn(
    turns: list[tuple[str, str]],
    max_len: int,
    truncate: bool = False,
) -> list[int]:
    """Build flat token sequence from [(q1,r1), (q2,r2), ...] dialogue."""
    tokens = []
    for q, r in turns:
        tokens += encode(q) + [SEP_TOKEN] + encode(r) + [SEP_TOKEN]
    tokens[-1] = EOS_TOKEN  # replace trailing SEP with EOS
    if len(tokens) > max_len:
        if truncate:
            return tokens[:max_len]
        return []
    return tokens
```

### Data file format

Two file types are accepted:
- **Single-turn** `Q|R`: existing format, handled by `make_sequence()`
- **Multi-turn** `Q1|R1|Q2|R2|...`: detected by even number of pipe-separated fields ≥ 4

`--multi-turn` flag enables multi-turn parsing. Both formats can coexist in
the same file.

## Inference Changes (tier2_transformer.ts)

### buildContextTokens

New export alongside `generate()`:
```typescript
export function buildContextTokens(
  history: Array<{q: string, r: string}>,
  query: string,
  maxLen: number,
): number[]
```

### generate() signature change

`generate()` gains an optional `history` parameter. When provided, it calls
`buildContextTokens` to build the full token sequence instead of just encoding
the query.

```typescript
async function* generate(
  model: LoadedModel,
  query: string,
  maxTokens: number,
  temperature: number,
  rand: () => number,
  cache?: KVCache,
  history?: Array<{q: string, r: string}>,  // NEW
)
```

## UI Changes (index.html)

### Per-model conversation history

```javascript
let conversationHistories = {};  // id → Array<{q, r}>

// After each successful generation:
conversationHistories[activeId] ??= [];
conversationHistories[activeId].push({ q: query, r: response });
// Cap at 20 turns to prevent unbounded growth
if (conversationHistories[activeId].length > 20) {
  conversationHistories[activeId].shift();
}
```

### History display

The terminal already shows Q/R pairs. No UI change needed for the user — they
see the conversation naturally. Add a `[CLEAR]` button to the input line that
resets `conversationHistories[activeId] = []` and clears the terminal.

### Context indicator

When history is being truncated (i.e. oldest turn was dropped), show a subtle
status bar message: `CONTEXT: 8/10 turns · oldest dropped`.

### Per-model behaviour

Multi-turn only enabled for models with ctx ≥ 256 (Spec512, Specter).
Wisp and Shade continue as single-turn — their ctx=128 doesn't leave room
for meaningful history.

## Quantization

Same 3-level strategy as Shade:
- fp32: 105MB reference (27.6M × 4 bytes)
- bf16: 54MB (near-lossless, top-16 bits of each float)
- 8-bit: 28MB (per-tensor scale, mixed precision)
- ~~4-bit~~: skip for Spec512 — quality degradation not worth it at this size

4-bit Spec512 would be 14MB but the quality hit at 25M+ params is worse than
at smaller models. 8-bit at 28MB is the right deployment target.

## Sequencing

1. ✅ soda_256.txt ingest running (ctx=256; re-run at ctx=1024 once arch confirmed)
2. → Re-ingest SODA as full dialogues (new `ingest_soda_dialogues.py`)
3. → Extend `gen_scenarios.py` with `--turns` flag
4. → Generate 15K multi-turn scenario dialogues
5. → Build Spec512 training set (~30K multi-turn dialogues + 5K single-turn)
6. → Add `make_sequence_multiturn()` to `train_transformer.py`
7. → Train Spec512 (est. 4-8 hours on current hardware)
8. → Add `buildContextTokens()` + history param to `generate()` in TS
9. → Wire conversation history into UI for Spec512
10. → Serialize Spec512 at fp32, bf16, 8-bit; deploy

## Open questions

- **Loss masking**: Should loss be computed only on response tokens? Would help
  the model learn "I generate responses, not queries" more cleanly. Adds
  complexity to the training loop. Defer — try without masking first.
- **SODA re-ingest**: soda_256.txt is ingesting at ctx=256 now. Will need a
  separate `ingest_soda_dialogues.py` that preserves full turn structure rather
  than flattening to Q|R pairs. Stop current ingest if we're committed to 1024.
- **Context indicator threshold**: How many dropped turns before we show the
  indicator? Suggest: show only when 2+ turns dropped.
