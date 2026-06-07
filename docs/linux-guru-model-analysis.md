# Linux Guru — Domain-Specialized Model: Viability Analysis

> **Status:** Analysis / pre-decision  
> **Branch:** `analysis/linux-guru-model`  
> **Relates to:** `docs/multi-model-plan.md` (the 3-tier Wisp/Shade/Specter stack)

---

## Executive summary

A Linux-oracle model is **technically feasible and architecturally well-matched** to this
project's constraints. The byte-level encoding is a natural fit for command-line content,
the measured answer lengths are shorter than expected (mean 71 bytes — well inside a
Shade-class context), and the data sourcing pipeline is substantially simpler than the
conversational pipeline. The primary risks are knowledge compression (can the model
actually recall which flag does what?) and the "is this better than `tldr ls`?" value
question. A low-cost MVP is well-defined and can validate both risks quickly.

**Recommendation: build the MVP. Gate the full model on MVP results.**

---

## 1. Context window viability

### Measuring real answer lengths

Twenty representative Linux questions were sampled and answered in the shortest form that
conveys complete, actionable information. Every answer was measured in raw bytes:

| Question | Answer (bytes) | Sample text |
|---|---|---|
| what does find -exec {} + do | 84 | "Like -exec {} ; but batches args into one invocation. Faster than one call per file." |
| how to change permissions recursively | 62 | "chmod -R 755 /path/to/dir -- applies to dir and all contents." |
| difference between > and >> | 73 | "> overwrites the file. >> appends to it. Both create the file if missing." |
| what does grep -v do | 60 | "Invert match -- print lines that do NOT contain the pattern." |
| ls sorted by time | 54 | "ls -lt lists newest first. ls -ltr lists oldest first." |
| what does chmod 755 mean | 92 | "Owner: rwx (7). Group: r-x (5). Others: r-x (5). Read+execute for all, write only for owner." |
| how to find a process by name | 57 | "pgrep nginx or ps aux \| grep nginx to list matching PIDs." |
| what does df -h show | 75 | "Disk filesystem usage in human-readable units (K/M/G). -h = human-readable." |
| how to kill a process by PID | 75 | "kill PID sends SIGTERM. kill -9 PID sends SIGKILL (uncatchable, immediate)." |
| what does tar xzf mean | 79 | "x=extract, z=gunzip, f=file. tar xzf archive.tar.gz extracts a gzipped tarball." |
| ssh on a non-standard port | 47 | "ssh -p 2222 user@host -- specify port with -p." |
| what does wc -l do | 69 | "Count lines in a file or stdin. wc -l file.txt prints the line count." |
| how to copy a directory recursively | 63 | "cp -r src/ dst/ -- copy directory and all contents recursively." |
| what does ps aux show | 79 | "All processes: a=all users, u=user-oriented format, x=include no-tty processes." |
| how to create a symlink | 77 | "ln -s target linkname -- creates a symbolic link. -s makes it soft/symbolic." |
| what does awk '{print $2}' do | 66 | "Prints the 2nd whitespace-separated field from each line of input." |
| grep recursively | 67 | "grep -r pattern /path -- search all files under /path for pattern." |
| what is /dev/null | 88 | "A black hole -- writes are discarded, reads return EOF. cmd > /dev/null silences output." |
| how to check open ports | 74 | "ss -tlnp or netstat -tlnp -- show TCP listening ports with process names." |
| what does set -e do in bash | 80 | "Exit the script immediately if any command fails (returns non-zero exit status)." |

**Summary statistics:** mean 71 bytes, min 47, max 92.  
Every single one of the 20 answers fits in ≤100 bytes with no information loss.

This is the most important finding in the entire analysis: **Linux oracle answers are
intrinsically short.** The information density of a command explanation is high, but the
surface area is narrow. "What does `-v` do in grep?" has exactly one correct answer and
it fits in one sentence.

### Context budget per tier

Typical Linux queries are also short. "what does df -h show" is 20 bytes; "how to
change permissions recursively" is 36 bytes. The measured mean is 27 bytes for the
20-question sample above.

| Model | Context | Query budget | Answer budget | Verdict |
|---|---|---|---|---|
| Wisp | 64 | ~27 bytes | ~36 bytes | Too tight — many queries overflow with no room for answer |
| Shade | 128 | ~27 bytes | ~100 bytes | **Good** — covers 100% of measured answers with headroom |
| Specter | 256 | ~27 bytes | ~228 bytes | Comfortable, even for multi-flag or multi-step answers |
| Wraith (proposed) | 256 | ~27 bytes | ~228 bytes | Optimal for this domain |

**Wisp is eliminated.** A 64-byte context works for 2-3 word conversational pings
("HELLO", "HOW ARE YOU") but not for "what does chmod -R 755 do". Even the query alone
can eat 40+ bytes.

**Shade (128 ctx) is viable for simple, single-flag questions.** The 95-byte answer
budget covers all 20 measured answers. However, it has no room for examples, and
multi-flag questions ("what are all the flags grep supports for context lines?") will
overflow. Shade-scale makes sense for an MVP but not the final model.

**Specter/Wraith (256 ctx) is the right operating point.** 228 bytes of answer budget
handles even verbose explanations with a command example included.

### The case for 512-byte context

A 512-byte context is worth considering for two specific use cases:

1. **Multi-step procedures** — "how do I set up SSH key auth?" requires generating keys,
   copying to the server, setting permissions, and testing. That's 4 steps, each ~60
   bytes = ~240 bytes of answer. Fine in a 512 ctx, overflows a 256 ctx once you add
   the query and an example.

2. **Man-page style summaries** — rather than a single Q&A, a richer format like
   "SYNOPSIS: ls [OPTION]... [FILE]... -- List directory contents." followed by the
   3 most-used flags is more useful than any single-flag answer. This runs 200-300 bytes.

**However:** 512-byte context is not free. The Wasm inference loop is O(T²) in attention
(no KV cache yet), so doubling context from 256 to 512 quadruples per-token cost.
The existing plan notes that Specter (256 ctx) already needs KV cache before it's usable
in-browser. A 512-byte model without KV cache would be ~4× slower again.

**Decision:** Start at 256 ctx. Validate with MVP. Consider 512 ctx only after KV cache
is implemented (Phase 3 in the existing plan).

### Can answers always be compressed to ≤200 bytes?

Yes — with appropriate teacher prompting. The compression test on `find -exec`:

- Man page source: 1,145 bytes
- Oracle-compressed: 122 bytes — "find -exec {} ; runs cmd once per file. find -exec {} + batches files into one call -- faster. {} is replaced by filename."

Zero information loss for practical usage. The key prompt constraint for the teacher:
"Explain this in one or two sentences, stating only the command name, what it does, and
the most important caveat. Do not explain options you are not asked about."

---

## 2. Data sourcing

### Man pages

**Volume on a typical Arch Linux system (`man -k .`):**

| Section | Pages | Description | Core? |
|---|---|---|---|
| (1) | 3,440 | User commands | Yes |
| (8) | 1,332 | System administration | Yes |
| (2) | 546 | System calls | Marginal |
| (5) | 570 | File formats | Marginal |
| (7) | 475 | Miscellaneous / conventions | Marginal |
| (3) | 12,847 | Library functions (C API) | No |
| (3ssl), (3perl), (3x), etc. | 7,000+ | Noisy, domain-specific | No |

For a user-facing terminal oracle, sections 1 and 8 are the core. That is **~4,772 pages**
totaling roughly **50 MB of raw text** (estimated at 10 KB/page for section 1, 12 KB for
section 8, based on direct measurement of 20 common commands: ls=7.5 KB, grep=30 KB,
find=81 KB, awk=112 KB, tar=40 KB, ssh=48 KB).

Note the wide size variance: `cat` is 1.9 KB, `awk` is 112 KB. The median is much closer
to ls than to awk.

**Redistribution:** Man pages for GNU coreutils, util-linux, and most open-source tools
are licensed GPL or MIT — freely redistributable. A few proprietary tools (e.g., some
network vendors) have non-free man pages, but these are absent from standard Linux
systems. The corpus as described is legally usable for training.

**Q&A pairs from man pages:** Each man page yields:
- 1 synopsis Q&A ("what does X do" → one-sentence description)
- 1-3 per-flag Q&A pairs for the most common flags
- 1 "when would I use X" Q&A from the DESCRIPTION section
- 1 "common gotchas" Q&A from the NOTES/BUGS section if present

Conservative estimate: **3 Q&A pairs/page × 4,772 pages = ~14,000 pairs**  
Generous estimate: **8 Q&A pairs/page × 4,772 pages = ~38,000 pairs**

With teacher-generated diversity (5 rephrasings per source Q&A), this becomes
**70K–190K pairs** — sufficient for a Shade-class model and the lower end of Specter.

### tldr-pages

- Repository: https://github.com/tldr-pages/tldr
- Pages: **7,243** (verified via GitHub API tree listing)
- Total text: **~4 MB** (estimated from GitHub-reported repo size of 57 MB including
  all translations — English-only pages are ~4 MB)
- License: **CC0-1.0** — public domain dedication, no attribution required, freely usable
  for training
- Format: structured Markdown with command name, description, and 5-8 example invocations

tldr pages are **purpose-built for the oracle use case.** Each page is already a
compressed man page: one-line description, then usage examples with human-readable
annotations. The format maps directly to training pairs:

```
# ls
> List directory contents. ...
- List all files including hidden files: `ls --all`
```

→ Q: "list all files including hidden" → A: `ls --all`  
→ Q: "what does ls --all do" → A: "List all files including hidden files."

At 7,243 pages × 5-8 examples each: **36K–58K direct pairs before any teacher expansion.**
With 3× teacher rephrasing: **~150K pairs**. This corpus alone is enough to train a
well-specialized Shade-scale model.

**tldr-pages is the best single data source for the MVP** due to:
1. Curated quality (community-reviewed, concise, practical)
2. Structure that maps cleanly to Q&A pairs without teacher reformatting
3. CC0 license — zero legal friction
4. Already covers the 500 most-used commands across all major tools

### Arch wiki

The Arch wiki (~10,000+ articles, estimated 150 MB of wikitext) is publicly available and
has been used in prior training datasets (e.g., RedPajama, Dolma). License is
CC-BY-SA-4.0 — attribution required but training use is generally accepted under fair use
in most jurisdictions.

**Usefulness for this model: moderate.** Arch wiki articles are comprehensive but
*not* oracle-style. An article on `systemd` units runs 20,000+ words. It cannot be
directly used as training pairs — it requires heavy teacher-mediated extraction.

The Arch wiki is better suited as a secondary corpus for a larger (Specter or 512-ctx)
model to handle "how do I set up X" questions that require multi-step configuration
context. For the MVP (Shade-scale), skip it entirely.

### `--help` output

`--help` is compact and structured. Measured sizes: `ls --help` = 12 KB, `grep --help`
= 48 KB, `find --help` = 6.5 KB, `tar --help` = 16.6 KB.

Systematic collection: iterate `dpkg -l` or `pacman -Ql | grep /usr/bin/`, run
`command --help 2>&1`, parse flag lines (lines starting with `-`). This is an entirely
automated, local operation. A simple script can collect `--help` output from every
binary in `/usr/bin` in minutes.

**However:** `--help` output is redundant with man pages for well-documented tools and
is often machine-generated (less readable) or terse (more readable) depending on the
project. It is best used as a cross-reference or deduplication check, not as a primary
training source.

### Kernel and systemd docs, POSIX spec

- **Kernel docs** (`/usr/share/doc/linux` or https://kernel.org/doc/): extensive but
  deeply technical. A user-facing oracle rarely needs kernel internals. Omit from MVP.
- **systemd docs** (`man systemd.unit`, `man systemctl`): these are man pages — already
  covered by the man page pipeline. The systemd man pages are detailed and high-quality.
- **POSIX spec**: the full POSIX spec is 5,000+ pages and overly formal for oracle use.
  The important POSIX content is already surfaced in GNU coreutils man pages.

### Total training pairs achievable

| Source | Pages/Articles | Direct pairs | After 3× rephrase |
|---|---|---|---|
| tldr-pages (sec 1+8) | 7,243 | 36K–58K | 108K–174K |
| Man pages sec 1+8 | 4,772 | 14K–38K | 42K–114K |
| --help flags (automated) | ~2,000 binaries | 10K–20K | 30K–60K |
| **Total** | | **~60K–116K direct** | **~180K–350K** |

**This is enough data to train a well-specialized Shade-scale model (10.9M params).**
The 50K–200K range is the target from the existing plan for Shade, and this domain has
the advantage of highly structured, low-noise data — each pair is verifiably correct.
For a Specter-scale model (57M params), the expanded 180K–350K range is adequate
if quality is consistent.

### Data quality advantage over conversational distillation

The conversational pipeline generates Q&A from free-form templates, which means:
- Some pairs are trivial ("HOW ARE YOU" → "GREAT THANKS")  
- Quality is hard to validate automatically
- Teacher refusals and preambles must be filtered

The Linux corpus has an inverse property: every piece of source data is ground truth.
If a man page says `-v` means verbose, that is correct. The teacher's job is only to
*reformat*, not to *generate* knowledge. This means:
- Filter rate will be much lower (fewer invalid pairs)
- Quality ceiling is higher (no hallucinated flags or incorrect syntax)
- Automated validation is possible (run the described command and check exit code)

---

## 3. Data pipeline differences

### Source-grounded generation

The existing `distill.py` / `batch_distill.py` pipeline works entirely from seed
queries — the teacher invents both query and response. For a technical model, this is
wrong: the teacher may hallucinate flag behavior or outdated syntax.

The Linux pipeline needs **grounded generation**: the teacher receives a chunk of source
text (man page section or tldr page) and generates Q&A pairs *from that text*. The
answer must be extractable from or paraphrasable from the source. This eliminates
hallucinations by construction.

System prompt for grounded generation:
```
You are generating training data for a command-line oracle.
You will be given a section of a Linux man page or tldr entry.
Generate QUESTION|ANSWER pairs where:
- The question is what a user would actually ask at a terminal
- The answer is the shortest complete response (1-2 sentences max, no more than 150 chars)
- The answer must be grounded in the source text — do not add information
- Preserve exact flag names, paths, and command syntax
- Output one pair per line in the format: QUESTION|ANSWER
Do not output any other text.
```

User message format:
```
Source text:
{man_page_section}

Generate 5 Q|A pairs from this text:
```

### Answer compression

The teacher must be constrained to oracle-style brevity. Unconstrained, it will reproduce
the man page prose ("The `-v` flag, which stands for verbose, enables verbose output mode,
which causes the program to print additional..."). The system prompt above enforces this,
but the `max_tokens` limit is also a hard constraint: set it to ~50 tokens (≈200 chars).

### Case normalization: the critical difference

The existing trainer does this:
```python
inp, tgt = make_sequence(q.upper().strip(), r.upper().strip(), model.max_len)
```

**This MUST be disabled for the Linux model.** Command-line content is case-sensitive:
- `ls -la` is not the same as `LS -LA`
- `/etc/fstab` is not `/ETC/FSTAB`
- `grep -E` is different from `grep -e` on some implementations
- Flag names like `--no-preserve-root` are meaningful precisely as-is

The conversational model benefits from case normalization because its vocabulary is just
English words, and uppercase reduces the effective vocabulary for a byte-level model
(the lowercase bytes 97-122 are never seen in training, freeing ~16% of the embedding
table for the conversational model). For technical content this trade-off reverses:
**uppercase suppression destroys information**.

Required change: add a `--preserve-case` flag to `train_transformer.py` that skips the
`.upper()` call. This is a single conditional in `make_sequence()`.

Implication: the Linux guru model cannot share training data with the conversational
models without case conversion. They are truly separate data pipelines.

### Format: still `query|response`, still ASCII

The `|` separator and ASCII-only constraint are fine. Linux commands are ASCII.
Non-ASCII characters do appear in man pages (Unicode in author names, some special
symbols) but can be stripped or transliterated without loss for the oracle use case.

The query format should also use lowercase input (as users would type it) with mixed-case
output (preserving flag syntax). This is an additional reason why the existing
`.upper()` normalization is wrong: users type "how do i use grep" in lowercase, and the
model needs to both accept lowercase queries and emit mixed-case answers.

### Prompt structure for grounded generation

For man page sections:
```
What does `find -exec {} +` do?|Like -exec {} ; but batches all matching files into a single command invocation -- faster and more efficient.
How to find files modified in the last 7 days?|find /path -mtime -7 -- finds files modified within 7 days. Use -mtime +7 for older than 7 days.
What is the difference between -exec and -execdir?|-execdir runs the command from the file's parent directory -- safer than -exec for security-sensitive operations.
```

For tldr pages (simpler — already structured):
```python
def tldr_to_pairs(tldr_text):
    """Convert a tldr page to Q|A pairs directly (no teacher needed for examples)."""
    lines = tldr_text.strip().split('\n')
    cmd = lines[0].lstrip('# ').strip()
    desc = lines[1].lstrip('> ').strip() if len(lines) > 1 else ""
    pairs = []
    # Description pair
    if desc:
        pairs.append(f"what does {cmd} do?|{desc}")
    # Example pairs
    i = 2
    while i < len(lines):
        if lines[i].startswith('- ') and i+1 < len(lines) and '`' in lines[i+1]:
            annotation = lines[i][2:].rstrip(':').strip()
            command = lines[i+1].strip().strip('`')
            pairs.append(f"{annotation}?|`{command}`")
            i += 2
        else:
            i += 1
    return pairs
```

Note: tldr example pairs can be generated **without any teacher calls** — the format
is already Q&A. This halves the teacher cost for the tldr-derived corpus.

---

## 4. Architecture fit

### Byte-level tokenization: the honest assessment

**Arguments for byte-level in this domain:**

1. **Command syntax is byte-precise.** `-rf` means something specific; `-Fr` means
   something slightly different. A subword tokenizer might merge flags in unpredictable
   ways. Byte-level has no merging — `-rf` is always 3 tokens.
2. **Paths are exact.** `/etc/fstab`, `/dev/null`, `~/.bashrc` tokenize as-is.
   A subword tokenizer trained on English web text has poor coverage of Unix paths.
3. **Flags are short.** The "long flag problem" (see below) is real but manageable
   with oracle-style compression.
4. **Same inference stack** — no new Wasm kernel, no new TS orchestrator changes needed.

**Arguments against:**

1. **Long flag names eat context.** `--no-preserve-root` is 18 bytes = 18 tokens.
   In a 256-ctx model with a 30-byte query, answering a question about two such flags
   plus a brief description consumes most of the budget.
2. **Repeated prefixes waste capacity.** Every `/usr/bin/` path wastes 9 tokens on the
   prefix. Every `--` wastes 2 tokens per long flag. A subword tokenizer would encode
   `/usr/bin/` in 2-3 tokens.
3. **The model sees ASCII bytes individually.** It must learn that `c`, `h`, `m`, `o`,
   `d` together mean something — no help from subword embeddings. This is fine for
   common commands but harder for obscure ones.

**Verdict:** Byte-level is viable and is clearly the right choice for this project given
the shared inference stack. The long-flag problem is mitigated by oracle-style answers
that prioritize short flags over long equivalents (`-v` before `--verbose`).

### Recommended architecture

Based on the data above, the recommended config for a dedicated Linux guru model:

| Parameter | Value | Rationale |
|---|---|---|
| d_model | 384 | Same as Shade — proven feasible, adequate for technical content |
| n_heads | 6 | d_head = 64 (required by TS forward) |
| n_layers | 8 | More depth than Shade (8 vs 6) — technical content needs more pattern composition |
| d_ff | 1536 | Same ratio as Shade |
| ctx | 256 | Covers all measured answers with headroom; matches Specter |
| vocab | 258 | Identical — same inference stack |
| Params | ~14M | "Wraith-C" from the config table |
| fp32 size | ~58 MB | Shippable on CDN; Shade-tier weight |
| 4-bit size | ~7.2 MB | Excellent — lighter than Shade 4-bit |

Why 8 layers rather than 6? Technical content has more compositional structure than
casual conversation. A question about `find -exec {} +` requires the model to associate:
(a) the `find` command context, (b) the `-exec` flag semantics, (c) the `{}` placeholder,
(d) the `+` vs `;` terminator distinction. This is more levels of indirection than
"what's your favorite color?". More layers help here.

The 256-context + 8-layer config makes this model heavier per token than Shade but
lighter than Specter. The 4-bit size of ~7.2 MB is very appealing for browser delivery.

### Should it share the Specter config?

It could — Specter is d=768, 8 layers, 256 ctx. But Specter is 57M params / 28 MB at
4-bit. For a domain-specialized model targeting one narrow use case, 57M is excessive.
The Wraith-C config (14M params) should specialize as well or better at a quarter of
the cost. Use Specter's ctx (256) but Shade's width and more depth.

---

## 5. Browser inference viability

### What works well (from weights, no RAG)

The oracle use case maps cleanly to what a small LM does well:

- **Pattern matching + reformatting.** "What does `-v` do in `grep`?" is a lookup
  pattern. The model has seen this association in hundreds of training pairs.
- **Short factual completions.** "The `-h` flag makes output human-readable" is the
  kind of short, high-confidence completion small LMs are reliable at.
- **Common command invocations.** The 200 most-used Linux commands appear thousands of
  times in the training corpus in various phrasings — effectively memorized.

### What fails badly

- **Rare commands.** If `inotifywait` appears only twice in training, the model cannot
  reliably recall its flags. It will hallucinate or fall back to generic descriptions.
- **Multi-step procedures.** "How do I set up a firewall with nftables?" requires 10+
  sequential steps with specific syntax. This exceeds context and requires procedural
  reasoning the model isn't trained for.
- **Version-specific behavior.** "Does `cp --reflink` work on my filesystem?" requires
  runtime knowledge. The model can only describe the flag, not evaluate the system.
- **Disambiguation.** "How do I set the date?" could mean `date`, `timedatectl`,
  `hwclock`, or `ntpdate` depending on context. Without system state, the model has to
  pick one.
- **Novel combinations.** A pipeline like `find . -name '*.log' | xargs grep -l ERROR |
  xargs rm` requires composing three commands. The model may hallucinate syntax at the
  joints.

### Value proposition over `man ls` or `tldr ls`

This is the hard question. The honest answer:

**Use case where it wins:** The user is in a terminal, mid-task, and wants to ask
in natural language without leaving the terminal. `man grep` requires knowing to `man grep`.
`tldr grep` requires a network connection and the `tldr` client installed. The browser-
based oracle requires only the tab to be open — and answers in natural-language queries,
not command lookups.

**Use case where it loses:** Anything that requires comprehensive or guaranteed-accurate
information. `tldr ls` is human-curated and always correct. This model may occasionally
confuse flags or conflate command versions. For anything important, `man` wins.

**The honest positioning:** This is a "I'm pretty sure I need `chmod -R` but can't
remember if it's uppercase or not" tool, not a "I need to know every flag `find` supports"
tool. It is fast, frictionless, and conversational. It is not authoritative.

### CRT terminal aesthetic — the fit is perfect

The existing demo has a green-screen CRT terminal aesthetic and the project is called
"Ghost in the Machine." A terminal-oracle mode in that same UI is a natural fit —
possibly the most on-brand thing this project could ship. The original Z80 micro-LLM
targeted a CRT-terminal aesthetic; a model that literally knows the terminal is the
poetic completion of that idea.

Proposed UI: pressing `Ctrl+L` (or typing `LINUX` or `ORACLE`) in the terminal switches
from conversational mode to Linux-oracle mode, with a mode indicator in the status bar
(e.g., `[ORACLE: LINUX]`). The model switcher planned in Phase 7 (`models.json`-driven)
covers this exactly.

---

## 6. Naming and tier table

### Naming

The existing names follow a ghost/spirit theme: **Wisp** (faint), **Shade** (dim spirit),
**Specter** (apparition). For a domain-specialized model:

- **Wraith** — a ghost that haunts a specific place (a terminal, in this case). Strong
  fit. Not too similar to Specter.
- **Phantom** — works, but slightly more generic.
- **Oracle** — functional, not thematic. Breaks the ghost naming convention.
- **Banshee** — Irish spirit that warns/informs. Thematic, memorable.

**Recommendation: Wraith.** It fits the theme, is distinct from existing names, and has
connotations of something lurking in the terminal — exactly right.

### Proposed tier table row

```
| **Wraith** (linux) | 384 | 1536 | 8 | 6 | 256 | 14.5M | ~58 MB | ~7.2 MB |
```

Full updated table:

| Name | d_model | d_ff | Layers | Heads | Ctx | Params | Float32 | 4-bit |
|---|---|---|---|---|---|---|---|---|
| **Wisp** (micro) | 256 | 1024 | 4 | 4 | 64 | 3.3M | 13 MB | ~1.7 MB |
| **Shade** (small) | 384 | 1536 | 6 | 6 | 128 | 10.9M | 42 MB | ~5.5 MB |
| **Specter** (large) | 768 | 3072 | 8 | 12 | 256 | 57.3M | 219 MB | ~28 MB |
| **Wraith** (linux) | 384 | 1536 | 8 | 6 | 256 | 14.5M | ~58 MB | ~7.2 MB |

### Domain model track, not a size variant

Wraith should be positioned as a **separate domain model track**, not a 4th tier in the
size progression. It is not "bigger than Specter" — it is a peer to Shade in raw capacity
but specialized. The tier table should show two tracks:

```
Generalist track:     Wisp → Shade → Specter
Domain model track:   Wraith (Linux)
```

This distinction matters for the UI (model switcher shows domain/generalist separately)
and for the training strategy (domain models skip the conversational pipeline entirely).

---

## 7. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Model memorizes wrong flags | High | Source-grounded generation + automated validation (run described commands) |
| Context too short for multi-step answers | Medium | Oracle-style compression in training; defer 512-ctx to after KV cache |
| Case normalization not disabled | High | Add `--preserve-case` flag; treat as a prerequisite, not an afterthought |
| Training data too domain-narrow (catastrophic forgetting of English) | Medium | Include 10-20% general conversational pairs to preserve language model baseline |
| Not useful enough vs. `tldr` | Low | MVP validates this before full build commitment |
| Teacher hallucination about obscure tools | Medium | Ground ALL generation in source text; reject any answer not checkable against source |

---

## 8. Recommendation

**Yes, build Wraith — but validate the MVP first.**

### Why yes

1. The data exists, is high-quality, and is CC0 licensed (tldr) or GPL (man pages).
2. The measured answer lengths (mean 71 bytes) confirm context viability at Shade/Specter
   scale — this is not a theoretical model, the numbers work.
3. The byte-level encoding is a natural fit for command-line content.
4. The CRT terminal aesthetic of the existing demo makes this the most on-brand possible
   extension.
5. The grounded data pipeline has a higher quality ceiling than the conversational one —
   no hallucinations by construction if source-grounded correctly.
6. Wraith at 4-bit is only ~7.2 MB — the second-lightest model in the stack, easy to
   ship.

### Why gate it on an MVP

The single biggest uncertainty is not technical — it is "does this actually work well
enough to be useful?" A byte-level 14M-parameter model answering from weights is an
ambitious oracle. It may be reliable for common commands and embarrassingly wrong for
obscure ones. An MVP tests this before investing in the full pipeline.

### The MVP

**Goal:** Verify that a small byte-level model can correctly answer 10 test questions
about Linux commands after training on a tiny curated dataset.

**Scope:**

1. **Data:** tldr-pages only, English, ~500 pages (covering the most-used commands).
   Parse directly to Q|A pairs using the structured format — no teacher calls needed
   for the example pairs. Add 200-300 teacher-generated rephrasing pairs for natural
   language query variation.
   - Total: ~3,000–4,000 pairs
   - Cost: ~300 teacher calls (for rephrasings only), ~15 minutes to generate

2. **Architecture:** Wisp-scale (d=256, 4 layers) **but** ctx=128 (Shade's context).
   This is the cheapest possible config that can fit a real Linux answer. Train for
   200-400 epochs on an RTX 5080: ~30-60 minutes.
   - **Critical:** Disable `.upper()` normalization

3. **Evaluation:** 10 hand-chosen test questions not present in training data (use
   commands from the second half of the tldr-pages alphabet — train on A-L, test on M-Z).
   Grade manually: 0 (wrong/hallucinated), 1 (correct direction but imprecise), 2 (correct
   and precise).
   - Target: 15+/20 points to proceed to full Wraith training

4. **Gate decision:** If the MVP scores ≥15/20 on the test questions, proceed to the
   full Wraith build (Wraith-C config, full tldr + man page pipeline, 50K+ pairs). If
   it scores below that, the byte-level approach needs rethinking (possibly the model
   needs more capacity, or the queries need further compression).

### Full build if MVP passes

1. **Data pipeline:** `expand_data_linux.py` (new script, parallel to conversational
   `expand_data.py`) that processes tldr pages and man page sections through the
   grounded teacher prompt. Target: 50K pairs (tldr rephrasings) + 20K pairs (man
   pages sections 1+8) = ~70K total pairs.
2. **Architecture:** Wraith-C (d=384, 8 layers, ctx=256). Train ~20-30 epochs on 70K
   pairs with val split and early stopping.
3. **Case handling:** `--preserve-case` flag in trainer; separate training data file
   that is NOT uppercased.
4. **Shipping:** 4-bit quantization → ~7.2 MB; served as `model_wraith.*` alongside
   the generalist models per the Phase 7 model switcher.
5. **UI:** `[ORACLE: LINUX]` mode in the existing CRT terminal demo.

---

## Appendix: measured data points

```
Man page sizes (raw text, `man X | col -b | wc -c`):
  cat:    1.9 KB     ls:     7.5 KB     chmod:  7.1 KB
  grep:  30.0 KB     ssh:   48.0 KB     tar:   39.7 KB
  find:  81.1 KB     awk:  112.4 KB     mount: 109.4 KB

Man page counts by section (`man -k . | awk '{print $2}' | sort | uniq -c`):
  Section (1) user commands:           3,440
  Section (8) system administration:   1,332
  Section (3) library functions:      12,847
  Section (3ssl) SSL library:          6,192
  (Sections 2, 5, 7, 3x, 3perl...):   ~2,800

tldr-pages stats (verified via GitHub API):
  Total pages across all platforms:   7,243
  License: CC0-1.0 (public domain)
  Total English text: ~4 MB
  Stars: 62,800+

Oracle answer length sample (20 questions):
  Mean: 71 bytes   Min: 47 bytes   Max: 92 bytes
  Fit in ≤100 bytes: 20/20 (100%)
  Fit in ≤150 bytes: 20/20 (100%)

Wraith-C config parameter count:
  d=384, L=8, ff=1536, ctx=256, vocab=258 → 14.5M params
  fp32: ~58 MB   8-bit: ~14.5 MB   4-bit: ~7.2 MB
```
