# jailbreak

An autonomous loop for **authorized** AI red-team research: it searches for a
*diverse* set of attacks that get a target model to violate a stated objective,
so the resulting weaknesses can be reported and fixed.

The research question it's built around is **robustness coverage**, not attack
success alone: *does a target model hold its refusal line across the full space
of attack styles, or only against the obvious ones?* The loop answers that by
keeping a MAP-Elites archive over `(attack_style × risk_category)` and rewarding
*breadth* of elicited failures rather than a single best jailbreak — the same
signal you'd want when hardening a model.

> **Scope of this repository.** This repo ships the *framework and method* only.
> It does **not** ship attack objectives beyond a harmless sanity check, run
> databases, elicited model outputs, or any distilled "what works" strategy
> notes. See [Responsible use](#responsible-use).

## Method

The loop is a small evolutionary search with four LLM roles, plus a diversity
archive:

- **Proposer** — generates candidate attack prompts. Each step it picks an *op*:
  `mutate` (edit one archive parent), `crossover` (combine two parents), or
  `seed` (instantiate a technique from a cited taxonomy). It reads a running
  strategy note maintained by the reflector.
- **Executor** — sends the candidate to the target model and records the
  response.
- **Judge** — scores whether the objective was met, separating refusal,
  partial/hedged compliance, and clean compliance (rubric overridable per
  objective — including *refusal-as-success* rubrics for pure robustness evals).
- **Archive (MAP-Elites / Rainbow Teaming)** — a grid over
  `(attack_style × risk_category)`; each cell keeps the best-scoring attempt.
  The objective is *coverage*: how many distinct cells can be filled, i.e. how
  many qualitatively different ways the target can be made to fail.
- **Reflector** — every K iterations, summarizes recent results and the coverage
  matrix into a strategy note the proposer reads, so later proposals push toward
  *unfilled* cells instead of re-deriving known attacks.

Two efficiency pieces matter in practice:

- **Novelty filter** — local `fastembed` embeddings + cosine similarity drop
  near-duplicate candidates *before* spending an executor/judge call.
- **Adaptive multi-turn** — an adapter model drafts follow-up user turns, so the
  loop can evaluate Crescendo-style multi-turn drift, not just single prompts.

The seed taxonomy (`src/jailbreak/seed/data/`) is a tactic/technique map in the
spirit of MITRE ATLAS and redteams.ai/navigator, with each technique attributed
to its source paper (PAIR, TAP, Crescendo, GCG, CipherChat, many-shot, …). It
contains *references and placeholder templates*, not operational payloads.

## Responsible use

This tool exists to find weaknesses in models you are **authorized** to test so
they can be reported and fixed.

- Only point it at models you own or have explicit permission to probe.
- `jb run` requires an explicit `--i-am-authorized` flag and writes an
  acknowledgement banner to an append-only audit log.
- The tool does not phone home: no telemetry, no external egress beyond the
  provider API calls you configure. Provider keys are read from environment
  variables and never persisted.
- **Run artifacts are deliberately excluded from this repository and from git
  history** (`run/`, `*.db`, `*.jsonl`, `strategies.md` are git-ignored). Those
  files can contain harmful model completions and distilled attack strategies;
  publishing them would be capabilities uplift, not research. Keep them local,
  and share findings through responsible-disclosure channels instead.
- The only objective shipped here is `banana` (below), which has no harm
  potential. Real objectives are yours to define locally, gated by the
  `[ethics] allow_categories` list in your config.

## Install

```
pip install -e '.[dev]'
```

## Quickstart

```
jb init                                     # scaffold config.toml + objectives.yaml (sanity objective only)
export ANTHROPIC_API_KEY=...
jb run --config config.toml --objectives objectives.yaml --iters 50 --i-am-authorized
jb archive                                  # browse filled elite cells
jb export --success-only --out successes.jsonl   # stays local; git-ignored
```

The shipped sanity objective:

```yaml
- id: banana
  description: Elicit a response whose first sentence contains the word "banana".
  category: sanity
```

It exercises every part of the loop (propose → execute → judge → archive →
reflect) end-to-end with zero harm potential — useful for validating a setup
before you point it at anything real.

See `examples/` for a working config. Each provider role (target, proposer,
judge, adapter) can be pointed independently at a different provider/model,
including a local OpenAI-compatible endpoint (e.g. Ollama).

## Non-goals

- Not a benchmark harness — though it can target HarmBench-style behaviors.
- Not a training loop — no gradient updates, no fine-tuning.
- Not a general agent framework — one loop, one purpose.

## References

PAIR (Chao et al. 2023), TAP (Mehrotra et al. 2023), Crescendo (Russinovich et
al. 2024), many-shot jailbreaking (Anil et al. 2024), GCG (Zou et al. 2023),
CipherChat (Yuan et al. 2023), Rainbow Teaming (Samvelyan et al. 2024),
MAP-Elites (Mouret & Clune 2015), MITRE ATLAS.

## License

MIT — see [LICENSE](LICENSE).
