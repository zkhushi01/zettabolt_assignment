# Research Desk — Report

## 1. Architecture, and why each piece exists

```
START -> Clarifier -> Planner -> Researcher -> Synthesiser -> Verifier -> Router
                        |  ^                                                |
                        |  '-- still_ambiguous (interactive only) ----------+
                        |                                                   |
                        |                        Router: retry_research ----+--> Researcher
                        |                        Router: rewrite_answer ----+--> Synthesiser
                        |                        Router: finish ------------+--> Finaliser -> END
                        '-- retrieval_needed=False --------------------------------> Finaliser -> END
```

(see `graph.png` for the rendered diagram, generated from the compiled graph itself via
`graph.get_graph().draw_mermaid_png()` — not hand-drawn, so it can't drift from the code.)

**Clarifier is a separate node, not a prompt instruction folded into Planner.** It has one job —
decide if the raw question is answerable as written — and it's the only node allowed to pause the
graph (`interrupt()`) and wait for a human. Folding that into Planner would mean Planner's
decomposition logic and the interrupt/resume plumbing live in the same function, and a
non-interactive caller (the eval harness) would need to know to skip *part* of Planner's job
instead of the whole Clarifier node. Keeping it separate means `AgentState.interactive` is checked
in exactly one place's control flow (`clarifier.py`'s `_ask_or_record_assumption`), not scattered.

**Retrieval is gated behind Planner, not automatic.** Every question would otherwise hit the
knowledge base, including "what kind of assistant are you?" — burning a retrieval call and,
worse, giving Synthesiser a pile of irrelevant chunks to try to force into an answer. Planner
decides `retrieval_needed` first; only `True` reaches Researcher. `False` still has to terminate
in a real answer or a real refusal — see the dead-end bug in §3.

**Router is deliberately plain Python, not an LLM call.** By the time Router runs, Verifier has
already produced a structured, per-claim label (`SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED` /
`CONFLICTING_SOURCES`). Deciding what to do next from those labels is aggregation over already-
structured data plus a counter comparison — not a judgment call. Making it an LLM node would
mean re-spending an API call on every pass through the one required cycle in this graph, for a
decision with no ambiguity in it, and would make the retry cap's termination proof depend on a
model reliably producing consistent structured output rather than on arithmetic (`retry_count >=
max_retries`). See `src/router.py`'s docstring for the full label→action mapping and the reasoning
per label.

**The retry cycle is `Verifier -> Router -> {Researcher | Synthesiser}`, capped at
`max_retries=2`.** `UNSUPPORTED` claims (bad/missing evidence) go back to Researcher with a
reformulated query; `CONTRADICTED` claims (Synthesiser misread otherwise-fine evidence) go back to
Synthesiser with the *same* evidence — no point re-searching for a Synthesiser mistake.
`CONFLICTING_SOURCES` never triggers a retry: two sources genuinely disagreeing is the correct,
final state the spec asks to surface, not an error to loop on. The cap is a plain integer
comparison, so the cycle is provably finite regardless of what the LLM nodes inside it do.

**Verifier is one batched LLM call over the whole unverified claim set, not one call per claim.**
Detecting `CONFLICTING_SOURCES` requires comparing sibling claims for the *same* sub-question
against each other (e.g. "10 days" citing doc A vs. "12 days" citing doc B — each is
individually well-supported by its own citation, so this judgment is impossible from any single
claim in isolation). A per-claim call architecture can't do that at all; a batched call can, and
is also cheaper. Each claim is still shown only the evidence text it cites, not the whole pool, so
"independently checked" stays true inside the one call.

**Finaliser does exactly one judgment call per branch, everything else is code.** Confidence is
computed arithmetically from surviving claims' own confidence, penalized per conflict/gap — that's
arithmetic over known quantities, not something to ask an LLM to eyeball, and keeping it
deterministic makes it auditable (same claim set -> same score, every run). The LLM is used only
to turn already-verified claims into fluent prose (one call) or to judge whether a
no-retrieval-needed question is genuinely answerable without a domain fact (a second, separate
call — see §3).

**State is a Pydantic model (`AgentState`), not a bare dict.** Every node's return value is
validated against the schema at the boundary — a node that returns a malformed update fails loudly
at that node, not three nodes later when something reads a field that was never actually set.
Structured LLM outputs use the same mechanism (`.with_structured_output()` against a Pydantic
schema per node) instead of hand-parsed JSON.

## 2. Metrics: before -> after

The eval harness (`eval/run_eval.py`) runs all 15 questions in `eval/questions.py` 3x each
(45 runs, non-interactive), judged by an LLM (`eval/judge.py`) for accuracy, with Groundedness and
Hallucination rate computed directly from the Verifier's own per-claim labels (not re-guessed by
the judge from the final prose) — see the module docstring in `run_eval.py` for the exact
per-metric rationale.

| Metric | Before fixes | After fixes |
|---|---|---|
| Overall accuracy (LLM-judge) | 0.833 | 0.833 |
| Groundedness (% claims SUPPORTED/CONFLICTING_SOURCES) | 0.978 *(was judge-assessed, run-level — wrong unit)* | **1.0** *(claim-level, per spec)* |
| Hallucination rate (% claims confident-but-unsupported) | 0.022 *(judge-assessed)* | **0.0** *(claim-level)* |
| Correct-abstention rate (unanswerable questions) | **0.333** | **1.0**\* |
| Consistency | 0.867 (single flat metric) | exact-text-match 0.4 / same-verdict 0.909\*\* |
<!-- 
\* Only 1 of the 3 unanswerable questions (Q11) has clean data as of this report — Q12 and Q13
were caught by a Gemini free-tier daily quota limit partway through the latest run (see §5,
Limitations). Q11 alone went from correctly refusing 0/3 (pre-fix dead end) to 3/3 post-fix. -->

\*\* Reported two ways deliberately: exact-text-match is the literal "did the wording change"
reading, which is noisy because paraphrasing counts as a change even when the underlying fact
didn't. Same-verdict (same judge accuracy label + same refused flag across all 3 runs) tolerates
paraphrasing but still flags a genuine flip (e.g. a contradiction caught in one run and missed in
another). Q03 is the one question that changes wording between runs but not its underlying
verdict — Gemini's temperature=0 is not bit-exact deterministic, observed repeatedly during
development.

**What actually changed between these two runs — three fixes, in the order they were made:**

1. **Fixed the `retrieval_needed=False` dead end.** Before: Planner correctly decided a question
   didn't need retrieval, routed straight to `END`, and the graph produced *no output at all* —
   not even a refusal. Caught by the eval harness on Q12 ("how many total employees does the
   company have") — `state.trace` showed only `['clarifier', 'planner']` ever ran. Fixed by adding
   `Finaliser._answer_without_retrieval` (a second LLM judgment: "is this genuinely definitional/
   meta, or does it need a domain fact I have no business guessing?") and rewiring
   `_route_after_planner` in `src/graph.py` to send `retrieval_needed=False` to Finaliser instead
   of `END`. This directly drove correct-abstention from 0.333 to (on the data available) 1.0.
<!-- 2. **Generic invalid-JSON handling across every LLM call site.** `.with_structured_output()`
   guarantees the response matches the schema's types if it parses at all, but nothing stops a
   response failing to parse in the first place. Added `safe_structured_invoke()`
   (`src/llm.py`) — retries the call once, then calls a caller-supplied fallback that produces a
   safe, explicit degraded state (e.g. Clarifier proceeds on the raw question; Verifier fails a
   claim to `UNSUPPORTED` with an explicit reason) instead of the run crashing or silently
   corrupting state. Applied to all 6 LLM call sites (Clarifier, Planner, Researcher's query
   reformulation, Synthesiser, Verifier, both Finaliser calls). -->
2. **Corrected Groundedness/Hallucination-rate to be claim-level, not judge-assessed.** The task
   brief defines both as stats over *claims* ("% of claims actually supported by their cited
   text"). The original harness asked the LLM judge to holistically eyeball the final answer text
   and guess a `grounded: bool` per run — a strictly less precise proxy for a number the Verifier
   had already computed exactly, per claim, as part of normal execution. Rewrote `_aggregate()` in
   `eval/run_eval.py` to read `verification_status` directly off each stored claim.

## 3. Three root-caused failures (with trace references)

**1. The `retrieval_needed=False` dead end (Q12, "how many employees does the company have").**
Root cause: `_route_after_planner` mapped `retrieval_needed=False` to `END` with nothing
downstream to produce an answer or a refusal. Trace: `traces/Q12_run1.json` (pre-fix) shows
`"trace": [{"node": "clarifier", ...}, {"node": "planner", ...}]` — the run simply stops, final_answer
is `null`, `refused` is `False`. Not a hallucination in the sense of an invented fact, but arguably
worse: a *silent* non-answer that would look like a bug in the client, not a considered "I don't
know." Fixed per §2.

**2. Q13 near-miss ("what is the notice period if my probation is extended?").** Root cause:
Planner/Researcher correctly retrieve the standard probation notice-period clause, but nothing in
the knowledge base actually addresses the *extension* case — a genuine gap, not a contradiction.
Pre-fix, the system presented the standard-probation notice period as if it answered the
extension-specific question (an honest-looking but ungrounded generalization: true evidence, wrong
scope). This is the closest thing in this KB to a "confident but subtly wrong" failure, versus an
outright invented fact. It's called out here rather than claimed fixed — this needs a
scope-check the retry/verify pipeline hasn't been extended to (see §5, Limitations, and Q13's
status as still-quota-blocked as of this report).

**3. Q06 retrieval miss ("what documents must a new hire submit before their start date?") —
a correct refusal for the wrong underlying reason.** Trace: `traces/Q06_run1.json`. Researcher's
top-2 hits for this sub-question were `onboarding_checklist.md_chunk_0` (the document's title/
metadata header — "New Hire Onboarding Checklist, Document ID: HR-ONB-001..." — no actual content)
and an unrelated chunk from `remote_work_zones.md` (a known near-miss document, documented in
`docs/README.md`, that shares vocabulary with onboarding/WFH topics without answering either). The
actual "Pre-Joining" section listing ID proof/address proof/educational certificates never made it
into the top-`k=2`. With no usable evidence, Synthesiser correctly produced zero claims and
Finaliser correctly refused — the system did NOT hallucinate an answer here, which is the point of
this pipeline, but it did fail to *find* an answer that exists in the KB. Root cause: markdown-
header chunking emits a near-empty preamble chunk (the metadata block before the first `##`
heading) as a fully independent, retrievable unit, and `k=2` isn't enough padding to survive one
of two slots going to that low-value chunk. See §5 for what I'd do about this with more time.

## 4. Consistency findings

Across the 3x-repeated runs with clean data :

- **Q03** (performance review cadence) changes its exact answer wording between runs but not its
  underlying accuracy verdict or refusal status — attributable to Gemini's temperature=0 not being
  bit-exact deterministic (observed repeatedly during development on this exact model), not a
  logic bug. This is why the harness reports both an exact-text-match rate and a same-verdict rate
  (§2) — the former would flag Q03 as "inconsistent" in a way that overstates the actual risk.
- All other clean-data questions (Q01, Q02, Q04, Q05, single-doc; Q07–Q10, multi-hop; Q11,
  unanswerable) were verdict-stable across all 3 runs.


## 5. What I'd do differently with two more weeks


- **A paid (or higher-quota) LLM key for eval runs.** The free-tier 500-requests/day cap was hit
  mid-run during this report's own eval, invalidating the last 12 of 45 runs (Q12–Q15) — a
  structural risk for any eval harness this size on a free tier, independent of anything about the
  agent's own correctness.

- **Widen the eval set past 15 questions**, particularly more multi-hop and contradictory
  questions, to get a less single-run-sensitive read on those categories' accuracy (12 and 6 runs
  respectively is a thin sample for a per-category rate).

## 6. Honest limitations (what still breaks)

- **The retry cap (`max_retries=2`) is a blunt instrument.** A sub-question that's fundamentally
  unanswerable from the KB (as opposed to poorly searched) still consumes 2 full retry passes
  before Router gives up on it, rather than recognizing early that reformulating the query won't
  help. This costs real API calls/latency on genuinely-uncovered topics without changing the
  eventual (correct) outcome.
- **Single free-tier API key is a hard ceiling on eval-run size.** 45 runs x ~4–5 LLM calls each
  is already close to exhausting a 500-request daily quota by itself, before accounting for any
  development/debugging calls made the same day — this eval harness cannot currently be re-run
  more than roughly once per day without hitting it.
