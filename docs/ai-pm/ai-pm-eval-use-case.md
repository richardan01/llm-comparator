# AI PM Evaluation Use Case

> How and why an AI Product Manager runs a structured LLM response comparison — before picking a model or shipping a feature.

## The scenario

A support org wants to ship an **answer-drafting assistant**. An agent pastes a customer question; the assistant drafts a reply **grounded in the company's help-center docs**. The agent edits and sends.

We're choosing the model behind the drafting step. Two candidates are on the table (call them **Model A** and **Model B**). Cost, latency, and quality all differ. We need a defensible recommendation — not "the demo felt better."

> The named-model version of this question ("Claude vs GPT-4-class vs an open model") works the same way. We use anonymized A/B labels here so the *method* is the takeaway, not a leaderboard.

## The decision at stake

| If we choose wrong... | Consequence |
|---|---|
| Model hallucinates a policy | Agent sends wrong info → trust + compliance risk |
| Model is verbose / off-tone | Agents stop using it → feature dies |
| Model is slow / costly | Margins erode, or we cap usage |
| Model is "safe" but useless | No draft worth editing → no time saved |

So the evaluation isn't "which is smarter." It's **which produces drafts an agent can trust and ship, at a cost we can live with.**

## Why this is a product-quality problem, not only a technical benchmark

Benchmarks (MMLU, HELM, leaderboard arenas) measure general capability on shared, generic tasks. They're useful for **shortlisting** candidate models — but they don't decide what ships.

| Benchmark thinking | Product-quality thinking |
|---|---|
| "Which model scores higher overall?" | "Which response is right for *this* workflow?" |
| One ranked number per model | Per-dimension scores, weighted by use case |
| Generic prompts | Real prompts from real user paths |
| Average performance | Worst-case behavior (vetoes on grounding/safety) |
| Capability only | Capability ↔ cost ↔ latency ↔ tone ↔ safety |
| Model-level | Response-level *and* model-level |

A model can win a benchmark and still be the wrong call here — e.g. it's eager, fluent, and ungrounded (exactly the failure in the [sample review](sample-comparison-review.md)). Product quality is **contextual, multi-dimensional, and trade-off bound**, which is why a PM — not a leaderboard — owns it.

## Why a PM owns this (not just ML)

Engineers can report benchmark scores. The PM owns the questions benchmarks don't answer:

- **What "good" means for *this* workflow** — for grounded support drafts, *grounding* and *safety* outrank cleverness.
- **Which failures are unacceptable vs annoying** — a hallucinated refund policy is a blocker; a slightly stiff tone is not.
- **The trade-off the business will accept** — is a 15% quality gain worth 2× cost and 400ms more latency?

## Where comparison fits in the product loop

```
Define use case  →  Pick quality dimensions  →  Collect candidate responses
      ↑                                                      │
      │                                                      ▼
Product decision  ←  Score side-by-side (rubric)  ←  Run same prompts on A & B
      │
      ▼
Ship / re-prompt / re-evaluate
```

The comparison tool (PAIR-code/llm-comparator) renders the side-by-side view. This repo's docs supply the **rubric, weighting, and decision logic** the PM brings to it.

## What a good evaluation produces

1. A **shared definition of quality** for the use case (the rubric).
2. **Scored, side-by-side evidence** instead of anecdotes.
3. A **recommendation with trade-offs**, so the call survives scrutiny.
4. A **re-runnable process** — when a new model ships, we re-score, not re-argue.

## Scope & honesty

This is a learning exercise. The scores in [`sample-comparison-review.md`](sample-comparison-review.md) are illustrative judgments on a constructed prompt, not a vendor benchmark. The value is the **method**: define quality → score → decide → revisit.
