# Product Decision Notes

> How the [sample review](sample-comparison-review.md) becomes a recommendation — the trade-offs weighed, the call made, and what would change it.

## Recommendation

**Ship the answer-drafting assistant on Model B**, with grounding + safety guardrails kept on. Model A is **not viable** for this grounded use case in its current form.

This is a decision about *fit for this workflow*, not a claim that Model B is "better" in general.

## Why (in one breath)

The product cannot afford ungrounded, off-policy drafts. Model A produced a fluent answer that **invented a refund policy and committed the company to an unauthorized refund** — the single most damaging failure for a support tool. Model B stayed grounded, refused the out-of-window refund correctly, and still offered a useful next step. On the dimensions this product weights most (grounding 25%, safety 20%), it isn't close.

## Trade-offs we accepted

| Trade-off | Call | Rationale |
|---|---|---|
| **Quality ↔ Cost** | Pay more for B if needed | A hallucinated-policy incident costs more than the token delta |
| **Quality ↔ Latency** | Accept modest extra latency | An agent will wait a beat for a draft they can trust |
| **Capability ↔ Safety** | Favor the on-policy model | In support, staying in-bounds beats eagerness |
| **Peak ↔ Consistency** | Optimize for consistency | One good A answer doesn't offset its risk profile |

These are explicit choices — if the economics or use case shift, so does the call.

## What this decision is *not*

- **Not** a general ranking of Model A vs Model B — A may win on a brainstorming or creative task where grounding barely matters.
- **Not** based on a real vendor benchmark — it's one constructed, grounded prompt scored by judgment.
- **Not** permanent — it's a snapshot to be re-run as models and needs change.

## Guardrails to ship alongside

Picking B doesn't end the risk:

1. **Keep a human in the loop** — agents edit before sending.
2. **Grounding/eligibility regression prompts** — the "20-day refund" test goes into a standing set.
3. **Monitor for drift** — re-score periodically; model updates can change behavior.
4. **Cost + latency budgets** — track p95 latency and $/draft against thresholds.

## What would change the recommendation

- Model A ships a **grounded/policy-constrained mode** that passes the veto dimensions.
- A new model **beats B on grounding + safety** at acceptable cost/latency → re-evaluate.
- The **use case changes** (e.g., creative marketing copy) → re-weight the rubric; A may then win.
- **Cost or latency** of B breaches budget at production volume → revisit the trade-off.

## The PM takeaway

The comparison didn't just pick a model — it **defined what "unacceptable" means for this product** (ungrounded, off-policy drafts) and produced a re-runnable test for it. That definition, more than this one verdict, is the durable output of the evaluation.

> Honesty note: this is a learning/portfolio exercise. The recommendation demonstrates *how* an AI PM reasons from comparison to decision — it is not a production endorsement of any model.
