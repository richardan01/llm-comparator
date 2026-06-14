# Response Quality Framework

> How the [rubric](model-comparison-rubric.md) becomes a repeatable process: weight dimensions by use case, aggregate fairly, and weigh the trade-offs that scores alone don't capture.

## The process

```
1. Fix the use case          → what is "good" here?
2. Weight the 7 dimensions   → not all matter equally
3. Score both responses      → 1–5 per dimension (rubric)
4. Apply vetoes              → any dealbreaker kills the candidate
5. Aggregate                 → weighted score, with rationale
6. Add non-quality factors   → cost, latency, reliability
7. Decide + record           → recommendation + trade-offs
```

## Step 1–2: Weight by use case

Quality is contextual. The **same rubric, different weights** for different products. Weights should sum to 100%.

| Dimension | Grounded support drafts | Brainstorming assistant | Code helper |
|---|---|---|---|
| Helpfulness | 20% | 25% | 25% |
| **Grounding** | **25%** | 5% | 15% |
| **Safety** | **20%** | 15% | 10% |
| Completeness | 15% | 15% | 20% |
| Tone | 5% | 15% | 5% |
| Reasoning clarity | 5% | 10% | 15% |
| Actionability | 10% | 15% | 10% |

> The point isn't precision in the percentages — it's forcing an explicit conversation about *what this product can't afford to get wrong.* For grounded support, that's grounding + safety.

## Step 4: Vetoes before averages

A weighted average can launder a fatal flaw into a decent number. So apply **hard vetoes first**:

- **Safety ≤ 2** → disqualified, regardless of other scores.
- **Grounding ≤ 2** on a grounded use case → disqualified.

Only surviving candidates get aggregated.

## Step 5: Aggregate

`Weighted score = Σ (dimension score × weight)`

Report it **with the per-dimension breakdown**, never the single number alone. A 4.1 built on a Grounding 2 tells a different story than a flat 4.1.

## Step 6: Non-quality factors

Two responses can tie on quality and still not be equal choices:

| Factor | Why it matters | Typical signal |
|---|---|---|
| **Cost / token** | Drives unit economics at scale | $ per 1K tokens |
| **Latency** | Affects whether agents actually wait for it | p50 / p95 ms |
| **Consistency** | One great answer ≠ reliably great | variance across many prompts |
| **Context limits** | Can it hold the docs it must ground on? | context window |
| **Operational fit** | Region, data handling, rate limits | meets policy y/n |

## The trade-offs that matter

Most real model decisions come down to balancing these tensions:

- **Quality ↔ Cost** — is the quality lift worth the per-call price at our volume?
- **Quality ↔ Latency** — a better answer the user won't wait for is a worse product.
- **Completeness ↔ Conciseness** — more coverage can mean more to read/edit.
- **Capability ↔ Safety** — a more eager model may also be more willing to go off-policy.
- **Peak ↔ Consistency** — pick the model that's reliably good over the one that's occasionally brilliant.

A PM's job is to name *which* trade-off this product is making, and why it's acceptable.

## Why repeatable matters

When a new model ships, you don't re-litigate "what is good." You **re-run the same prompts, re-score, re-aggregate.** The framework turns model selection from a one-off debate into a standing evaluation you can trust over time.

## Honesty note

This framework formalizes judgment; it doesn't remove it. Scores are subjective, weights are choices, and a single grounded prompt is a sample of one. Treat outputs as **structured evidence for a decision**, not proof.
