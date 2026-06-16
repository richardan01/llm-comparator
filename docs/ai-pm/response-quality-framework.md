# Response Quality Framework

> How the [rubric](model-comparison-rubric.md) becomes a repeatable process: weight dimensions by use case, aggregate fairly, and weigh the trade-offs that scores alone don't capture. Grounded in [Hamel & Shreya's Analyze–Measure–Improve lifecycle](eval-methodology.md).

## The process

```
0. Error analysis            → review 50–100 traces; build failure taxonomy  ← H&S foundation
1. Fix the use case          → what is "good" here?
2. Weight the 7 dimensions   → not all matter equally
3. Score both responses      → 1–5 per dimension (rubric, for exploration)
4. Apply vetoes              → any dealbreaker kills the candidate
5. Aggregate                 → weighted score, with rationale
6. Add non-quality factors   → cost, latency, reliability
7. Decide + record           → recommendation + trade-offs
```

> **Step 0 is the most skipped and the most important.** Hamel & Shreya: "Error analysis is what makes measurement meaningful." Running the rubric without it means you're measuring the wrong things. See [`eval-methodology.md`](eval-methodology.md) for the full error analysis process.

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

## Benchmark vs evaluation

This framework is deliberately different from running a benchmark. Both have a place — they answer different questions.

| | Public benchmark | This framework |
|---|---|---|
| **Question** | "How capable is the model?" | "Is this response right for our workflow?" |
| **Prompts** | Generic, shared, often academic | The product's real prompts + context |
| **Unit** | Model | Response (and model, by aggregation) |
| **Output** | One score / rank | Per-dimension scores + trade-offs + decision |
| **Use it to** | Shortlist candidates | Decide what ships |

Treat benchmarks as the **filter**; treat this framework as the **fit test**. See [`ai-pm-eval-use-case.md`](ai-pm-eval-use-case.md) for the longer contrast.

## When to automate

H&S are explicit: automate sparingly, in cost order:

```
1. Simple assertions / regex / schema checks  ← start here
2. Reference-based checks (compare to known-good answer)
3. LLM-as-a-judge                             ← only for subjective, persistent failures
```

Don't build an LLM judge until error analysis confirms the failure is real, frequent, and can't be caught by a rule. For the CI/CD vs. production distinction:

| Context | Approach |
|---|---|
| **CI/CD** | Small curated set (~100 examples). Fast binary assertions. Gates deployment. |
| **Production** | Sample live traffic. LLM judges acceptable (no ground truth). Track confidence intervals. |

When production monitoring finds a new failure pattern, add it to the CI/CD set to prevent regression.

## Why repeatable matters

When a new model ships, you don't re-litigate "what is good." You **re-run the same prompts, re-score, re-aggregate.** The framework turns model selection from a one-off debate into a standing evaluation you can trust over time.

## Honesty note

This framework formalizes judgment; it doesn't remove it. Scores are subjective, weights are choices, and a single grounded prompt is a sample of one. Treat outputs as **structured evidence for a decision**, not proof.
