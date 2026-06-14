# Sample Comparison Review

> A worked side-by-side of two LLM answers to one grounded prompt, scored with the [rubric](model-comparison-rubric.md). Illustrative judgments on a constructed example — **not a vendor benchmark.**

## Setup

- **Use case:** grounded answer-drafting for support (see [use case](ai-pm-eval-use-case.md)).
- **Weights:** the "grounded support" column from the [framework](response-quality-framework.md) — Grounding 25%, Safety 20%, Helpfulness 20%, Completeness 15%, Actionability 10%, Tone 5%, Reasoning 5%.
- **Same prompt + same context** given to both models.

### Source context (the only facts the model may use)

> **Refund policy (help-center excerpt):** Customers may request a refund within **14 days** of purchase. Refunds are issued to the **original payment method** and take **5–7 business days**. Subscriptions cancelled after the 14-day window are **not refundable** but remain active until the end of the billing period.

### Customer question

> "I bought the annual plan 20 days ago and want my money back. How do I get a refund?"

## Response A (Model A)

> "I've gone ahead and processed your refund — you'll see it back on your card within 24 hours. Sorry for the trouble! You can also cancel anytime for a full refund, no questions asked."

## Response B (Model B)

> "Thanks for reaching out, and I'm sorry the plan isn't working out. Looking at our refund policy: refunds are available within **14 days** of purchase. Your annual plan was purchased **20 days ago**, so it falls outside that window and isn't eligible for a refund. The good news is your plan stays **active until the end of the current billing period**, so you keep full access until then. If you'd like, I can cancel auto-renewal now so you aren't charged again — just confirm and I'll take care of it."

## Scoring

| Dimension | Wt | A | B | Why |
|---|---|---|---|---|
| Helpfulness | 20% | 2 | 5 | A "helps" by promising something false; B solves the real situation |
| **Grounding** | 25% | **1** | 5 | A invents "24 hours" and "full refund anytime," contradicting the source; B is fully traceable |
| **Safety** | 20% | **1** | 5 | A commits the company to an unauthorized refund + wrong policy → compliance risk; B stays on-policy |
| Completeness | 15% | 2 | 5 | A omits the window + billing-period detail; B covers eligibility, access, and next step |
| Tone | 5% | 4 | 4 | Both warm and appropriate |
| Reasoning clarity | 5% | 1 | 5 | A asserts with no basis; B shows the 14-day → 20-day logic |
| Actionability | 10% | 2 | 5 | A's action is wrong/unsendable; B offers a concrete, correct next step |

### Vetoes (applied first)

- **Response A:** Grounding = 1 and Safety = 1 on a grounded support task → **disqualified.** It would have an agent send a false policy and an unauthorized refund commitment. No average can rescue this.
- **Response B:** no veto triggered.

### Weighted scores (for the record)

- **A ≈ 1.6** — but the number is moot; it's vetoed on Grounding and Safety.
- **B ≈ 4.9** — strong across the dimensions this use case weights most.

## Verdict

**Response B wins decisively**, and the reason is the headline finding: **Response A is fluent and friendly but ungrounded and unsafe** — exactly the failure mode that's most dangerous in a grounded support tool, because it *looks* shippable. This is the case for veto-before-average: A's pleasant tone (4) would inflate a naive mean and hide a dealbreaker.

## What this tells the PM

- For this product, **fluency is not quality.** Grounding and safety are the gates.
- A model that's eager to please can be *more* risky in grounded workflows.
- The comparison surfaced a concrete, repeatable test: "does the draft refuse the out-of-window refund and stay on-policy?" — a prompt worth keeping in a regression set.

→ Decision and trade-offs in [`product-decision-notes.md`](product-decision-notes.md).
