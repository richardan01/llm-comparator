# Eval Methodology: Hamel & Shreya's Framework Applied

> This doc maps Hamel Husain and Shreya Shankar's LLM evaluation methodology to this repo's comparison workflow. Their approach is the conceptual backbone behind *how* the comparison and rubric docs were designed.
>
> **Attribution:** The methodology described here comes from Hamel Husain and Shreya Shankar's course and writing — see [hamel.dev/notes/llm/evals](https://hamel.dev/notes/llm/evals/) and their co-authored FAQ "LLM Evals: Everything You Need to Know" (Jan 2025). I'm applying their framework, not inventing it.

## The core lifecycle: Analyze → Measure → Improve

H&S's framework is a loop, not a checklist:

```
Analyze (error analysis)
    ↓
Measure (targeted criteria, binary pass/fail)
    ↓
Improve (prompt, data, model)
    ↓
Analyze again (new traces, new failure patterns)
```

Most teams skip straight to "Measure" with generic metrics. The result is evaluations that look good but don't catch what actually breaks in production. **Error analysis is what makes measurement meaningful.**

---

## Step 1: Analyze — error analysis before any rubric

Before scoring anything, spend time *looking at data*:

1. **Collect 50–100+ representative traces** — real prompts and responses from your target workflow.
2. **Open coding** — a domain expert reviews each trace and writes open-ended notes. No pre-set categories. Just: "what's wrong here? what surprised me?"
3. **Axial coding** — group the notes into a failure taxonomy. Count how often each failure type appears.
4. **Iterate until saturation** — keep going until new traces stop revealing new failure types.

> H&S: "Error analysis is the most important activity in evals and helps you decide what evals to write in the first place."

The **Benevolent Dictator** pattern: appoint one domain expert as the quality decision-maker — not a committee. A PM who knows the users deeply is a strong fit. Consensus labeling creates inconsistency; one trusted voice drives clarity.

### How this connects to this repo

The `sample-comparison-review.md` scenario (grounded refund-policy question) is a single constructed example. In a real eval, you'd run this process on 50–100 real support queries to *discover* what fails — before deciding whether "grounding" or "safety" or something else is the right thing to measure. The rubric comes *after* the failure taxonomy, not before.

---

## Step 2: Measure — binary criteria, not generic scales

Once you have a failure taxonomy, translate it into **binary pass/fail criteria specific to your failures**, not generic 1–5 ratings.

**Why binary over Likert scales:**
- "What's the real difference between 3 and 4?" — annotators hedge into middle values
- Binary forces a clearer question: "Is this response acceptable for our users, yes or no?"
- Faster to annotate, easier to reach consistency, simpler to aggregate

**Where this repo's 7-dimension rubric fits:**
The 7 dimensions (helpfulness, grounding, safety, completeness, tone, reasoning clarity, actionability) are useful as **calibration and exploration tools** — they help a team develop shared intuitions and spot which dimensions matter for a use case. They are *not* production eval criteria.

In practice, error analysis on your specific traces would likely surface 2–4 domain-specific failure modes (e.g. "cites non-existent policy," "correct policy but doesn't name the user's next step"). Those become your binary criteria. The generic dimensions fade into the background.

**From rubric to binary criteria — an example:**

| Generic rubric dimension | Domain-specific binary criterion (after error analysis) |
|---|---|
| Grounding | Pass: every factual claim traces to the provided help-center doc. Fail: any invented policy or unsupported number. |
| Actionability | Pass: draft includes a concrete next step the agent can send. Fail: vague or no next step. |
| Safety | Pass: does not commit to a refund outside the 14-day window. Fail: makes an unauthorized commitment. |

→ See [`model-comparison-rubric.md`](model-comparison-rubric.md) for the full 7-dimension rubric and its limitations.

---

## Step 3: Improve — prompt, data, model

Scores are only useful if they drive change:

- **New failure type found** → update prompt system instructions, re-run eval
- **Failure persists across prompt changes** → is it a model limitation? → run model comparison
- **A comparison surfaces a winner** → document the trade-offs, ship, monitor

This is where the comparison workflow in this repo lives: **model comparison is a tool for the Improve step**, used when prompt changes don't resolve a failure pattern.

---

## When to automate

H&S are explicit: **automate sparingly, and in the right order.**

```
1. Simple assertions / regex / schema checks  (cheapest — start here)
2. Reference-based checks (compare to a known-good answer)
3. LLM-as-a-judge  (expensive — only for subjective, persistent failures)
```

Don't build an LLM judge until you've: run error analysis, confirmed the failure type is real and frequent, and verified a simple rule can't catch it. "Eval-driven development" (building metrics before understanding failures) creates wasted effort and metrics that look good but miss what matters.

### CI/CD vs. production monitoring

| Context | Approach |
|---|---|
| **CI/CD** | Small curated dataset (100+ examples). Fast assertions, binary criteria. Gates deployment. |
| **Production** | Sample live traffic asynchronously. LLM judges acceptable (no ground truth). Track confidence intervals. |

New production failures feed back into CI/CD: when monitoring surfaces a new failure pattern, add it to the curated test set to prevent regression.

---

## Summary: what this means for using this repo

| This repo provides | Its role in the H&S lifecycle |
|---|---|
| 7-dimension rubric | Calibration / exploration (not primary eval criteria) |
| Comparison workflow | Improve step: choose between models once failures are understood |
| Sample review | Illustrates the workflow on one constructed case |
| This doc | Maps the methodology so you know what's missing (error analysis, failure taxonomy) |

What this repo **doesn't provide** (and can't substitute for):
- Your real production traces
- Your domain-specific failure taxonomy
- Binary criteria derived from your actual users' failure modes

That work has to happen in your product context. This repo shows *how to think about* that process.
