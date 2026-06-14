# LLM Comparator 🔍

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Type](https://img.shields.io/badge/type-exploration%20%2F%20fork-orange)
![Built on](https://img.shields.io/badge/built%20on-PAIR--code%2Fllm--comparator-lightgrey)
![Focus](https://img.shields.io/badge/focus-AI%20PM%20evaluation-purple)
![Status](https://img.shields.io/badge/status-learning%20%2F%20portfolio-green)

> A hands-on workspace for learning how AI Product Managers compare LLM responses and turn those comparisons into product decisions.

## What this is

This repo is an **evaluation workspace built on top of [PAIR-code/llm-comparator](https://github.com/PAIR-code/llm-comparator)**. The underlying side-by-side comparison tool is Google's open-source project — I did **not** build it. What's mine is the layer on top: a set of AI-PM-oriented docs that show *how* I'd use response comparison to evaluate model quality and make a recommendation.

Think of it as a study project: a real comparison tool plus the product judgment, rubrics, and decision notes that wrap around it.

## Why it matters for AI PMs

Choosing a model — or accepting a response — isn't only a technical benchmark question. It's a product quality question:

> Which response is more useful, grounded, safe, and aligned with the user's workflow — and what are we trading away to get it?

These docs make that judgment explicit and repeatable instead of vibes-based.

## How to use it as an eval workflow

1. **Frame the decision** → [`docs/ai-pm-eval-use-case.md`](docs/ai-pm-eval-use-case.md)
2. **Score against a rubric** → [`docs/model-comparison-rubric.md`](docs/model-comparison-rubric.md)
3. **Apply the framework** (weighting, aggregation, trade-offs) → [`docs/response-quality-framework.md`](docs/response-quality-framework.md)
4. **Run a side-by-side review** → [`docs/sample-comparison-review.md`](docs/sample-comparison-review.md)
5. **Turn it into a decision** → [`docs/product-decision-notes.md`](docs/product-decision-notes.md)

The comparison tool itself renders responses side by side; these docs supply the evaluation logic a PM brings to that view.

## Quality dimensions

Every response is scored 1–5 on seven dimensions:

| Dimension | Question it answers |
|---|---|
| **Helpfulness** | Does it actually solve the user's problem? |
| **Grounding** | Is it supported by the source/context, not invented? |
| **Safety** | Does it avoid harmful, biased, or policy-violating output? |
| **Completeness** | Does it cover what the task requires, no gaps? |
| **Tone** | Is the register appropriate for the audience? |
| **Reasoning clarity** | Is the logic easy to follow and verify? |
| **Actionability** | Can the user act on it without more work? |

## Example use case

A support team is deciding which model powers an **answer-drafting assistant** that must stay grounded in help-center docs. We compare two anonymized models (Model A vs Model B) on the same grounded prompt, score both, and recommend one — with the trade-offs spelled out. Full walkthrough in [`docs/sample-comparison-review.md`](docs/sample-comparison-review.md).

## Honest attribution

- The comparison UI/tooling is **[PAIR-code/llm-comparator](https://github.com/PAIR-code/llm-comparator)** (Apache-2.0), not my work.
- Original code, license, and authorship are retained.
- My contribution is the **AI PM evaluation layer** in [`docs/`](docs/) — rubrics, frameworks, sample reviews, and decision notes.
- Any model names mentioned are illustrative; this repo contains **no benchmark results I ran against real vendor models**.

## Status

Learning / portfolio project. **Not production software** and not an endorsement of any model. The goal is to demonstrate AI PM evaluation thinking, not to ship an eval service.
