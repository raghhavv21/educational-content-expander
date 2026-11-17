[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/f7mj2mtrx2/badge.svg)](https://msrit.ask.thili.ai/b/f7mj2mtrx2)

# Educational Content Expander

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/f7mj2mtrx2/badge.svg)](https://ask.thili.ai/b/f7mj2mtrx2)

This CLI tool assists educators in transforming brief notes or bullet points into complete, well-formed sentences for lesson plans and teaching materials. By leveraging next-word prediction trained on educational texts, it provides contextually relevant expansions to streamline content creation. It's designed to be a helpful utility for quickly fleshing out ideas.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps educators expand short phrases into coherent sentences by generating next-word suggestions based on a corpus of educational texts.

**Done when.** The tool successfully takes short text inputs and generates contextually relevant and grammatically plausible expansions using next-word prediction, demonstrating the core functionality of `lfllm-lfnextword`.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Gather a corpus of educational texts relevant to a subject.**  
      Hint: Use file I/O to load and preprocess your chosen educational texts.
- [x] **2. Build the n-gram model, including add-k smoothing.**  
      Hint: Apply the n-gram counting and smoothing techniques from the 'Build a Next-Word Predictor from Scratch' drill.
- [x] **3. Implement text generation to expand short inputs.**  
      Hint: Utilize the next-word prediction logic to iteratively generate words based on the input and previous predictions.
- [x] **4. Develop a simple CLI for text input and expansion output.**  
      Hint: Use basic input/output functions to create a user-friendly command-line interface.

## Run it

You need Python 3.12 and git.


## Usage

```bash
# Expand a short educational phrase
python -m app.cli expand "photosynthesis is" --n 3 --k 0.5 --min-len 4 --max-len 18

# View top-k next-word suggestions
python -m app.cli suggest "newton's first law" --topk 5

# Interactive step-by-step sentence builder
python -m app.cli interactive "cellular respiration"

# Evaluate held-out perplexity on an 80/20 corpus split
python -m app.cli evaluate --n 2 --k 0.5

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Offer multiple next-word suggestions at each step, allowing the user to choose the best fit.
- Implement a simple interactive mode in the CLI where the user can accept or reject generated words step-by-step.
- Allow the user to specify a minimum or maximum length for the generated expansion.
