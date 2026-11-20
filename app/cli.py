"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import random

from app.core import (
    NgramLM,
    compare_models,
    load_corpus,
    milestone_2,
    milestone_3,
    to_sentences,
    to_tokens,
)


def _run_interactive(model: NgramLM, phrase: str, topk: int = 3) -> str:
    """Interactive step-by-step word acceptance/rejection loop."""
    tokens = to_tokens(phrase)
    print(f"\nStarting phrase: {' '.join(tokens)}")
    print("Commands: [1-9] pick suggestion | [c] type custom word | [d] done\n")

    while True:
        current = " ".join(tokens)
        suggestions = model.suggest_next(current, topk=topk)
        print(f"Current: {current}")
        for idx, (word, prob) in enumerate(suggestions, start=1):
            print(f"  {idx}. {word} (p={prob:.4f})")

        choice = input("Choose next word (number, 'c' for custom, 'd' to finish): ").strip().lower()
        if choice in ("d", "done", "q", "quit", ""):
            break
        if choice == "c":
            custom = input("Enter word: ").strip().lower()
            if custom:
                tokens.extend(to_tokens(custom))
        elif choice.isdigit() and 1 <= int(choice) <= len(suggestions):
            tokens.append(suggestions[int(choice) - 1][0])
        else:
            print("Invalid choice. Try again.")

    result = " ".join(tokens).strip()
    if result:
        result = result[0].upper() + result[1:]
        if not result.endswith((".", "!", "?")):
            result += "."
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="educational-content-expander",
        description="Educators use this CLI tool to expand bullet points or short phrases into coherent sentences for lesson plans by generating next-word suggestions based on educational texts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Smoke-test command
    sub.add_parser("hello", help="check that everything is wired up")

    # Expand command
    p_expand = sub.add_parser("expand", help="expand a short phrase into a complete sentence")
    p_expand.add_argument("phrase", help="short phrase or note to expand")
    p_expand.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    p_expand.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_expand.add_argument("--min-len", type=int, default=3, help="minimum number of generated words")
    p_expand.add_argument("--max-len", type=int, default=20, help="maximum number of generated words")
    p_expand.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    p_expand.add_argument("--corpus", default=None, help="path to educational corpus .txt file")

    # Suggest command
    p_suggest = sub.add_parser("suggest", help="show top-k next-word suggestions for a phrase")
    p_suggest.add_argument("phrase", help="input phrase")
    p_suggest.add_argument("--topk", type=int, default=5, help="number of suggestions to display")
    p_suggest.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    p_suggest.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_suggest.add_argument("--corpus", default=None, help="path to educational corpus .txt file")

    # Interactive mode
    p_inter = sub.add_parser("interactive", help="build a sentence step-by-step by accepting/rejecting suggestions")
    p_inter.add_argument("phrase", help="initial short phrase")
    p_inter.add_argument("--topk", type=int, default=3, help="suggestions shown per step")
    p_inter.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    p_inter.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_inter.add_argument("--corpus", default=None, help="path to educational corpus .txt file")

    # Evaluate perplexity
    p_eval = sub.add_parser("evaluate", help="measure held-out perplexity on an 80/20 corpus split")
    p_eval.add_argument("--n", type=int, default=2, help="n-gram order (default: 2)")
    p_eval.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_eval.add_argument("--corpus", default=None, help="path to educational corpus .txt file")
    p_eval.add_argument("--compare-baseline", action="store_true", help="compare N-gram models against simple unigram baseline")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Educational Content Expander is ready. Run with --help for available commands.")
    elif args.command == "expand":
        expanded = milestone_3(
            phrase=args.phrase,
            corpus_path=args.corpus,
            n=args.n,
            k=args.k,
            seed=args.seed,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        print(expanded)
    elif args.command == "suggest":
        model = milestone_2(corpus_path=args.corpus, n=args.n, k=args.k)
        suggestions = model.suggest_next(args.phrase, topk=args.topk)
        print(f"Next-word suggestions for '{args.phrase}':")
        for word, prob in suggestions:
            print(f"  - {word:<15} (p={prob:.4f})")
    elif args.command == "interactive":
        model = milestone_2(corpus_path=args.corpus, n=args.n, k=args.k)
        final_sentence = _run_interactive(model, args.phrase, topk=args.topk)
        print(f"\nFinal sentence: {final_sentence}")
    elif args.command == "evaluate":
        if args.compare_baseline:
            res = compare_models(corpus_path=args.corpus)
            ds = res["dataset"]
            print("=" * 80)
            print("BASELINE COMPARISON & MODEL EVALUATION REPORT")
            print("=" * 80)
            print(f"Corpus: {ds['total_sentences']} sentences ({ds['train_sentences']} train / {ds['test_sentences']} test split, |V|={ds['vocabulary_size']} words)\n")
            print(f"{'Model Architecture':<32} | {'Train PPL':<11} | {'Test PPL':<11} | {'Sample Output (prompt: photosynthesis is)'}")
            print("-" * 115)
            for m in res["models"].values():
                print(f"{m['name']:<32} | {m['train_ppl']:<11.2f} | {m['test_ppl']:<11.2f} | \"{m['sample_expansion']}\"")
            print("-" * 115)
            print("Key Finding: Contextual N-gram models dramatically outperform the simple unigram baseline")
            print("by leveraging local educational syntactic structure and reducing perplexity.")
        else:
            sents = to_sentences(load_corpus(args.corpus))
            rng = random.Random(0)
            rng.shuffle(sents)
            cut = max(1, int(0.8 * len(sents)))
            train_text = ". ".join(sents[:cut]) + "."
            test_text = ". ".join(sents[cut:]) + "."
            model = NgramLM(n=args.n, k=args.k).fit(train_text)
            print(f"Train perplexity    (n={args.n}, k={args.k}): {model.perplexity(train_text):.2f}")
            print(f"Held-out perplexity (n={args.n}, k={args.k}): {model.perplexity(test_text):.2f}")


if __name__ == "__main__":
    main()