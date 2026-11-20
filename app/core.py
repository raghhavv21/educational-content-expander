"""Core n-gram language model and educational text expansion logic."""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CORPUS_PATH = Path("data/corpus.txt")

FALLBACK_CORPUS = """
Photosynthesis is the biological process by which green plants convert light energy into chemical energy.
Cellular respiration breaks down glucose molecules to release usable energy in the form of adenosine triphosphate.
Mitosis is a type of cell division that produces two genetically identical daughter cells for growth and repair.
Natural selection favors individuals with advantageous traits that improve survival and reproductive success.
Newton's first law of motion states that an object remains at rest or in uniform motion unless acted upon by an external force.
The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another.
Scientific inquiry relies on testable hypotheses, controlled experiments, empirical measurement, and peer review.
""".strip()


def load_corpus(path: str | Path | None = None) -> str:
    """Load educational corpus text from disk, falling back to built-in educational sentences."""
    target = Path(path) if path else DEFAULT_CORPUS_PATH
    if target.exists():
        return target.read_text(encoding="utf-8")
    return FALLBACK_CORPUS


def to_sentences(text: str) -> list[str]:
    """Split raw text into cleaned sentences."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def to_tokens(sentence: str) -> list[str]:
    """Lowercase and tokenize a sentence into word tokens."""
    return re.findall(r"[a-z']+", sentence.lower())


def build(sents: list[str], n: int) -> defaultdict[tuple[str, ...], Counter[str]]:
    """Build n-gram continuation counts: context (n-1 words) -> Counter of next words."""
    if n < 2:
        raise ValueError("n must be at least 2")
    foll: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for s in sents:
        tokens = to_tokens(s)
        if not tokens:
            continue
        toks = ["<s>"] * (n - 1) + tokens + ["</s>"]
        for i in range(len(toks) - n + 1):
            ctx = tuple(toks[i : i + n - 1])
            nxt = toks[i + n - 1]
            foll[ctx][nxt] += 1
    return foll


class NgramLM:
    """A smoothed n-gram language model with add-k smoothing, suggestion, expansion, and perplexity."""

    def __init__(self, n: int = 3, k: float = 0.5):
        if n < 2:
            raise ValueError("n must be at least 2")
        if k <= 0:
            raise ValueError("k must be positive for add-k smoothing")
        self.n = n
        self.k = k
        self.sents: list[str] = []
        self.foll: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.V: set[str] = {"</s>"}

    def fit(self, text: str) -> NgramLM:
        """Fit n-gram counts and vocabulary from raw corpus text."""
        self.sents = to_sentences(text)
        self.foll = build(self.sents, self.n)
        self.V = {t for s in self.sents for t in to_tokens(s)} | {"</s>"}
        return self

    def prob(self, ctx: tuple[str, ...] | str, word: str) -> float:
        """Compute add-k smoothed probability P(word | ctx)."""
        if isinstance(ctx, str):
            ctx_tuple = tuple((["<s>"] * (self.n - 1) + to_tokens(ctx))[-(self.n - 1) :])
        else:
            ctx_tuple = tuple((("<s>",) * (self.n - 1) + ctx)[-(self.n - 1) :])
        c = self.foll.get(ctx_tuple, Counter())
        total = sum(c.values())
        return (c[word.lower()] + self.k) / (total + self.k * len(self.V))

    def _context_from_tokens(self, tokens: list[str]) -> tuple[str, ...]:
        padded = ["<s>"] * (self.n - 1) + tokens
        return tuple(padded[-(self.n - 1) :])

    def suggest_next(self, text: str, topk: int = 3) -> list[tuple[str, float]]:
        """Return top-k next-word suggestions with smoothed probabilities."""
        tokens = to_tokens(text)
        ctx = self._context_from_tokens(tokens)
        c = self.foll.get(ctx, Counter())

        # Fall back to shorter context if exact n-gram context is unseen
        if not c and len(tokens) >= 1 and self.n > 2:
            fallback_ctx = (tokens[-1],)
            candidates = Counter()
            for k_ctx, counts in self.foll.items():
                if k_ctx[-1:] == fallback_ctx:
                    candidates.update(counts)
            c = candidates

        ranked = [
            (w, round(self.prob(ctx, w), 4))
            for w, _ in c.most_common()
            if w != "</s>"
        ]
        if ranked:
            return ranked[:topk]

        # Global unigram fallback from vocabulary if context is completely novel
        unigram_counts: Counter[str] = Counter()
        for counts in self.foll.values():
            unigram_counts.update(counts)
        fallback = [
            (w, round(self.prob(ctx, w), 4))
            for w, _ in unigram_counts.most_common()
            if w != "</s>"
        ]
        return fallback[:topk]

    def expand(
        self,
        prefix: str,
        seed: int = 0,
        min_len: int = 3,
        max_len: int = 20,
    ) -> str:
        """Expand a short educator phrase into a coherent sentence."""
        rng = random.Random(seed)
        seed_tokens = to_tokens(prefix)
        generated: list[str] = list(seed_tokens)
        added = 0

        for _ in range(max_len):
            ctx = self._context_from_tokens(generated)
            c = self.foll.get(ctx)

            # Backoff to last token if exact context is unseen
            if not c and generated and self.n > 2:
                backed_off: Counter[str] = Counter()
                for k_ctx, counts in self.foll.items():
                    if k_ctx[-1] == generated[-1]:
                        backed_off.update(counts)
                c = backed_off

            if not c:
                break

            items = list(c.items())
            if added < min_len:
                non_end = [(w, cnt) for w, cnt in items if w != "</s>"]
                if non_end:
                    items = non_end

            words, weights = zip(*items)
            nxt = rng.choices(words, weights=weights)[0]
            if nxt == "</s>":
                break
            generated.append(nxt)
            added += 1

        sentence = " ".join(generated).strip()
        if not sentence:
            return ""
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    def perplexity(self, text: str) -> float:
        """Compute cross-entropy perplexity on held-out text."""
        log_sum = 0.0
        total_steps = 0
        for s in to_sentences(text):
            tokens = to_tokens(s)
            if not tokens:
                continue
            toks = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(len(toks) - self.n + 1):
                ctx = tuple(toks[i : i + self.n - 1])
                w = toks[i + self.n - 1]
                log_sum += math.log(self.prob(ctx, w))
                total_steps += 1
        if total_steps == 0:
            raise ValueError("Cannot compute perplexity on empty text.")
        return math.exp(-log_sum / total_steps)


# --- Milestone 1: Gather a corpus of educational texts relevant to a subject.
def milestone_1(corpus_path: str | Path | None = None) -> dict:
    """Load and preprocess the educational corpus, returning summary statistics."""
    raw = load_corpus(corpus_path)
    sentences = to_sentences(raw)
    tokens = [t for s in sentences for t in to_tokens(s)]
    return {
        "sentences": sentences,
        "tokens": tokens,
        "vocabulary": sorted(set(tokens)),
    }


# --- Milestone 2: Build the n-gram model, including add-k smoothing.
def milestone_2(corpus_path: str | Path | None = None, n: int = 3, k: float = 0.5) -> NgramLM:
    """Fit and return a smoothed n-gram language model on the educational corpus."""
    raw = load_corpus(corpus_path)
    return NgramLM(n=n, k=k).fit(raw)


# --- Milestone 3: Implement text generation to expand short inputs.
def milestone_3(
    phrase: str,
    corpus_path: str | Path | None = None,
    n: int = 3,
    k: float = 0.5,
    seed: int = 0,
    min_len: int = 3,
    max_len: int = 20,
) -> str:
    """Expand a short input phrase into a complete educational sentence."""
    model = milestone_2(corpus_path=corpus_path, n=n, k=k)
    return model.expand(phrase, seed=seed, min_len=min_len, max_len=max_len)


# --- Milestone 4: Develop a simple CLI for text input and expansion output.
def milestone_4(argv: list[str] | None = None) -> None:
    """Invoke the CLI entry point."""
    from app.cli import main

    main(argv)


class UnigramLM:
    """A simple unigram language model (zero-context baseline).

    Predicts next words purely based on independent token frequencies with add-k smoothing,
    serving as a baseline to demonstrate the performance gain of contextual n-gram models.
    """

    def __init__(self, k: float = 0.5):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.counts: Counter[str] = Counter()
        self.total_tokens: int = 0
        self.V: set[str] = {"</s>"}

    def fit(self, text: str) -> UnigramLM:
        self.counts = Counter()
        sents = to_sentences(text)
        for s in sents:
            toks = to_tokens(s) + ["</s>"]
            self.counts.update(toks)
        self.total_tokens = sum(self.counts.values())
        self.V = set(self.counts.keys())
        return self

    def prob(self, word: str) -> float:
        w = word.lower()
        return (self.counts[w] + self.k) / (self.total_tokens + self.k * len(self.V))

    def expand(self, prefix: str, seed: int = 0, min_len: int = 3, max_len: int = 20) -> str:
        rng = random.Random(seed)
        generated = list(to_tokens(prefix))
        words, weights = zip(*self.counts.items())
        added = 0
        for _ in range(max_len):
            nxt = rng.choices(words, weights=weights)[0]
            if nxt == "</s>":
                if added >= min_len:
                    break
                continue
            generated.append(nxt)
            added += 1
        sent = " ".join(generated).strip()
        if not sent:
            return ""
        sent = sent[0].upper() + sent[1:]
        if not sent.endswith((".", "!", "?")):
            sent += "."
        return sent

    def perplexity(self, text: str) -> float:
        log_sum = 0.0
        total_steps = 0
        for s in to_sentences(text):
            toks = to_tokens(s) + ["</s>"]
            for w in toks:
                log_sum += math.log(self.prob(w))
                total_steps += 1
        if total_steps == 0:
            raise ValueError("Cannot compute perplexity on empty text.")
        return math.exp(-log_sum / total_steps)


def compare_models(
    corpus_path: str | Path | None = None,
    test_ratio: float = 0.2,
    seed: int = 0,
    sample_phrase: str = "photosynthesis is",
) -> dict:
    """Compare the smoothed N-gram model against the simple Unigram baseline."""
    raw = load_corpus(corpus_path)
    sents = to_sentences(raw)
    rng = random.Random(seed)
    shuffled = list(sents)
    rng.shuffle(shuffled)
    cut = max(1, int((1.0 - test_ratio) * len(shuffled)))
    train_text = ". ".join(shuffled[:cut]) + "."
    test_text = ". ".join(shuffled[cut:]) + "."

    # 1. Simple Baseline: Unigram (zero-context)
    unigram = UnigramLM(k=0.5).fit(train_text)

    # 2. Bigram Model (n=2, k=0.5)
    bigram = NgramLM(n=2, k=0.5).fit(train_text)

    # 3. Trigram Model (n=3, k=0.5)
    trigram = NgramLM(n=3, k=0.5).fit(train_text)

    return {
        "dataset": {
            "total_sentences": len(sents),
            "train_sentences": cut,
            "test_sentences": len(shuffled) - cut,
            "vocabulary_size": len(set(t for s in sents for t in to_tokens(s))),
        },
        "models": {
            "unigram_baseline": {
                "name": "Unigram (Zero-Context Baseline)",
                "n": 1,
                "train_ppl": round(unigram.perplexity(train_text), 2),
                "test_ppl": round(unigram.perplexity(test_text), 2),
                "sample_expansion": unigram.expand(sample_phrase, seed=seed, min_len=4, max_len=12),
            },
            "bigram": {
                "name": "Bigram Model (n=2, add-0.5)",
                "n": 2,
                "train_ppl": round(bigram.perplexity(train_text), 2),
                "test_ppl": round(bigram.perplexity(test_text), 2),
                "sample_expansion": bigram.expand(sample_phrase, seed=seed, min_len=4, max_len=12),
            },
            "trigram": {
                "name": "Trigram Model (n=3, add-0.5)",
                "n": 3,
                "train_ppl": round(trigram.perplexity(train_text), 2),
                "test_ppl": round(trigram.perplexity(test_text), 2),
                "sample_expansion": trigram.expand(sample_phrase, seed=seed, min_len=4, max_len=12),
            },
        },
    }