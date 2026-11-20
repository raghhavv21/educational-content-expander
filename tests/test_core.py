"""Tests for Milestones 1–4 of Educational Content Expander."""
from app.core import (
    NgramLM,
    UnigramLM,
    compare_models,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
)


def test_milestone_1(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Photosynthesis converts sunlight into chemical energy. "
        "Plants absorb carbon dioxide and water.",
        encoding="utf-8",
    )
    data = milestone_1(sample)
    assert len(data["sentences"]) == 2
    assert "photosynthesis" in data["vocabulary"]
    assert len(data["tokens"]) == 12


def test_milestone_2(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Energy flows through ecosystems. "
        "Energy transforms from potential energy into kinetic energy.",
        encoding="utf-8",
    )
    model = milestone_2(corpus_path=sample, n=2, k=0.5)
    assert isinstance(model, NgramLM)
    # Unseen pair must still receive strictly positive probability under add-k smoothing
    unseen_prob = model.prob("ecosystems", "quantum")
    seen_prob = model.prob("kinetic", "energy")
    assert unseen_prob > 0.0
    assert seen_prob > unseen_prob
    assert model.perplexity("Energy flows through ecosystems.") > 1.0


def test_milestone_3(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Cellular respiration breaks down glucose molecules to release usable energy. "
        "Cellular respiration powers essential metabolic functions inside living cells.",
        encoding="utf-8",
    )
    expanded = milestone_3(
        "cellular respiration",
        corpus_path=sample,
        n=3,
        k=0.5,
        seed=1,
        min_len=3,
        max_len=12,
    )
    assert expanded.startswith("Cellular respiration")
    assert expanded.endswith(".")
    assert len(expanded.split()) >= 5


def test_milestone_4(capsys, tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Natural selection drives evolutionary adaptation across generations.",
        encoding="utf-8",
    )
    milestone_4(["expand", "natural selection", "--corpus", str(sample), "--min-len", "2", "--max-len", "8"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("Natural selection")
    assert out.endswith(".")


def test_unigram_baseline(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Photosynthesis converts sunlight into energy. Cells metabolize nutrients.",
        encoding="utf-8",
    )
    model = UnigramLM(k=0.5).fit(sample.read_text(encoding="utf-8"))
    assert model.prob("photosynthesis") > 0.0
    assert model.prob("unseen_word") > 0.0
    ppl = model.perplexity("Cells metabolize nutrients.")
    assert ppl > 1.0
    expanded = model.expand("photosynthesis", seed=1, min_len=2, max_len=6)
    assert expanded.startswith("Photosynthesis")
    assert expanded.endswith(".")


def test_compare_models_and_cli(capsys, tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Photosynthesis converts light into energy. "
        "Cells undergo respiration to produce adenosine triphosphate. "
        "Mitosis divides cells for tissue growth and repair. "
        "DNA stores biological inheritance across generations.",
        encoding="utf-8",
    )
    result = compare_models(corpus_path=sample, test_ratio=0.25, seed=42)
    assert "models" in result
    assert "unigram_baseline" in result["models"]
    assert "bigram" in result["models"]
    assert "trigram" in result["models"]

    # Verify CLI evaluate with --compare-baseline
    milestone_4(["evaluate", "--corpus", str(sample), "--compare-baseline"])
    out = capsys.readouterr().out
    assert "BASELINE COMPARISON & MODEL EVALUATION REPORT" in out
    assert "Unigram (Zero-Context Baseline)" in out
    assert "Trigram Model" in out


def test_cli_hello(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out