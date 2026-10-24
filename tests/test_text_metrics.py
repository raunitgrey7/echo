"""Normalisation, number folding, WER/CER alignment and pooled rates."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from echo.metrics.text_norm import fold_numbers, normalize_text, ordinal_suffix, tokens
from echo.metrics.wer import (
    EditCounts,
    align_counts,
    char_error_rate,
    corpus_counts,
    word_counts,
    word_error_rate,
)


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("forty submissions and question three", "40 submissions and question 3"),
        ("one hundred and thirty five students", "135 students"),
        ("from twelve percent to nine percent", "from 12% to 9%"),
        ("the twelfth of November", "the 12th of November"),
        ("Louis the sixteenth", "Louis the 16th"),
        ("lasted forty three minutes", "lasted 43 minutes"),
        ("two thousand and five", "2005"),
        ("twenty one and twenty two", "21 and 22"),
        ("one and two", "1 and 2"),
        ("the third law", "the 3rd law"),
    ],
)
def test_numbers_fold_to_the_same_form(spoken, written):
    assert normalize_text(spoken) == normalize_text(written)


def test_normalize_keeps_meaning_bearing_pieces():
    assert (
        normalize_text("Newton's laws: 1,789 by-product, don't “quote”") == "newton's laws 1789 by product don't quote"
    )
    assert normalize_text("bread and butter") == "bread and butter"
    assert normalize_text("thousands of people") == "thousands of people"
    assert tokens("") == []
    assert tokens("  ...  ") == []


def test_ordinal_suffixes():
    assert [ordinal_suffix(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 111)] == [
        "st",
        "nd",
        "rd",
        "th",
        "th",
        "th",
        "th",
        "st",
        "nd",
        "rd",
        "st",
        "th",
    ]


def test_fold_numbers_leaves_plain_words():
    assert fold_numbers(["no", "numbers", "here"]) == ["no", "numbers", "here"]
    assert fold_numbers(["hundred", "people"]) == ["hundred", "people"]


def test_alignment_counts_each_operation():
    counts = align_counts("a b c d".split(), "a x c d e".split())
    assert counts == EditCounts(substitutions=1, deletions=0, insertions=1, reference_length=4)
    assert counts.errors == 2
    assert counts.rate == pytest.approx(0.5)
    deletion = align_counts("a b c".split(), "a c".split())
    assert (deletion.substitutions, deletion.deletions, deletion.insertions) == (0, 1, 0)


def test_empty_reference_and_hypothesis():
    assert align_counts([], []).rate == 0.0
    assert align_counts([], ["x", "y"]).rate == 2.0
    assert align_counts(["x"], []).rate == 1.0


def test_word_and_char_error_rates():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("The cat sat.", "the cat sat") == 0.0
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert char_error_rate("abc", "abd") == pytest.approx(1 / 3)


def test_corpus_rate_pools_over_reference_length():
    pairs = [("red green blue black", "red green blue black"), ("pink grey", "pink gold")]
    pooled = corpus_counts(pairs)
    assert pooled.reference_length == 6
    assert pooled.rate == pytest.approx(1 / 6)
    assert corpus_counts(pairs, level="char").reference_length == len("red green blue black") + len("pink grey")


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.sampled_from(["a", "b", "c", "d"]), max_size=12),
    st.lists(st.sampled_from(["a", "b", "c", "d"]), max_size=12),
)
def test_alignment_is_a_valid_edit_distance(ref, hyp):
    counts = align_counts(ref, hyp)
    assert counts.errors >= abs(len(ref) - len(hyp))
    assert counts.errors <= max(len(ref), len(hyp))
    assert counts.reference_length - counts.deletions - counts.substitutions >= 0
    assert len(hyp) == len(ref) - counts.deletions + counts.insertions


@settings(max_examples=25, deadline=None)
@given(st.text(alphabet="abcdefghij ", min_size=0, max_size=40))
def test_identical_texts_have_zero_error(text):
    assert word_error_rate(text, text) == 0.0
    assert word_counts(text, text).errors == 0
