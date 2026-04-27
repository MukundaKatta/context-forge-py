"""Tests for ``context_forge.estimate``."""

from context_forge import estimate_tokens


def test_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_none_is_zero():
    assert estimate_tokens(None) == 0


def test_short_text_rounds_up():
    # "abc" is 3 chars -> ceil(3/4) = 1
    assert estimate_tokens("abc") == 1


def test_exact_multiple_of_4():
    # "abcd" is 4 chars -> 1 token; "abcdefgh" is 8 -> 2 tokens
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_non_string_coerces():
    assert estimate_tokens(1234) == 1  # str(1234)="1234" -> 1
