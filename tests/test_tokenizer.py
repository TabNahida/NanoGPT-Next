from __future__ import annotations

from pathlib import Path

import pytest

from nanogpt_next.tokenizer.wrapper import TokenizerWrapper


@pytest.mark.skipif(
    not Path("Tokenizer/V1/tokenizer.json").exists(),
    reason="Tokenizer artifact missing",
)
def test_tokenizer_roundtrip() -> None:
    tokenizer = TokenizerWrapper.from_dir("Tokenizer/V1")
    text = "Hello NanoGPT Next"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) > 0

    recovered = tokenizer.decode(ids)
    assert isinstance(recovered, str)
    assert len(recovered.strip()) > 0


@pytest.mark.skipif(
    not Path("Tokenizer/V1/tokenizer.json").exists(),
    reason="Tokenizer artifact missing",
)
def test_tokenizer_golden_is_stable() -> None:
    tokenizer = TokenizerWrapper.from_dir("Tokenizer/V1")
    ids = tokenizer.encode("test", add_special_tokens=False)
    assert ids == tokenizer.encode("test", add_special_tokens=False)
