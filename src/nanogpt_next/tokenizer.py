from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as HFTokenizer


@dataclass(slots=True)
class TokenizerSpec:
    backend: str
    path: str


class TokenizerAdapter:
    def __init__(self, backend: str, tokenizer: Any) -> None:
        self.backend = backend
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        if self.backend == "hf":
            return int(self._tokenizer.get_vocab_size())
        vocab_size = getattr(self._tokenizer, "vocab_size", None)
        if callable(vocab_size):
            return int(vocab_size())
        if vocab_size is not None:
            return int(vocab_size)
        raise AttributeError("Tokenizer backend does not expose vocab size.")

    @property
    def bos_token_id(self) -> int | None:
        return self._resolve_token_id(["<s>", "[BOS]", "<bos>"])

    @property
    def eos_token_id(self) -> int | None:
        return self._resolve_token_id(["</s>", "[EOS]", "<eos>"])

    def encode(self, text: str) -> list[int]:
        if self.backend == "hf":
            return self._tokenizer.encode(text).ids
        encoded = self._tokenizer.encode(text)
        if isinstance(encoded, list):
            return [int(token) for token in encoded]
        ids = getattr(encoded, "ids", None)
        if ids is None:
            raise TypeError("TokenFlux tokenizer encode() did not return token ids.")
        return [int(token) for token in ids]

    def decode(self, token_ids: list[int]) -> str:
        if self.backend == "hf":
            return self._tokenizer.decode(token_ids)
        if hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(token_ids)
        raise AttributeError("Tokenizer backend does not expose decode().")

    def _resolve_token_id(self, candidates: list[str]) -> int | None:
        if self.backend == "hf":
            for token in candidates:
                token_id = self._tokenizer.token_to_id(token)
                if token_id is not None:
                    return int(token_id)
            return None
        token_to_id = getattr(self._tokenizer, "token_to_id", None)
        if callable(token_to_id):
            for token in candidates:
                token_id = token_to_id(token)
                if token_id is not None:
                    return int(token_id)
        return None


def _load_tokenflux(path: str | Path) -> Any:
    import TokenFlux  # type: ignore

    path_str = str(path)
    attempts = []
    tokenizer_cls = getattr(TokenFlux, "Tokenizer", None)
    if tokenizer_cls is not None:
        if hasattr(tokenizer_cls, "from_file"):
            attempts.append(lambda: tokenizer_cls.from_file(path_str))
        attempts.append(lambda: tokenizer_cls(path_str))
    for attr_name in ("load_tokenizer", "from_file", "load"):
        attr = getattr(TokenFlux, attr_name, None)
        if callable(attr):
            attempts.append(lambda attr=attr: attr(path_str))
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"Unable to initialize TokenFlux tokenizer from {path_str}") from last_error
    raise RuntimeError("TokenFlux is installed but no supported tokenizer constructor was found.")


def build_tokenizer(spec: TokenizerSpec) -> TokenizerAdapter:
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    backend = spec.backend
    if backend == "auto":
        try:
            tokenizer = _load_tokenflux(path)
            return TokenizerAdapter("tokenflux", tokenizer)
        except Exception:
            pass
        return TokenizerAdapter("hf", HFTokenizer.from_file(str(path)))
    if backend == "tokenflux":
        return TokenizerAdapter("tokenflux", _load_tokenflux(path))
    if backend == "hf":
        return TokenizerAdapter("hf", HFTokenizer.from_file(str(path)))
    raise ValueError(f"Unknown tokenizer backend: {backend}")
