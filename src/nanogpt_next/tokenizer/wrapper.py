from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpecialTokens:
    bos_token: str | None = "<s>"
    eos_token: str | None = "</s>"
    pad_token: str | None = "<pad>"
    unk_token: str | None = "<unk>"


class TokenizerWrapper:
    def __init__(self, tokenizer_impl, special_tokens: SpecialTokens) -> None:
        self._tok = tokenizer_impl
        self.special_tokens = special_tokens

        self.bos_id = self._token_to_id(special_tokens.bos_token)
        self.eos_id = self._token_to_id(special_tokens.eos_token)
        self.pad_id = self._token_to_id(special_tokens.pad_token)
        self.unk_id = self._token_to_id(special_tokens.unk_token)

    @classmethod
    def from_dir(
        cls,
        tokenizer_dir: str | Path,
        special_tokens: SpecialTokens | None = None,
    ) -> TokenizerWrapper:
        special_tokens = special_tokens or SpecialTokens()
        tokenizer_dir = Path(tokenizer_dir)

        # Prefer TokenFluxPlusPlus when available; fallback to tokenizers JSON.
        try:
            from tokenfluxplusplus import Tokenizer as TFTokenizer  # type: ignore

            impl = TFTokenizer.from_pretrained(str(tokenizer_dir))
            return cls(impl, special_tokens)
        except Exception:
            pass

        from tokenizers import Tokenizer

        tok_json = tokenizer_dir / "tokenizer.json"
        if not tok_json.exists():
            raise FileNotFoundError(f"Expected tokenizer JSON at: {tok_json}")
        impl = Tokenizer.from_file(str(tok_json))
        return cls(impl, special_tokens)

    @property
    def vocab_size(self) -> int:
        if hasattr(self._tok, "get_vocab_size"):
            return int(self._tok.get_vocab_size())
        if hasattr(self._tok, "vocab_size"):
            return int(self._tok.vocab_size)
        raise AttributeError("Tokenizer implementation does not expose vocab size")

    def _token_to_id(self, token: str | None) -> int | None:
        if token is None:
            return None
        if hasattr(self._tok, "token_to_id"):
            return self._tok.token_to_id(token)
        if hasattr(self._tok, "convert_tokens_to_ids"):
            return self._tok.convert_tokens_to_ids(token)
        return None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not hasattr(self._tok, "encode"):
            raise AttributeError("Tokenizer implementation does not expose encode")

        encoded = self._tok.encode(text)
        ids = list(encoded.ids) if hasattr(encoded, "ids") else list(encoded)
        if not add_special_tokens:
            return ids

        out = []
        if self.bos_id is not None:
            out.append(self.bos_id)
        out.extend(ids)
        if self.eos_id is not None:
            out.append(self.eos_id)
        return out

    def encode_batch(
        self,
        texts: Sequence[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        ids = list(token_ids)
        if skip_special_tokens:
            skip_ids = {x for x in [self.bos_id, self.eos_id, self.pad_id] if x is not None}
            ids = [x for x in ids if x not in skip_ids]
        if not hasattr(self._tok, "decode"):
            raise AttributeError("Tokenizer implementation does not expose decode")
        return self._tok.decode(ids)

    def format_chat(self, messages: Sequence[dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user").strip().upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]\n{content}")
        return "\n\n".join(lines)
