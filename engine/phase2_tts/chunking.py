"""Text chunking for TTS synthesis.

Groups body blocks into paragraphs, splits on sentence boundaries,
and caps chunk length at ~500 characters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_CHUNK_CHARS = 500

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HARD_LIMIT = 500


@dataclass
class Chunk:
    text: str
    source_block_indices: list[int]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? followed by whitespace."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_blocks(body_blocks: list[dict]) -> list[Chunk]:
    """Group body blocks into TTS-ready chunks.

    Rules:
    - Accumulate consecutive blocks into a paragraph buffer.
    - When the buffer exceeds ~500 chars, flush at the next sentence boundary.
    - A single sentence exceeding 500 chars is hard-capped (split at char limit).
    - Each chunk tracks which original block indices contributed text.
    """
    chunks: list[Chunk] = []
    buf_text = ""
    buf_indices: list[int] = []

    def _flush() -> None:
        nonlocal buf_text, buf_indices
        text = buf_text.strip()
        if text and buf_indices:
            chunks.append(Chunk(text=text, source_block_indices=list(buf_indices)))
        buf_text = ""
        buf_indices = []

    for block in body_blocks:
        block_idx = block.get("_index", 0)
        block_text = block["text"].strip()
        if not block_text:
            continue

        candidate = f"{buf_text} {block_text}".strip() if buf_text else block_text

        if len(candidate) <= _MAX_CHUNK_CHARS:
            buf_text = candidate
            buf_indices.append(block_idx)
        else:
            # Flush what we have, start new buffer with this block
            _flush()
            # If this single block exceeds the cap, split it into sentences
            if len(block_text) > _MAX_CHUNK_CHARS:
                sentences = _split_sentences(block_text)
                for sent in sentences:
                    if len(sent) <= _MAX_CHUNK_CHARS:
                        chunks.append(Chunk(text=sent, source_block_indices=[block_idx]))
                    else:
                        # Hard cap: split at character limit
                        for i in range(0, len(sent), _MAX_CHUNK_CHARS):
                            chunks.append(Chunk(text=sent[i:i + _MAX_CHUNK_CHARS], source_block_indices=[block_idx]))
            else:
                buf_text = block_text
                buf_indices = [block_idx]

    _flush()
    return chunks
