"""Chunking.

Recursive character splitting: try the coarsest separator first (a blank line,
i.e. a paragraph), and fall back to finer ones (line, sentence, word) only when
a piece is still too long. That keeps chunks semantically whole wherever the
document allows it.

Parameters and why they are what they are:

CHUNK_SIZE = 800 characters
    ~150-200 English words. Big enough to hold a complete argument, small
    enough to stay well inside the embedding model's 512-token window, so nothing is
    silently truncated at embedding time.

CHUNK_OVERLAP = 150 characters
    ~20% of the chunk. A sentence that straddles a boundary appears whole in
    at least one of the two chunks, so retrieval does not lose it.

Chunks are built per page, so every chunk carries its document name and page
number and a citation is always exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Coarsest first. " " last so a single unbroken string still gets cut.
SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, with everything a citation needs."""

    doc_name: str
    page: int
    chunk_index: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_name}, p.{self.page}"


def _split_recursive(text: str, size: int, overlap: int, separators: Sequence[str]) -> List[str]:
    """Split `text` into pieces of at most `size`, preferring coarse boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    separator = separators[-1]
    for candidate in separators:
        if candidate in text:
            separator = candidate
            break

    parts = text.split(separator) if separator else [text]
    pieces: List[str] = []
    buffer = ""

    for position, part in enumerate(parts):
        piece = part + separator if separator and position < len(parts) - 1 else part
        if len(piece) > size:
            # This part alone is too long: flush what we have, then split it
            # with the next finer separator.
            if buffer.strip():
                pieces.append(buffer.strip())
                buffer = ""
            finer = list(separators[separators.index(separator) + 1 :]) if separator in separators else []
            pieces.extend(_split_recursive(piece, size, overlap, finer) if finer else [piece[:size]])
            continue
        if len(buffer) + len(piece) <= size:
            buffer += piece
        else:
            if buffer.strip():
                pieces.append(buffer.strip())
            buffer = piece

    if buffer.strip():
        pieces.append(buffer.strip())

    return _apply_overlap(pieces, overlap)


def _apply_overlap(pieces: List[str], overlap: int) -> List[str]:
    """Prepend the tail of each piece to the next one."""
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    overlapped = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:]):
        overlapped.append((previous[-overlap:] + " " + current).strip())
    return overlapped


def chunk_pages(pages: List, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Turn ingested pages into chunks, numbered across the whole document."""
    chunks: List[Chunk] = []
    index = 0
    for page in pages:
        for piece in _split_recursive(page.text, size, overlap, SEPARATORS):
            chunks.append(Chunk(doc_name=page.doc_name, page=page.page, chunk_index=index, text=piece))
            index += 1
    return chunks
