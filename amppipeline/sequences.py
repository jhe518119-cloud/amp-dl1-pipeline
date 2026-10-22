"""Sequence validation and one-hot encoding.

Peptides are represented as strings over the 20 standard amino acids. For
modelling they are encoded as ``MAX_LEN x 21`` arrays: one channel per standard
amino acid plus one padding channel. Positions beyond the end of a sequence are
set to the padding channel.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .constants import AA_INDEX, ALPHABET, AMINO_ACIDS, MAX_LEN, PAD_CHAR


def clean_sequence(sequence: str) -> str:
    """Return the sequence upper-cased with all whitespace removed."""
    return "".join(str(sequence).split()).upper()


def is_valid_sequence(sequence: str, max_len: int = MAX_LEN) -> bool:
    """True if the sequence contains only standard residues and fits ``max_len``."""
    sequence = clean_sequence(sequence)
    if not sequence or len(sequence) > max_len:
        return False
    return all(residue in AMINO_ACIDS for residue in sequence)


def filter_valid_sequences(
    records: Iterable[Tuple[str, str]], max_len: int = MAX_LEN
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split FASTA-style records into (valid, rejected).

    Rejected sequences are returned unchanged so callers can log why they were
    dropped. Duplicate sequences are *not* removed here; that is a separate
    screening step.
    """
    valid: List[Tuple[str, str]] = []
    rejected: List[Tuple[str, str]] = []
    for identifier, sequence in records:
        cleaned = clean_sequence(sequence)
        if is_valid_sequence(cleaned, max_len=max_len):
            valid.append((identifier, cleaned))
        else:
            rejected.append((identifier, cleaned))
    return valid, rejected


def one_hot_encode(sequence: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode one peptide as a ``(max_len, 21)`` float32 array."""
    sequence = clean_sequence(sequence)
    if len(sequence) > max_len:
        raise ValueError(
            "sequence longer than %d residues: %s" % (max_len, sequence)
        )
    unknown = sorted(set(sequence) - set(AMINO_ACIDS))
    if unknown:
        raise ValueError("non-standard residues %s in %s" % (unknown, sequence))
    encoded = np.zeros((max_len, len(ALPHABET)), dtype=np.float32)
    for position, residue in enumerate(sequence):
        encoded[position, AA_INDEX[residue]] = 1.0
    encoded[len(sequence) :, AA_INDEX[PAD_CHAR]] = 1.0
    return encoded


def one_hot_encode_many(
    sequences: Sequence[str], max_len: int = MAX_LEN
) -> np.ndarray:
    """Encode many peptides as a ``(n, max_len, 21)`` float32 array."""
    if not sequences:
        return np.zeros((0, max_len, len(ALPHABET)), dtype=np.float32)
    return np.stack([one_hot_encode(seq, max_len=max_len) for seq in sequences])


def decode_one_hot(encoded: np.ndarray) -> str:
    """Decode a ``(max_len, 21)`` array (or a probability matrix) to a sequence.

    Trailing padding channels are trimmed. Internal padding, if the model emits
    it, is preserved so that malformed outputs remain visible downstream.
    """
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or encoded.shape[1] != len(ALPHABET):
        raise ValueError("expected an array of shape (length, %d)" % len(ALPHABET))
    indices = encoded.argmax(axis=-1)
    return "".join(ALPHABET[index] for index in indices).rstrip(PAD_CHAR)
