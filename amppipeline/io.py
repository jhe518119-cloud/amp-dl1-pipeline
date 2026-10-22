"""Reading and writing FASTA and CSV files."""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .sequences import clean_sequence


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read a FASTA file into ``(identifier, sequence)`` pairs.

    The identifier is the header up to the first whitespace character, matching
    the convention used by CD-HIT and most sequence tools.
    """
    records: List[Tuple[str, str]] = []
    identifier = None
    chunks: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, clean_sequence("".join(chunks))))
                identifier = line[1:].split()[0] if line[1:].split() else ""
                chunks = []
            else:
                if identifier is None:
                    raise ValueError("sequence data before the first header in %s" % path)
                chunks.append(line)
    if identifier is not None:
        records.append((identifier, clean_sequence("".join(chunks))))
    return records


def write_fasta(records: Iterable[Tuple[str, str]], path: str) -> str:
    """Write ``(identifier, sequence)`` pairs as FASTA. Returns the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in records:
            handle.write(">%s\n%s\n" % (identifier, sequence))
    return path


def write_table(rows: Sequence[Dict[str, object]], path: str, fields: Sequence[str]) -> str:
    """Write a list of dictionaries as a CSV file with the given column order."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def read_predictions(path: str) -> pd.DataFrame:
    """Read a predictions table and validate the required columns.

    Required columns: ``id``, ``sequence`` and one of ``predicted_log10_mic``
    or ``predicted_mic_um``. If only ``predicted_mic_um`` is present, the
    log10 column is added; if only the log10 column is present, ``predicted_mic_um``
    is added.
    """
    frame = pd.read_csv(path)
    required = {"id", "sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "predictions file %s is missing required columns: %s"
            % (path, ", ".join(sorted(missing)))
        )
    has_log = "predicted_log10_mic" in frame.columns
    has_um = "predicted_mic_um" in frame.columns
    if not has_log and not has_um:
        raise ValueError(
            "predictions file %s needs 'predicted_log10_mic' or 'predicted_mic_um'"
            % path
        )
    if not has_log:
        frame["predicted_log10_mic"] = np_log10(frame["predicted_mic_um"].astype(float))
    if not has_um:
        frame["predicted_mic_um"] = (10.0 ** frame["predicted_log10_mic"].astype(float))
    frame["sequence"] = frame["sequence"].map(clean_sequence)
    return frame


def np_log10(values):
    """log10 with a clear error for non-positive MIC values."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    if (array <= 0).any():
        raise ValueError("MIC values must be positive before log10 transformation")
    return np.log10(array)
