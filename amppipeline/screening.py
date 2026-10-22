"""Screening and ranking of generated peptides.

The pipeline order follows Text S1, section S1.4:

1. retain sequences with a predicted MIC below 10 micromolar;
2. remove potentially toxic peptides (ToxinPred 3.0 web server, default
   parameters);
3. remove redundant sequences (CD-HIT);
4. rank the remaining candidates by a composite score and select the top ones.

ToxinPred has no public batch API, so the ToxinPred step is performed by the
user on the web server and supplied here as a CSV file with columns ``id`` and
``prediction`` (``Toxin`` or ``Non-Toxin``). CD-HIT is an external binary; a
pure-Python greedy clustering fallback is provided for small sets and for the
bundled example, but it is not CD-HIT and is documented as approximate.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Step 1: predicted MIC cutoff
# --------------------------------------------------------------------------- #


def filter_by_predicted_mic(
    frame: pd.DataFrame, cutoff_um: float = 10.0
) -> pd.DataFrame:
    """Keep sequences with a predicted MIC strictly below ``cutoff_um``."""
    if "predicted_mic_um" not in frame.columns:
        raise ValueError("frame needs a 'predicted_mic_um' column")
    return frame[frame["predicted_mic_um"].astype(float) < float(cutoff_um)].copy()


# --------------------------------------------------------------------------- #
# Step 2: ToxinPred
# --------------------------------------------------------------------------- #


def _normalise_toxinpred_label(label: str) -> str:
    text = str(label).strip().lower().replace("_", "-").replace(" ", "-")
    if text in {"toxin", "toxic", "1", "true", "yes"}:
        return "toxin"
    if text in {"non-toxin", "nontoxin", "non-toxic", "nontoxic", "0", "false", "no"}:
        return "non-toxin"
    raise ValueError("unrecognised ToxinPred label: %r" % label)


def read_toxinpred(path: str) -> Dict[str, str]:
    """Read a ToxinPred result table into ``{identifier: 'toxin'|'non-toxin'}``."""
    table = pd.read_csv(path)
    columns = {column.lower(): column for column in table.columns}
    id_column = None
    for candidate in ("id", "identifier", "name", "sequence_id"):
        if candidate in columns:
            id_column = columns[candidate]
            break
    label_column = None
    for candidate in ("prediction", "result", "class", "label"):
        if candidate in columns:
            label_column = columns[candidate]
            break
    if id_column is None or label_column is None:
        raise ValueError(
            "ToxinPred file %s needs an id column and a prediction column" % path
        )
    result: Dict[str, str] = {}
    for _, row in table.iterrows():
        result[str(row[id_column]).strip()] = _normalise_toxinpred_label(
            row[label_column]
        )
    return result


def apply_toxinpred(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    """Keep rows predicted to be non-toxic.

    Every sequence must have a ToxinPred result; a missing result raises an
    error rather than silently dropping the sequence, so that the screening
    funnel stays auditable.
    """
    predictions = read_toxinpred(path)
    missing = [identifier for identifier in frame["id"] if str(identifier) not in predictions]
    if missing:
        raise ValueError(
            "ToxinPred results missing for %d sequence(s), e.g. %s"
            % (len(missing), ", ".join(map(str, missing[:5])))
        )
    keep = frame["id"].map(lambda identifier: predictions[str(identifier)] == "non-toxin")
    return frame[keep].copy()


# --------------------------------------------------------------------------- #
# Step 3: redundancy removal
# --------------------------------------------------------------------------- #


def sequence_identity(first: str, second: str) -> float:
    """Gap-free identity over the shorter sequence.

    CD-HIT reports identity over the shorter sequence; this is the same
    convention, without alignment. It is an approximation used only by the
    pure-Python fallback.
    """
    shorter = min(len(first), len(second))
    if shorter == 0:
        return 0.0
    matches = sum(1 for a, b in zip(first, second) if a == b)
    return matches / shorter


def greedy_dereplicate(
    records: Sequence[Tuple[str, str]],
    identity: float = 0.9,
    max_sequences: int = 20000,
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Greedy identity clustering used when CD-HIT is unavailable.

    Sequences are visited longest first; each is assigned to the first cluster
    whose representative is at least ``identity`` similar, otherwise it starts a
    new cluster. This reproduces the *purpose* of CD-HIT but not its exact
    clustering. It is quadratic in the number of sequences and is therefore
    capped by ``max_sequences``.
    """
    if len(records) > max_sequences:
        raise ValueError(
            "the pure-Python fallback is limited to %d sequences; install CD-HIT "
            "and pass --cdhit-bin for larger sets" % max_sequences
        )
    order = sorted(range(len(records)), key=lambda index: (-len(records[index][1]), index))
    representatives: List[Tuple[str, str]] = []
    representative_of: Dict[str, str] = {}
    for index in order:
        identifier, sequence = records[index]
        assigned = None
        for representative_id, representative_sequence in representatives:
            if sequence_identity(sequence, representative_sequence) >= identity:
                assigned = representative_id
                break
        if assigned is None:
            representatives.append((identifier, sequence))
            representative_of[identifier] = identifier
        else:
            representative_of[identifier] = assigned
    return representatives, representative_of


def _cdhit_word_size(identity: float) -> int:
    """CD-HIT's recommended word size for a given identity threshold."""
    if identity >= 0.7:
        return 5
    if identity >= 0.6:
        return 4
    if identity >= 0.5:
        return 3
    return 2


def _parse_cdhit_clusters(path: str) -> Dict[str, str]:
    """Parse a CD-HIT ``.clstr`` file into ``{member_id: representative_id}``."""
    mapping: Dict[str, str] = {}
    current: List[Tuple[str, bool]] = []

    def flush() -> None:
        representative = None
        for member, is_representative in current:
            if is_representative:
                representative = member
                break
        if representative is None and current:
            representative = current[0][0]
        for member, _ in current:
            mapping[member] = representative

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if current:
                    flush()
                current = []
                continue
            # Lines look like: "0  279aa, >identifier... *"
            if ">" not in line:
                continue
            identifier = line.split(">", 1)[1].split("...", 1)[0].strip()
            current.append((identifier, line.rstrip().endswith("*")))
    if current:
        flush()
    return mapping


def run_cdhit(
    records: Sequence[Tuple[str, str]],
    identity: float = 0.9,
    binary: str = "cd-hit",
    threads: int = 1,
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Run CD-HIT and return ``(representatives, {member: representative})``."""
    executable = shutil.which(binary)
    if executable is None and os.path.exists(binary):
        executable = binary
    if executable is None:
        raise FileNotFoundError(
            "CD-HIT binary not found (%s); install CD-HIT or use "
            "--dereplicate greedy for small sets" % binary
        )
    with tempfile.TemporaryDirectory(prefix="ampdl_cdhit_") as workdir:
        input_fasta = os.path.join(workdir, "input.fasta")
        output_fasta = os.path.join(workdir, "representatives.fasta")
        with open(input_fasta, "w", encoding="utf-8", newline="\n") as handle:
            for identifier, sequence in records:
                handle.write(">%s\n%s\n" % (identifier, sequence))
        command = [
            executable,
            "-i",
            input_fasta,
            "-o",
            output_fasta,
            "-c",
            "%.2f" % identity,
            "-n",
            str(_cdhit_word_size(identity)),
            "-T",
            str(int(threads)),
            "-M",
            "0",
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "CD-HIT failed (%d): %s" % (completed.returncode, completed.stderr.strip())
            )
        mapping = _parse_cdhit_clusters(output_fasta + ".clstr")
        representatives = [
            (identifier, sequence)
            for identifier, sequence in records
            if mapping.get(identifier) == identifier
        ]
    return representatives, mapping


def dereplicate(
    frame: pd.DataFrame,
    method: str = "greedy",
    identity: float = 0.9,
    cdhit_binary: str = "cd-hit",
    threads: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Remove redundant sequences and return one row per cluster representative."""
    records = list(zip(frame["id"].astype(str), frame["sequence"]))
    if method == "cdhit":
        representatives, mapping = run_cdhit(
            records, identity=identity, binary=cdhit_binary, threads=threads
        )
    elif method == "greedy":
        representatives, mapping = greedy_dereplicate(records, identity=identity)
    else:
        raise ValueError("dereplicate must be 'cdhit' or 'greedy'")
    representative_ids = [identifier for identifier, _ in representatives]
    selected = frame[frame["id"].astype(str).isin(representative_ids)].copy()
    selected["cluster_representative"] = selected["id"].astype(str)
    return selected, mapping


# --------------------------------------------------------------------------- #
# Step 4: composite score and selection
# --------------------------------------------------------------------------- #

#: Default composite score: a single term, the negated min-max-normalised
#: predicted log10 MIC, so that lower predicted MIC scores higher. The
#: composition and weights of the score used in the original study were not
#: recorded; this default is a documented, transparent stand-in and can be
#: replaced through ``weights``.
DEFAULT_WEIGHTS: Dict[str, float] = {"predicted_log10_mic": -1.0}


def composite_score(
    frame: pd.DataFrame, weights: Dict[str, float] | None = None
) -> pd.DataFrame:
    """Add a ``composite_score`` column (higher is better).

    Each named column is min-max normalised, then combined as a weighted sum.
    A negative weight means "smaller is better" for that column.
    """
    weights = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)
    if not weights:
        raise ValueError("at least one score term is required")
    score = np.zeros(len(frame), dtype=float)
    for column, weight in weights.items():
        if column not in frame.columns:
            raise ValueError("score term %r is not a column of the frame" % column)
        values = frame[column].astype(float).to_numpy()
        low, high = float(np.min(values)), float(np.max(values))
        if high > low:
            normalised = (values - low) / (high - low)
        else:
            normalised = np.zeros_like(values)
        score += float(weight) * normalised
    result = frame.copy()
    result["composite_score"] = score
    return result


def rank_candidates(frame: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Sort by composite score (descending) and optionally keep the top ``top_n``."""
    ranked = frame.sort_values(
        ["composite_score", "predicted_mic_um"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    if top_n is not None:
        ranked = ranked.head(int(top_n)).copy()
    return ranked


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def screen(
    frame: pd.DataFrame,
    mic_cutoff_um: float = 10.0,
    toxinpred_path: str | None = None,
    dereplicate_method: str = "greedy",
    identity: float = 0.9,
    cdhit_binary: str = "cd-hit",
    threads: int = 1,
    n_candidates: int = 1000,
    top_n: int = 10,
    weights: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run the full screening funnel and return stages plus a summary.

    Returns a dictionary with keys ``stage1_mic``, ``stage2_toxinpred``,
    ``stage3_dereplicated``, ``candidates``, ``top`` and ``summary``.
    """
    summary: Dict[str, object] = {"input": int(len(frame))}

    stage1 = filter_by_predicted_mic(frame, cutoff_um=mic_cutoff_um)
    summary["stage1_mic_cutoff"] = int(len(stage1))

    if toxinpred_path:
        stage2 = apply_toxinpred(stage1, toxinpred_path)
    else:
        stage2 = stage1.copy()
    summary["stage2_toxinpred"] = int(len(stage2))

    stage3, mapping = dereplicate(
        stage2,
        method=dereplicate_method,
        identity=identity,
        cdhit_binary=cdhit_binary,
        threads=threads,
    )
    summary["stage3_dereplicated"] = int(len(stage3))
    summary["dereplicate_method"] = dereplicate_method

    scored = composite_score(stage3, weights=weights)
    ranked = rank_candidates(scored)
    candidates = ranked.head(int(n_candidates)).copy()
    summary["candidates"] = int(len(candidates))
    summary["n_candidates_requested"] = int(n_candidates)

    top = candidates.head(int(top_n)).copy()
    summary["top"] = int(len(top))
    summary["mic_cutoff_um"] = float(mic_cutoff_um)
    summary["identity"] = float(identity)

    return {
        "stage1_mic": stage1,
        "stage2_toxinpred": stage2,
        "stage3_dereplicated": stage3,
        "candidates": candidates,
        "top": top,
        "summary": summary,
        "cluster_mapping": mapping,
    }
