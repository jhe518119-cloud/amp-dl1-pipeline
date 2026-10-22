"""Generate the synthetic example tables bundled with the repository.

The outputs are deliberately synthetic: they exercise the file formats and the
screening funnel, and they must never be presented as experimental data. The
generator is deterministic so the committed files can be regenerated exactly.

Usage: python tools/make_example_data.py
"""

from __future__ import annotations

import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CATIONIC = "KR"
HYDROPHOBIC = "AILMFVW"


def make_sequence(rng: np.random.Generator) -> str:
    length = int(rng.integers(8, 49))
    weights = np.ones(20)
    for index, residue in enumerate(AMINO_ACIDS):
        if residue in CATIONIC:
            weights[index] = 4.0
        elif residue in HYDROPHOBIC:
            weights[index] = 2.0
    weights = weights / weights.sum()
    return "".join(rng.choice(list(AMINO_ACIDS), size=length, p=weights))


def synthetic_score(sequence: str, rng: np.random.Generator) -> float:
    cations = sum(1 for residue in sequence if residue in CATIONIC)
    hydrophobic = sum(1 for residue in sequence if residue in HYDROPHOBIC)
    score = 1.05 - 0.045 * cations + 0.004 * hydrophobic + 0.0008 * len(sequence)
    score += float(rng.normal(0.0, 0.18))
    return score


def main() -> None:
    os.makedirs(EXAMPLES, exist_ok=True)
    rng = np.random.default_rng(20261019)

    rows = []
    seen = set()
    while len(rows) < 560:
        sequence = make_sequence(rng)
        if sequence in seen:
            continue
        seen.add(sequence)
        log10_mic = synthetic_score(sequence, rng)
        rows.append(
            {
                "id": "syn_%04d" % (len(rows) + 1),
                "sequence": sequence,
                "predicted_log10_mic": "%.4f" % log10_mic,
                "predicted_mic_um": "%.4f" % (10.0 ** log10_mic),
                "source": "synthetic_example",
            }
        )

    # Add near-duplicates so that the dereplication stage has something to do.
    base_rows = list(rows)
    for index in rng.choice(len(base_rows), size=40, replace=False):
        template = base_rows[int(index)]
        sequence = list(template["sequence"])
        position = int(rng.integers(0, len(sequence)))
        sequence[position] = AMINO_ACIDS[int(rng.integers(0, len(AMINO_ACIDS)))]
        variant = "".join(sequence)
        if variant in seen:
            continue
        seen.add(variant)
        log10_mic = float(template["predicted_log10_mic"]) + float(rng.normal(0.0, 0.02))
        rows.append(
            {
                "id": "syn_%04d" % (len(rows) + 1),
                "sequence": variant,
                "predicted_log10_mic": "%.4f" % log10_mic,
                "predicted_mic_um": "%.4f" % (10.0 ** log10_mic),
                "source": "synthetic_example",
            }
        )

    predictions_path = os.path.join(EXAMPLES, "example_predictions.csv")
    with open(predictions_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "sequence",
                "predicted_log10_mic",
                "predicted_mic_um",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    toxinpred_path = os.path.join(EXAMPLES, "example_toxinpred.csv")
    with open(toxinpred_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction"])
        for row in rows:
            # Deterministic pseudo-toxicity, roughly one sequence in ten.
            toxic = (sum(ord(character) for character in row["sequence"]) % 10) == 0
            writer.writerow([row["id"], "Toxin" if toxic else "Non-Toxin"])

    below = sum(1 for row in rows if float(row["predicted_mic_um"]) < 10.0)
    print("wrote %d synthetic rows (%d with predicted MIC < 10 uM)" % (len(rows), below))
    print(predictions_path)
    print(toxinpred_path)


if __name__ == "__main__":
    main()
