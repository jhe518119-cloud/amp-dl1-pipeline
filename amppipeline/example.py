"""Bundled offline example: run the screening funnel on synthetic data.

The example exercises every stage after model inference (MIC cutoff, ToxinPred
filter, redundancy removal, composite ranking) without requiring TensorFlow or
any external binary. The bundled sequences and predictions are synthetic and
exist only to demonstrate file formats and the order of operations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .io import read_predictions
from .screening import screen


def example_directory() -> Path:
    """Directory holding the bundled example inputs."""
    return Path(__file__).resolve().parent.parent / "examples"


def run_example(out_dir: str | None = None) -> int:
    directory = example_directory()
    predictions_path = directory / "example_predictions.csv"
    toxinpred_path = directory / "example_toxinpred.csv"
    if out_dir is None:
        out_dir = str(directory / "output")

    frame = read_predictions(str(predictions_path))
    print("example predictions: %d rows" % len(frame))
    result = screen(
        frame,
        mic_cutoff_um=10.0,
        toxinpred_path=str(toxinpred_path) if toxinpred_path.exists() else None,
        dereplicate_method="greedy",
        identity=0.9,
        n_candidates=50,
        top_n=10,
    )
    os.makedirs(out_dir, exist_ok=True)
    result["candidates"].to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
    result["top"].to_csv(os.path.join(out_dir, "top.csv"), index=False)
    with open(os.path.join(out_dir, "screen_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2)
        handle.write("\n")

    summary = result["summary"]
    print(
        "funnel: %(input)d -> %(stage1_mic_cutoff)d (MIC < 10 uM) -> "
        "%(stage2_toxinpred)d (non-toxic) -> %(stage3_dereplicated)d "
        "(dereplicated) -> %(candidates)d candidates -> %(top)d top"
        % summary
    )
    print("outputs written to %s" % os.path.abspath(out_dir))
    return 0
