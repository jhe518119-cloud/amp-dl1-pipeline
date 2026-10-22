"""End-to-end smoke test for the AMP-DL1 pipeline.

Generates a small synthetic data set, then runs every CLI subcommand once:

    train-vae -> generate -> train-predictor -> predict -> screen

The point is to prove that the code paths execute, not to produce meaningful
models. Everything here is synthetic and deterministic; no external data,
GPU, ToxinPred, or CD-HIT is required.

Usage:
    python tools/smoke_test.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _random_sequence(rng: random.Random, min_len: int = 10, max_len: int = 30) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def _write_fasta(sequences, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(">smoke_%04d\n%s\n" % (index, sequence))


def _write_mic_csv(records, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "mic_um"])
        for sequence, mic in records:
            writer.writerow([sequence, "%.4f" % mic])


def _run(command) -> None:
    printable = " ".join(str(part) for part in command)
    print("\n$ %s" % printable, flush=True)
    subprocess.run([str(part) for part in command], cwd=str(REPO_ROOT), check=True)


def _count_fasta_records(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def run_smoke(workdir: Path) -> None:
    rng = random.Random(20261022)

    vae_sequences = [_random_sequence(rng, 8, 30) for _ in range(400)]
    vae_fasta = workdir / "vae_train.fasta"
    _write_fasta(vae_sequences, vae_fasta)

    mic_records = [
        (_random_sequence(rng, 8, 30), 10.0 ** rng.uniform(-1.0, 2.0))
        for _ in range(200)
    ]
    mic_csv = workdir / "mic_train.csv"
    _write_mic_csv(mic_records, mic_csv)

    models = workdir / "models"
    models.mkdir()

    _run(
        [
            sys.executable,
            "-m",
            "amppipeline",
            "train-vae",
            "--fasta",
            vae_fasta,
            "--out-dir",
            models,
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--latent-dim",
            "16",
            "--gru-units",
            "32",
            "--validation-split",
            "0.1",
            "--patience",
            "2",
            "--seed",
            "0",
        ]
    )

    generated_fasta = workdir / "generated.fasta"
    _run(
        [
            sys.executable,
            "-m",
            "amppipeline",
            "generate",
            "--weights",
            models / "vae.weights.h5",
            "--n",
            "100",
            "--batch-size",
            "64",
            "--out-fasta",
            generated_fasta,
            "--seed",
            "0",
        ]
    )

    # An untrained VAE can occasionally decode only padding characters. Fall
    # back to the synthetic training peptides so that the downstream commands
    # are still exercised; the `generate` command itself has already run.
    predict_input = generated_fasta
    if _count_fasta_records(generated_fasta) == 0:
        print("WARNING: generate produced no valid sequences; using synthetic input")
        predict_input = vae_fasta

    _run(
        [
            sys.executable,
            "-m",
            "amppipeline",
            "train-predictor",
            "--csv",
            mic_csv,
            "--out-dir",
            models,
            "--epochs",
            "2",
            "--batch-size",
            "32",
            "--validation-fraction",
            "0.2",
            "--test-fraction",
            "0.2",
            "--patience",
            "3",
            "--seed",
            "0",
        ]
    )

    predictions_csv = workdir / "predictions.csv"
    _run(
        [
            sys.executable,
            "-m",
            "amppipeline",
            "predict",
            "--weights",
            models / "predictor.weights.h5",
            "--input",
            predict_input,
            "--out",
            predictions_csv,
            "--batch-size",
            "64",
        ]
    )

    screening_dir = workdir / "screening"
    # The MIC cutoff is deliberately permissive: the predictor is trained for
    # two epochs on 200 synthetic records, so its absolute predictions carry no
    # meaning. The cutoff logic itself is covered by the unit tests.
    _run(
        [
            sys.executable,
            "-m",
            "amppipeline",
            "screen",
            "--predictions",
            predictions_csv,
            "--out-dir",
            screening_dir,
            "--mic-cutoff",
            "1000",
            "--dereplicate",
            "greedy",
            "--identity",
            "0.9",
            "--n-candidates",
            "20",
            "--top",
            "5",
        ]
    )

    expected = [
        models / "vae.weights.h5",
        models / "vae.weights.config.json",
        models / "predictor.weights.h5",
        models / "predictor.weights.config.json",
        predictions_csv,
        screening_dir / "candidates.csv",
        screening_dir / "top.csv",
        screening_dir / "screen_summary.json",
    ]
    missing = [path for path in expected if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(
            "smoke test failed; missing outputs: %s" % ", ".join(str(p) for p in missing)
        )
    print("\nsmoke test OK: all commands ran and all expected outputs exist")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        default=None,
        help="directory for intermediate files (default: a temporary directory)",
    )
    args = parser.parse_args(argv)

    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise SystemExit(
            "TensorFlow is required for the smoke test. Install it with "
            "'pip install \"tensorflow>=2.11,<2.16\"' or run the test in CI."
        )

    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        run_smoke(workdir)
        print("intermediate files kept in %s" % workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="ampdl_smoke_") as temporary:
            run_smoke(Path(temporary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
