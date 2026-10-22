"""Command-line interface for the AMP-DL1 design pipeline.

Subcommands
-----------
train-vae         train the generative model on a peptide FASTA file
train-predictor   train the CNN-RNN MIC predictor on a MIC table
generate          sample de novo peptides from a trained VAE
predict           predict MIC for a peptide FASTA file
screen            run the screening funnel (MIC cutoff, ToxinPred, CD-HIT, rank)
example           run the bundled offline example (no TensorFlow required)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .io import read_fasta, read_predictions, write_fasta
from .sequences import (
    decode_one_hot,
    filter_valid_sequences,
    is_valid_sequence,
    one_hot_encode_many,
)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(int(seed))
    try:
        import tensorflow as tf

        tf.random.set_seed(int(seed))
    except ImportError:
        pass


def _command_train_vae(args: argparse.Namespace) -> int:
    from .vae import DEFAULT_CONFIG, build_vae, save_vae

    _set_seed(args.seed)
    records = read_fasta(args.fasta)
    valid, rejected = filter_valid_sequences(records, max_len=args.max_len)
    print("read %d sequences; %d valid, %d rejected" % (len(records), len(valid), len(rejected)))
    if not valid:
        raise SystemExit("no valid sequences in %s" % args.fasta)

    import tensorflow as tf

    from .sequences import one_hot_encode

    def encode(records):
        for _, sequence in records:
            yield one_hot_encode(sequence, max_len=args.max_len)

    signature = tf.TensorSpec(shape=(int(args.max_len), 21), dtype=tf.float32)
    n_validation = int(round(len(valid) * float(args.validation_split)))
    n_validation = min(max(n_validation, 1), len(valid) - 1) if args.validation_split > 0 else 0
    train_records = valid[:-n_validation] if n_validation else valid
    validation_records = valid[-n_validation:] if n_validation else []
    train_dataset = tf.data.Dataset.from_generator(
        lambda: encode(train_records), output_signature=signature
    )
    train_dataset = (
        train_dataset.shuffle(min(len(train_records), int(args.shuffle_buffer)), seed=args.seed)
        .batch(int(args.batch_size))
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = None
    if validation_records:
        validation_dataset = (
            tf.data.Dataset.from_generator(
                lambda: encode(validation_records), output_signature=signature
            )
            .batch(int(args.batch_size))
            .prefetch(tf.data.AUTOTUNE)
        )

    config: Dict[str, object] = dict(DEFAULT_CONFIG)
    config.update(
        {
            "max_len": int(args.max_len),
            "latent_dim": int(args.latent_dim),
            "gru_units": int(args.gru_units),
            "beta": float(args.beta),
            "learning_rate": float(args.learning_rate),
        }
    )
    vae, _, config = build_vae(config)
    callbacks = []
    from tensorflow import keras

    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_dataset is not None else "loss",
            patience=int(args.patience),
            restore_best_weights=True,
        )
    )
    history = vae.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(args.epochs),
        callbacks=callbacks,
        verbose=2,
    )
    weights_path = os.path.join(args.out_dir, "vae.weights.h5")
    config_path = save_vae(vae, config, weights_path)
    history_path = os.path.join(args.out_dir, "vae_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump({key: [float(v) for v in values] for key, values in history.history.items()}, handle, indent=2)
        handle.write("\n")
    print("saved weights: %s" % weights_path)
    print("saved config:  %s" % config_path)
    print("saved history: %s" % history_path)
    return 0


def _read_mic_table(path: str, max_len: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(path)
    columns = {column.lower(): column for column in frame.columns}
    sequence_column = columns.get("sequence")
    mic_column = None
    for candidate in ("mic_um", "mic", "mic_umolar"):
        if candidate in columns:
            mic_column = columns[candidate]
            break
    if sequence_column is None or mic_column is None:
        raise SystemExit(
            "%s needs a 'sequence' column and a 'mic_um' column" % path
        )
    frame = frame[[sequence_column, mic_column]].rename(
        columns={sequence_column: "sequence", mic_column: "mic_um"}
    )
    frame["sequence"] = frame["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    frame = frame[frame["sequence"].map(lambda s: is_valid_sequence(s, max_len=max_len))]
    frame = frame[pd.to_numeric(frame["mic_um"], errors="coerce") > 0].copy()
    mic = frame["mic_um"].astype(float).to_numpy()
    encoded = one_hot_encode_many(frame["sequence"].tolist(), max_len=max_len)
    return encoded, np.log10(mic).astype(np.float32), frame


def _command_train_predictor(args: argparse.Namespace) -> int:
    from .predictor import DEFAULT_CONFIG, build_predictor, save_predictor

    _set_seed(args.seed)
    encoded, target, frame = _read_mic_table(args.csv, max_len=args.max_len)
    print("read %d usable MIC records" % len(frame))
    if len(frame) < 10:
        raise SystemExit("too few usable records to train")

    order = np.random.default_rng(args.seed).permutation(len(frame))
    encoded, target = encoded[order], target[order]
    n_test = max(1, int(round(len(frame) * args.test_fraction)))
    n_val = max(1, int(round(len(frame) * args.validation_fraction)))
    test_x, test_y = encoded[:n_test], target[:n_test]
    val_x, val_y = encoded[n_test : n_test + n_val], target[n_test : n_test + n_val]
    train_x, train_y = encoded[n_test + n_val :], target[n_test + n_val:]

    config: Dict[str, object] = dict(DEFAULT_CONFIG)
    config.update(
        {
            "max_len": int(args.max_len),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
        }
    )
    model, config = build_predictor(config)
    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(args.patience), restore_best_weights=True
        )
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        callbacks=callbacks,
        verbose=2,
    )
    predictions = model.predict(test_x, batch_size=int(args.batch_size), verbose=0).ravel()
    rmse = float(np.sqrt(np.mean((predictions - test_y) ** 2)))
    pearson = float(np.corrcoef(predictions, test_y)[0, 1]) if len(test_y) > 1 else float("nan")
    print("test RMSE (log10 MIC): %.4f" % rmse)
    print("test Pearson r:        %.4f" % pearson)

    weights_path = os.path.join(args.out_dir, "predictor.weights.h5")
    config_path = save_predictor(model, config, weights_path)
    metrics = {
        "n_train": int(len(train_x)),
        "n_validation": int(len(val_x)),
        "n_test": int(len(test_x)),
        "test_rmse_log10_mic": rmse,
        "test_pearson_r": pearson,
        "best_val_loss": float(min(history.history.get("val_loss", [float("nan")]))),
    }
    metrics_path = os.path.join(args.out_dir, "predictor_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print("saved weights: %s" % weights_path)
    print("saved config:  %s" % config_path)
    print("saved metrics: %s" % metrics_path)
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    from .vae import load_vae, sample_latent

    _set_seed(args.seed)
    _, decoder, config = load_vae(args.weights, args.config)
    latent = sample_latent(int(args.n), int(config["latent_dim"]), seed=args.seed)

    probabilities = decoder.predict(latent, batch_size=int(args.batch_size), verbose=0)
    records: List[Tuple[str, str]] = []
    seen = set()
    invalid = 0
    for index, row in enumerate(probabilities):
        sequence = decode_one_hot(row)
        if not is_valid_sequence(sequence, max_len=int(config["max_len"])):
            invalid += 1
            continue
        if sequence in seen:
            continue
        seen.add(sequence)
        records.append(("generated_%06d" % index, sequence))

    write_fasta(records, args.out_fasta)
    print("sampled %d latent vectors" % int(args.n))
    print("decoded %d unique valid sequences (%d invalid discarded)" % (len(records), invalid))
    print("written: %s" % args.out_fasta)
    return 0


def _command_predict(args: argparse.Namespace) -> int:
    from .predictor import load_predictor

    model, config = load_predictor(args.weights, args.config)
    records = read_fasta(args.input)
    valid, rejected = filter_valid_sequences(records, max_len=int(config["max_len"]))
    print("read %d sequences; %d valid, %d rejected" % (len(records), len(valid), len(rejected)))
    if not valid:
        raise SystemExit("no valid sequences in %s" % args.input)
    encoded = one_hot_encode_many([sequence for _, sequence in valid], max_len=int(config["max_len"]))
    predicted = model.predict(encoded, batch_size=int(args.batch_size), verbose=0).ravel()
    frame = pd.DataFrame(
        {
            "id": [identifier for identifier, _ in valid],
            "sequence": [sequence for _, sequence in valid],
            "predicted_log10_mic": predicted.astype(float),
        }
    )
    frame["predicted_mic_um"] = np.power(10.0, frame["predicted_log10_mic"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frame.to_csv(args.out, index=False)
    print("written: %s (%d rows)" % (args.out, len(frame)))
    return 0


def _command_screen(args: argparse.Namespace) -> int:
    from .screening import screen

    frame = read_predictions(args.predictions)
    print("read %d predictions" % len(frame))
    try:
        result = screen(
            frame,
            mic_cutoff_um=float(args.mic_cutoff),
            toxinpred_path=args.toxinpred,
            dereplicate_method=args.dereplicate,
            identity=float(args.identity),
            cdhit_binary=args.cdhit_bin,
            threads=int(args.threads),
            n_candidates=int(args.n_candidates),
            top_n=int(args.top),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit("screening failed: %s" % exc)
    os.makedirs(args.out_dir, exist_ok=True)
    result["candidates"].to_csv(os.path.join(args.out_dir, "candidates.csv"), index=False)
    result["top"].to_csv(os.path.join(args.out_dir, "top.csv"), index=False)
    summary_path = os.path.join(args.out_dir, "screen_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2)
        handle.write("\n")
    print("stage summary: %s" % json.dumps(result["summary"], sort_keys=True))
    print("written: %s" % os.path.join(args.out_dir, "candidates.csv"))
    print("written: %s" % os.path.join(args.out_dir, "top.csv"))
    print("written: %s" % summary_path)
    return 0


def _command_example(args: argparse.Namespace) -> int:
    from .example import run_example

    return run_example(args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ampdl",
        description="Open re-implementation of the AMP-DL1 deep-learning design pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_vae = subparsers.add_parser("train-vae", help="train the VAE on peptide sequences")
    train_vae.add_argument("--fasta", required=True, help="training peptides (FASTA)")
    train_vae.add_argument("--out-dir", required=True)
    train_vae.add_argument("--epochs", type=int, default=200)
    train_vae.add_argument("--batch-size", type=int, default=64)
    train_vae.add_argument("--validation-split", type=float, default=0.05)
    train_vae.add_argument("--patience", type=int, default=20)
    train_vae.add_argument("--shuffle-buffer", type=int, default=10000)
    train_vae.add_argument("--max-len", type=int, default=50)
    train_vae.add_argument("--latent-dim", type=int, default=50)
    train_vae.add_argument("--gru-units", type=int, default=512)
    train_vae.add_argument("--beta", type=float, default=1.0)
    train_vae.add_argument("--learning-rate", type=float, default=1e-3)
    train_vae.add_argument("--seed", type=int, default=0)
    train_vae.set_defaults(func=_command_train_vae)

    train_predictor = subparsers.add_parser(
        "train-predictor", help="train the CNN-RNN MIC predictor"
    )
    train_predictor.add_argument("--csv", required=True, help="table with sequence and mic_um columns")
    train_predictor.add_argument("--out-dir", required=True)
    train_predictor.add_argument("--epochs", type=int, default=500)
    train_predictor.add_argument("--batch-size", type=int, default=64)
    train_predictor.add_argument("--validation-fraction", type=float, default=0.1)
    train_predictor.add_argument("--test-fraction", type=float, default=0.1)
    train_predictor.add_argument("--patience", type=int, default=20)
    train_predictor.add_argument("--max-len", type=int, default=50)
    train_predictor.add_argument("--learning-rate", type=float, default=1e-3)
    train_predictor.add_argument("--seed", type=int, default=0)
    train_predictor.set_defaults(func=_command_train_predictor)

    generate = subparsers.add_parser("generate", help="sample de novo peptides")
    generate.add_argument("--weights", required=True)
    generate.add_argument("--config", default=None)
    generate.add_argument("--n", type=int, default=100000)
    generate.add_argument("--batch-size", type=int, default=1024)
    generate.add_argument("--out-fasta", required=True)
    generate.add_argument("--seed", type=int, default=0)
    generate.set_defaults(func=_command_generate)

    predict = subparsers.add_parser("predict", help="predict MIC for peptides")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--config", default=None)
    predict.add_argument("--input", required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--batch-size", type=int, default=1024)
    predict.set_defaults(func=_command_predict)

    screen = subparsers.add_parser("screen", help="run the screening funnel")
    screen.add_argument("--predictions", required=True)
    screen.add_argument("--out-dir", required=True)
    screen.add_argument("--mic-cutoff", type=float, default=10.0)
    screen.add_argument("--toxinpred", default=None, help="ToxinPred CSV (id, prediction)")
    screen.add_argument("--dereplicate", choices=("greedy", "cdhit"), default="greedy")
    screen.add_argument("--identity", type=float, default=0.9)
    screen.add_argument("--cdhit-bin", default="cd-hit")
    screen.add_argument("--threads", type=int, default=1)
    screen.add_argument("--n-candidates", type=int, default=1000)
    screen.add_argument("--top", type=int, default=10)
    screen.set_defaults(func=_command_screen)

    example = subparsers.add_parser("example", help="run the bundled offline example")
    example.add_argument("--out-dir", default="examples/output")
    example.set_defaults(func=_command_example)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
