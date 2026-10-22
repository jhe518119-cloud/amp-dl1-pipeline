# AMP-DL1 design pipeline

Open re-implementation of the deep-learning pipeline used to design the
antimicrobial peptide AMP-DL1, described in the Supplemental Text S1 of:

> **A machine-learning-designed antimicrobial peptide potentiates colistin
> against *mcr-1*-positive multidrug-resistant *Escherichia coli***
> Jie He, Hongxia Ma (corresponding author)

> **Re-implementation notice.** The original training code and model weights
> were not preserved. This repository is an independent re-implementation of
> the architecture and screening procedure described in the paper. It is not
> the code that produced the published results, and retraining it will not
> reproduce the exact sequences, scores, or rankings reported there. The
> manuscript's results come from the original implementation; see
> [Reproducibility](#reproducibility) for what this repository does and does
> not reproduce.

## Pipeline

```
UniProt peptides ──► VAE pretraining ──► GRAMPA AMPs ──► VAE fine-tuning
                                                             │
                                                     sample 100,000
                                                             ▼
GRAMPA MIC records ──► CNN-RNN training ──► predicted MIC against E. coli
                                                             │
                                              predicted MIC < 10 μM
                                                             ▼
                                        ToxinPred 3.0 (web server, default)
                                                             ▼
                                                    CD-HIT (redundancy)
                                                             ▼
                                             composite score ► top 10
```

The screening order follows Text S1 §S1.4. ToxinPred is a web service with no
public batch API, so its step is performed by the user and supplied as a CSV;
CD-HIT is an external binary, with a pure-Python fallback for small sets.

## Architecture

As described in Text S1 §S1.2:

| Module | Architecture |
|---|---|
| VAE encoder | five 1D convolutional layers → 50-dimensional latent space |
| VAE decoder | four transposed convolutional layers → 512-unit GRU → 21-channel softmax |
| MIC predictor | two 1D convolutional layers (64 filters each) → 256-unit LSTM → fully connected output |

Peptides are one-hot encoded as 50 × 21 matrices (20 amino acids + padding).
The predictor is trained on log10-transformed MIC values with mean squared
error, the Adam optimizer, batch size 64, early stopping, and a maximum of
200–500 epochs (Text S1 §S1.3).

## Repository layout

```
amppipeline/
  constants.py     alphabet and input geometry
  sequences.py     validation, one-hot encoding, decoding
  io.py            FASTA and CSV readers/writers
  vae.py           VAE model, training loss, save/load
  predictor.py     CNN-RNN MIC predictor
  screening.py     MIC cutoff, ToxinPred, CD-HIT, composite ranking
  cli.py           command-line interface
  example.py       bundled offline example
examples/          synthetic inputs for the offline example
tools/             synthetic-data generator and end-to-end smoke test
tests/             unit tests (model tests skip without TensorFlow)
.github/workflows/ CI: unit tests + smoke test on Python 3.10
```

## Install

The original study used Python 3.9 and TensorFlow 2.11 on an NVIDIA RTX 4090.
TensorFlow 2.11 supports Python 3.9–3.10.

```bash
conda env create -f environment.yml
conda activate amp-dl1
# or:
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Data

No training data are bundled. The two public sources are:

- **UniProt** — natural peptide sequences shorter than 50 residues (about
  1.5 × 10⁶ sequences in the original study).
- **GRAMPA** — 5,319 experimentally validated antimicrobial peptides and
  51,345 quantitative MIC records (downloaded in 2024; version not recorded).

Download them from their respective resources and convert them to the input
formats below. The repository ships **synthetic** example data only, for
smoke-testing; it must not be used as scientific data.

Input formats:

```
# peptides for VAE training
>identifier
KKLLWWAA...

# MIC table for predictor training (CSV)
sequence,mic_um
KKLLWWAA...,4.0
```

## Quick start (no TensorFlow needed)

The screening half of the pipeline runs offline on the bundled synthetic data:

```bash
python -m amppipeline example --out-dir examples/output
```

Expected output:

```
funnel: 595 -> 476 (MIC < 10 uM) -> 423 (non-toxic) -> 401 (dereplicated) -> 50 candidates -> 10 top
```

## Full pipeline

```bash
# 1. Train the VAE (GPU recommended; CPU works for a small test)
python -m amppipeline train-vae \
    --fasta data/uniprot_peptides.fasta \
    --out-dir models --epochs 200 --batch-size 64
#    Sequences are one-hot encoded on the fly and streamed with tf.data, so the
#    full 1.5 x 10^6-sequence corpus is not materialised as a single array.

# 2. Fine-tuning on GRAMPA AMPs uses the same command with the GRAMPA FASTA
#    and a shorter schedule; the original fine-tuning setup was not recorded.

# 3. Train the CNN-RNN MIC predictor
python -m amppipeline train-predictor \
    --csv data/grampa_mic.csv \
    --out-dir models --epochs 500 --batch-size 64

# 4. Sample de novo peptides
python -m amppipeline generate \
    --weights models/vae.weights.h5 --n 100000 \
    --out-fasta output/generated.fasta

# 5. Predict MIC against E. coli
python -m amppipeline predict \
    --weights models/predictor.weights.h5 \
    --input output/generated.fasta --out output/predictions.csv

# 6. Submit output/predictions.csv sequences to the ToxinPred 3.0 web server
#    (default parameters) and save the result as output/toxinpred.csv with
#    columns: id, prediction

# 7. Screen and rank
python -m amppipeline screen \
    --predictions output/predictions.csv \
    --toxinpred output/toxinpred.csv \
    --dereplicate cdhit --identity 0.9 --cdhit-bin cd-hit \
    --n-candidates 1000 --top 10 \
    --out-dir output/screening
```

Outputs: `candidates.csv` (the 1,000-candidate pool), `top.csv` (top-ranked
peptides), and `screen_summary.json` (funnel counts at every stage).

### Screening options

| Option | Default | Notes |
|---|---|---|
| `--mic-cutoff` | 10 μM | Text S1 §S1.4 |
| `--toxinpred` | none | CSV from the ToxinPred web server; if omitted the step is skipped |
| `--dereplicate` | `greedy` | `cdhit` requires the CD-HIT binary |
| `--identity` | 0.9 | CD-HIT threshold was not recorded in the original study |
| `--n-candidates` | 1000 | pool size reported in the paper |
| `--top` | 10 | peptides synthesized experimentally |

The `greedy` fallback clusters sequences by gap-free identity over the shorter
sequence. It is quadratic and capped at 20,000 sequences; use CD-HIT for
full-scale runs.

## Reproducibility

**What this repository reproduces**

- the model architectures described in Text S1;
- the training objective and optimiser settings that were recorded;
- the order and logic of the screening funnel;
- reproducible file formats and command-line workflows.

**What it cannot reproduce**

- the original trained weights (not preserved);
- the exact 100,000 sampled sequences and the 1,000-candidate pool;
- the exact top-10 ranking and therefore the identity of AMP-DL1–AMP-DL10;
- the reported performance (RMSE 0.443, Pearson 0.779), which belongs to the
  original training run.

**Parameters that were not recorded, and the defaults used here**

| Parameter | Default in this repository | Status |
|---|---|---|
| Learning rate | 1e-3 | not recorded |
| VAE encoder/decoder filter widths | 32/64/128/128/128 and 32/64/64/64 | architecture described, widths not recorded |
| KL weight (β) | 1.0 | not recorded |
| Early-stopping patience | 20 epochs | not recorded |
| Train/validation/test split | 80/10/10 | not recorded |
| CD-HIT identity | 0.9 | not recorded |
| Composite score | negated min–max-normalised predicted log10 MIC | composition not recorded |
| ToxinPred | web server, default parameters | as reported |

All of these are configurable. The composite score is exposed as a weighted sum
of normalised terms (`weights` in `screening.composite_score`); the shipped
default is a transparent stand-in, not a claim about the original formula.

## Tests

```bash
python -m unittest discover -s tests        # unit tests
python tools/smoke_test.py                  # end-to-end, needs TensorFlow
```

Model tests are skipped automatically when TensorFlow is not installed. The
smoke test generates a tiny synthetic data set and runs every subcommand once
(two short training runs, generation, prediction, screening) on CPU.

CI (`.github/workflows/tests.yml`) runs both on Python 3.10 for every push and
pull request.

## Environment

- Python 3.9–3.10
- TensorFlow 2.11 (reported in the paper); 2.11–2.15 supported by the pinned
  requirement
- numpy, pandas
- Optional: CD-HIT for full-scale redundancy removal

## Citation

If you use this code, please cite the manuscript above. The citation will be
updated with the journal reference upon publication.

## License

MIT for the code in this repository. The bundled example data are synthetic.
UniProt and GRAMPA data are subject to their own terms of use and are not
redistributed here.
