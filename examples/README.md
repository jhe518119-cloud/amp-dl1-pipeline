# Example data (synthetic)

These files are **synthetic** and exist only to demonstrate the file formats and
exercise the screening funnel without TensorFlow or external tools.

| File | Contents |
|---|---|
| `example_predictions.csv` | 595 synthetic peptides with mock predicted MIC values (`id, sequence, predicted_log10_mic, predicted_mic_um, source`) |
| `example_toxinpred.csv` | mock ToxinPred output (`id, prediction`) covering every sequence |

The `source` column is `synthetic_example` in every row. The sequences were
drawn from a biased random model with a deterministic seed; the predicted MIC
values come from a simple score, not from a trained network. Near-duplicate
sequences were added so that the dereplication stage has an effect.

Regenerate with:

```bash
python tools/make_example_data.py
```

Run the example with:

```bash
python -m amppipeline example --out-dir examples/output
```

Do not use these sequences or values as experimental data.
