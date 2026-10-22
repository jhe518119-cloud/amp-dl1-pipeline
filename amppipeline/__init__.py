"""Open re-implementation of the AMP-DL1 deep-learning design pipeline.

The package implements the pipeline described in the manuscript's
Supplemental Text S1: a variational autoencoder that generates peptides, a
CNN-RNN model that predicts MIC against *Escherichia coli*, and the screening
funnel (predicted MIC cutoff, ToxinPred, CD-HIT, composite ranking).

The original training code and weights were not preserved. This is a
re-implementation from the published description, not a recovered model, and
it does not reproduce the exact sequences or rankings reported in the paper.
"""

__version__ = "1.0.0"
