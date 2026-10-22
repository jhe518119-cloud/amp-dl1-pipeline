"""Shared constants for the AMP-DL1 design pipeline.

The sequence alphabet follows the description in the manuscript: peptides are
one-hot encoded as 50 x 21 matrices, where the 21 channels are the 20 standard
amino acids plus a padding symbol.
"""

#: The 20 standard amino acids, in the order used for one-hot encoding.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Padding symbol, used for sequences shorter than ``MAX_LEN``.
PAD_CHAR = "-"

#: Full alphabet: 20 amino acids followed by the padding symbol (21 channels).
ALPHABET = AMINO_ACIDS + PAD_CHAR

#: Index of each symbol in the one-hot encoding.
AA_INDEX = {symbol: index for index, symbol in enumerate(ALPHABET)}

#: Maximum peptide length, matching the 50 x 21 input shape.
MAX_LEN = 50
