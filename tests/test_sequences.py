"""Tests for sequence validation and one-hot encoding."""

import unittest

import numpy as np

from amppipeline.constants import ALPHABET, PAD_CHAR
from amppipeline.sequences import (
    decode_one_hot,
    filter_valid_sequences,
    is_valid_sequence,
    one_hot_encode,
    one_hot_encode_many,
)


class SequenceEncodingTests(unittest.TestCase):
    def test_one_hot_shape_and_padding(self):
        encoded = one_hot_encode("ACD", max_len=10)
        self.assertEqual(encoded.shape, (10, 21))
        np.testing.assert_allclose(encoded[:3].sum(axis=1), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            encoded[3:, ALPHABET.index(PAD_CHAR)], np.ones(7, dtype=np.float32)
        )

    def test_round_trip(self):
        sequence = "KKLLWW"
        self.assertEqual(decode_one_hot(one_hot_encode(sequence, max_len=12)), sequence)

    def test_invalid_residue_rejected(self):
        self.assertFalse(is_valid_sequence("ACDX"))
        with self.assertRaises(ValueError):
            one_hot_encode("ACDX")

    def test_too_long_rejected(self):
        self.assertFalse(is_valid_sequence("A" * 51))
        self.assertTrue(is_valid_sequence("A" * 50))

    def test_whitespace_and_case(self):
        self.assertEqual(decode_one_hot(one_hot_encode(" acd\n")), "ACD")

    def test_filter_valid_sequences(self):
        records = [("a", "ACD"), ("b", "AC1"), ("c", "A" * 60)]
        valid, rejected = filter_valid_sequences(records)
        self.assertEqual([identifier for identifier, _ in valid], ["a"])
        self.assertEqual([identifier for identifier, _ in rejected], ["b", "c"])

    def test_encode_many_empty(self):
        encoded = one_hot_encode_many([])
        self.assertEqual(encoded.shape, (0, 50, 21))


if __name__ == "__main__":
    unittest.main()
