"""Tests for the screening funnel."""

import os
import tempfile
import unittest

import pandas as pd

from amppipeline.screening import (
    _parse_cdhit_clusters,
    apply_toxinpred,
    composite_score,
    filter_by_predicted_mic,
    greedy_dereplicate,
    rank_candidates,
    screen,
    sequence_identity,
)


def make_frame():
    return pd.DataFrame(
        {
            "id": ["p1", "p2", "p3", "p4", "p5"],
            "sequence": ["KKLLWWAA", "KKLLWWAA", "KKLLWWAB", "RRFFYYCC", "GGDDTTEE"],
            "predicted_log10_mic": [0.0, 0.1, 0.2, 0.9, 1.2],
            "predicted_mic_um": [1.0, 1.26, 1.58, 7.94, 15.85],
        }
    )


class FilterTests(unittest.TestCase):
    def test_mic_filter_is_strictly_below(self):
        filtered = filter_by_predicted_mic(make_frame(), cutoff_um=10.0)
        self.assertEqual(list(filtered["id"]), ["p1", "p2", "p3", "p4"])

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            filter_by_predicted_mic(pd.DataFrame({"id": ["a"]}))


class ToxinPredTests(unittest.TestCase):
    def _write(self, rows):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        handle.write("id,prediction\n")
        for identifier, label in rows:
            handle.write("%s,%s\n" % (identifier, label))
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_label_normalisation_and_filter(self):
        path = self._write(
            [("p1", "Non-Toxin"), ("p2", "Toxin"), ("p3", "non_toxin"), ("p4", "TOXIN"), ("p5", "Non-Toxin")]
        )
        kept = apply_toxinpred(make_frame(), path)
        self.assertEqual(list(kept["id"]), ["p1", "p3", "p5"])

    def test_missing_result_raises(self):
        path = self._write([("p1", "Non-Toxin")])
        with self.assertRaises(ValueError):
            apply_toxinpred(make_frame(), path)


class DereplicationTests(unittest.TestCase):
    def test_identity_uses_shorter_sequence(self):
        self.assertAlmostEqual(sequence_identity("AAAA", "AAAB"), 0.75)
        self.assertEqual(sequence_identity("AAAA", "AAAA"), 1.0)

    def test_greedy_clusters_near_duplicates(self):
        records = [
            ("a", "KKLLWWAA"),
            ("b", "KKLLWWAA"),  # exact duplicate -> same cluster
            ("c", "KKLLWWAB"),  # one substitution in eight -> identity 0.875
            ("d", "RRFFYYCC"),
        ]
        representatives, mapping = greedy_dereplicate(records, identity=0.9)
        self.assertEqual(len(representatives), 3)
        self.assertEqual(mapping["b"], "a")
        self.assertNotEqual(mapping["c"], "a")
        self.assertEqual(mapping["d"], "d")

    def test_parse_cdhit_clusters(self):
        content = (
            ">Cluster 0\n"
            "0\t279aa, >p1... *\n"
            "1\t250aa, >p2... at 95%\n"
            ">Cluster 1\n"
            "0\t300aa, >p3... *\n"
        )
        handle = tempfile.NamedTemporaryFile("w", suffix=".clstr", delete=False)
        handle.write(content)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        mapping = _parse_cdhit_clusters(handle.name)
        self.assertEqual(mapping, {"p1": "p1", "p2": "p1", "p3": "p3"})


class ScoringTests(unittest.TestCase):
    def test_default_composite_prefers_lower_mic(self):
        frame = composite_score(make_frame())
        ranked = rank_candidates(frame)
        self.assertEqual(ranked.iloc[0]["id"], "p1")
        self.assertEqual(ranked.iloc[-1]["id"], "p5")
        self.assertTrue(ranked["composite_score"].is_monotonic_decreasing)

    def test_unknown_term_raises(self):
        with self.assertRaises(ValueError):
            composite_score(make_frame(), weights={"not_a_column": 1.0})

    def test_screen_funnel_counts(self):
        result = screen(make_frame(), mic_cutoff_um=10.0, n_candidates=2, top_n=2)
        summary = result["summary"]
        self.assertEqual(summary["input"], 5)
        self.assertEqual(summary["stage1_mic_cutoff"], 4)
        self.assertEqual(summary["stage2_toxinpred"], 4)
        self.assertEqual(summary["candidates"], 2)
        self.assertEqual(summary["top"], 2)
        self.assertEqual(len(result["top"]), 2)


if __name__ == "__main__":
    unittest.main()
