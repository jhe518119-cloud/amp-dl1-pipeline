"""Model construction tests (skipped when TensorFlow is unavailable)."""

import importlib.util
import unittest

import numpy as np

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is not installed")
class VaeTests(unittest.TestCase):
    def test_build_and_forward(self):
        from amppipeline.vae import build_vae, sample_latent

        config = {
            "max_len": 10,
            "latent_dim": 8,
            "encoder_filters": [4, 8, 8, 8, 8],
            "decoder_start_len": 5,
            "decoder_channels": 4,
            "decoder_filters": [4, 8, 8, 8],
            "decoder_strides": [1, 2, 1, 1],
            "gru_units": 8,
        }
        vae, decoder, resolved = build_vae(config)
        self.assertEqual(resolved["max_len"], 10)
        encoded = np.zeros((3, 10, 21), dtype=np.float32)
        encoded[:, :, 0] = 1.0
        output = vae.predict(encoded, verbose=0)
        self.assertEqual(output.shape, (3, 10, 21))
        np.testing.assert_allclose(output.sum(axis=-1), np.ones((3, 10)), rtol=1e-5)
        latent = sample_latent(3, 8, seed=1)
        self.assertEqual(latent.shape, (3, 8))
        decoded = decoder.predict(latent, verbose=0)
        self.assertEqual(decoded.shape, (3, 10, 21))

    def test_single_training_step(self):
        """A one-step fit catches loss-shape errors that build/forward misses."""
        from amppipeline.vae import build_vae

        config = {
            "max_len": 10,
            "latent_dim": 8,
            "encoder_filters": [4, 8, 8, 8, 8],
            "decoder_start_len": 5,
            "decoder_channels": 4,
            "decoder_filters": [4, 8, 8, 8],
            "decoder_strides": [1, 2, 1, 1],
            "gru_units": 8,
        }
        vae, _, _ = build_vae(config)
        encoded = np.zeros((2, 10, 21), dtype=np.float32)
        encoded[:, :, 0] = 1.0
        history = vae.fit(encoded, epochs=1, batch_size=2, verbose=0)
        self.assertIn("loss", history.history)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

    def test_mismatched_decoder_geometry_raises(self):
        from amppipeline.vae import build_vae

        with self.assertRaises(ValueError):
            build_vae(
                {
                    "max_len": 10,
                    "decoder_start_len": 4,
                    "decoder_strides": [1, 2, 1, 1],
                }
            )


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is not installed")
class PredictorTests(unittest.TestCase):
    def test_build_and_forward(self):
        from amppipeline.predictor import build_predictor

        model, config = build_predictor(
            {"max_len": 10, "conv_filters": 4, "conv_layers": 2, "lstm_units": 8}
        )
        self.assertEqual(config["conv_filters"], 4)
        encoded = np.zeros((2, 10, 21), dtype=np.float32)
        encoded[:, :, 0] = 1.0
        output = model.predict(encoded, verbose=0)
        self.assertEqual(output.shape, (2, 1))


class TensorFlowAbsentTests(unittest.TestCase):
    @unittest.skipIf(HAS_TENSORFLOW, "TensorFlow is installed")
    def test_import_error_is_informative(self):
        from amppipeline.predictor import build_predictor

        with self.assertRaises(ImportError) as context:
            build_predictor()
        self.assertIn("TensorFlow is required", str(context.exception))


if __name__ == "__main__":
    unittest.main()
