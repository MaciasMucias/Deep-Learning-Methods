import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from project3_catgen.dcgan_trainer import DCGANTrainer
from project3_catgen.generation import interpolate_latents
from project3_catgen.models import Discriminator, Generator


class DCGANModelTests(unittest.TestCase):
    def test_model_shapes_for_planned_resolutions(self) -> None:
        for image_size in (64, 128):
            with self.subTest(image_size=image_size):
                generator = Generator(
                    latent_dim=16,
                    feature_maps=8,
                    image_size=image_size,
                )
                discriminator = Discriminator(
                    feature_maps=8,
                    image_size=image_size,
                )
                images = generator(torch.randn(2, 16, 1, 1))
                logits = discriminator(images)
                self.assertEqual(images.shape, (2, 3, image_size, image_size))
                self.assertEqual(logits.shape, (2,))
                self.assertLessEqual(images.max().item(), 1.0)
                self.assertGreaterEqual(images.min().item(), -1.0)

    def test_interpolation_keeps_endpoints(self) -> None:
        start = torch.randn(8, 1, 1)
        end = torch.randn(8, 1, 1)
        latents, alphas = interpolate_latents(start, end, steps=10)
        self.assertEqual(latents.shape, (10, 8, 1, 1))
        self.assertTrue(torch.equal(latents[0], start))
        self.assertTrue(torch.equal(latents[-1], end))
        self.assertEqual(alphas.numel(), 10)

    def test_checkpoint_round_trip(self) -> None:
        generator = Generator(latent_dim=8, feature_maps=4, image_size=16)
        discriminator = Discriminator(feature_maps=4, image_size=16)
        generator_optimizer = Adam(generator.parameters(), lr=2e-4)
        discriminator_optimizer = Adam(discriminator.parameters(), lr=2e-4)

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = DCGANTrainer(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                torch.device("cpu"),
                Path(temp_dir),
                latent_dim=8,
                sample_grid_size=4,
            )
            original_noise = trainer.fixed_noise.clone()
            trainer.save_checkpoint("last", epoch=3)

            for parameter in generator.parameters():
                parameter.data.zero_()
            trainer.load_checkpoint("last", restore_rng=False)

            self.assertEqual(trainer.start_epoch, 4)
            self.assertTrue(torch.equal(trainer.fixed_noise, original_noise))
            self.assertTrue(any(parameter.any() for parameter in generator.parameters()))

    def test_single_training_epoch(self) -> None:
        generator = Generator(latent_dim=8, feature_maps=4, image_size=16)
        discriminator = Discriminator(feature_maps=4, image_size=16)
        generator_optimizer = Adam(generator.parameters(), lr=2e-4)
        discriminator_optimizer = Adam(discriminator.parameters(), lr=2e-4)
        dataloader = DataLoader(
            TensorDataset(torch.rand(4, 3, 16, 16) * 2.0 - 1.0),
            batch_size=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = DCGANTrainer(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                torch.device("cpu"),
                Path(temp_dir),
                latent_dim=8,
                sample_grid_size=4,
            )
            metrics = trainer.train_one_epoch(dataloader)

        self.assertGreater(metrics["generator_loss"], 0.0)
        self.assertGreater(metrics["discriminator_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
