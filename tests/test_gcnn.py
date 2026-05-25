"""
Unit tests for Group Equivariant CNN watermarking modules.

Run:
    pytest tests/test_gcnn.py -v
"""

import pytest
import torch

from videoseal.models.g_extractor import build_g_extractor

DEFAULT_MESSAGE_BITS = 32
DEFAULT_STAGE_DEPTHS = [2, 2, 2]
DEFAULT_STAGE_DIMS = [32, 64, 128]
DEFAULT_IMAGE_SIZE = 128
DETECTION_CHANNELS = 1
INVARIANCE_TOLERANCE = 0.05
FLOAT_EQUALITY_TOLERANCE = 1e-6
MINIMUM_MESSAGE_DIFFERENCE = 0.001


def _rotate_90(tensor: torch.Tensor, k: int = 1) -> torch.Tensor:
    return torch.rot90(tensor, k, dims=[-2, -1])


def _build_extractor(group_type: str = "C4") -> torch.nn.Module:
    return build_g_extractor(
        nbits=DEFAULT_MESSAGE_BITS,
        depths=DEFAULT_STAGE_DEPTHS,
        dims=DEFAULT_STAGE_DIMS,
        group_type=group_type,
    )


def _build_embedder():
    try:
        from videoseal.models.g_embedder import build_g_embedder
        import omegaconf
    except ImportError:
        pytest.skip("g_embedder or omegaconf not available")

    config = omegaconf.OmegaConf.create({
        "group_type": "C4",
        "msg_processor": {
            "nbits": DEFAULT_MESSAGE_BITS,
            "hidden_size": DEFAULT_MESSAGE_BITS * 2,
            "msg_processor_type": "binary+concat",
        },
        "unet": {
            "in_channels": 3,
            "out_channels": 3,
            "z_channels": 32,
            "z_channels_mults": [1, 2],
            "num_blocks": 1,
            "last_tanh": True,
        },
    })
    return build_g_embedder(config, nbits=DEFAULT_MESSAGE_BITS, hidden_size_multiplier=2)


def _random_image(batch_size: int = 1, size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    return torch.rand(batch_size, 3, size, size)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestExtractorConstruction:

    @pytest.mark.parametrize("group_type", ["C4", "C8", "D4"])
    def test_builds_for_all_supported_groups(self, group_type):
        extractor = _build_extractor(group_type)
        assert extractor.nbits == DEFAULT_MESSAGE_BITS

    def test_has_trainable_parameters(self):
        extractor = _build_extractor()
        trainable_count = sum(1 for p in extractor.parameters() if p.requires_grad)
        assert trainable_count > 0


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestExtractorForwardPass:

    def test_output_shape_matches_detection_plus_message_bits(self):
        extractor = _build_extractor()
        extractor.eval()
        image = _random_image(batch_size=2)
        expected_output_channels = DETECTION_CHANNELS + DEFAULT_MESSAGE_BITS

        with torch.no_grad():
            output = extractor(image)

        assert output.shape == (2, expected_output_channels)

    @pytest.mark.parametrize("image_size", [64, 128, 256])
    def test_accepts_various_image_sizes(self, image_size):
        extractor = _build_extractor()
        extractor.eval()
        image = _random_image(size=image_size)
        expected_output_channels = DETECTION_CHANNELS + DEFAULT_MESSAGE_BITS

        with torch.no_grad():
            output = extractor(image)

        assert output.shape == (1, expected_output_channels)


# ---------------------------------------------------------------------------
# Rotation invariance
# ---------------------------------------------------------------------------

class TestRotationInvariance:

    @pytest.mark.parametrize("rotation_steps", [1, 2, 3])
    def test_logits_unchanged_after_90_degree_rotations(self, rotation_steps):
        extractor = _build_extractor(group_type="C4")
        extractor.eval()
        torch.manual_seed(42)
        image = _random_image()
        rotated_image = _rotate_90(image, k=rotation_steps)

        with torch.no_grad():
            original_logits = extractor(image)
            rotated_logits = extractor(rotated_image)

        mean_difference = (original_logits - rotated_logits).abs().mean().item()
        assert mean_difference < INVARIANCE_TOLERANCE, (
            f"Rotation by {rotation_steps * 90} degrees changed logits "
            f"by {mean_difference:.4f}"
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_all_parameters_receive_gradients(self):
        extractor = _build_extractor()
        extractor.train()
        image = _random_image()
        image.requires_grad_(True)
        expected_output_channels = DETECTION_CHANNELS + DEFAULT_MESSAGE_BITS
        target = torch.randint(0, 2, (1, expected_output_channels)).float()

        output = extractor(image)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        total_parameters = sum(1 for p in extractor.parameters())
        parameters_with_gradients = sum(
            1 for p in extractor.parameters() if p.grad is not None
        )
        assert parameters_with_gradients == total_parameters


# ---------------------------------------------------------------------------
# Embedder sanity checks
# ---------------------------------------------------------------------------

class TestEmbedderSanity:

    def test_different_messages_produce_different_watermarks(self):
        embedder = _build_embedder()
        embedder.eval()
        image = _random_image()
        all_zeros_message = torch.zeros(1, DEFAULT_MESSAGE_BITS)
        all_ones_message = torch.ones(1, DEFAULT_MESSAGE_BITS)

        with torch.no_grad():
            watermarked_with_zeros = embedder(image, all_zeros_message)
            watermarked_with_ones = embedder(image, all_ones_message)

        mean_difference = (
            (watermarked_with_zeros - watermarked_with_ones).abs().mean().item()
        )
        assert mean_difference > MINIMUM_MESSAGE_DIFFERENCE, (
            f"Different messages produced near-identical outputs "
            f"(diff={mean_difference:.6f})"
        )

    def test_same_input_produces_identical_output(self):
        embedder = _build_embedder()
        embedder.eval()
        torch.manual_seed(0)
        image = _random_image()
        message = torch.randint(0, 2, (1, DEFAULT_MESSAGE_BITS)).float()

        with torch.no_grad():
            first_output = embedder(image, message)
            second_output = embedder(image, message)

        maximum_difference = (first_output - second_output).abs().max().item()
        assert maximum_difference < FLOAT_EQUALITY_TOLERANCE


# ---------------------------------------------------------------------------
# Message group replication
# ---------------------------------------------------------------------------

class TestMessageGroupReplication:

    def test_message_is_identical_across_all_rotation_channels(self):
        embedder = _build_embedder()
        embedder.eval()
        group_order = embedder.gunet.group_order
        hidden_size = embedder.msg_processor.hidden_size

        from escnn import nn as enn

        gspace = embedder.gunet.gspace
        base_channels = 16
        total_channels = base_channels * group_order
        spatial_size = 16
        latent_type = enn.FieldType(
            gspace, base_channels * [gspace.regular_repr]
        )
        latent_tensor = torch.randn(1, total_channels, spatial_size, spatial_size)
        latent = enn.GeometricTensor(latent_tensor, latent_type)
        message = torch.randint(0, 2, (1, DEFAULT_MESSAGE_BITS)).float()

        fused = embedder.msg_processor(latent, message)

        message_channels = fused.tensor[:, total_channels:, :, :]
        reshaped = message_channels.view(
            1, hidden_size, group_order, spatial_size, spatial_size
        )
        reference_rotation = reshaped[:, :, 0, :, :]

        for rotation_index in range(1, group_order):
            current_rotation = reshaped[:, :, rotation_index, :, :]
            maximum_difference = (
                (reference_rotation - current_rotation).abs().max().item()
            )
            assert maximum_difference < FLOAT_EQUALITY_TOLERANCE, (
                f"Message differs between rotation 0 and rotation "
                f"{rotation_index} (max diff={maximum_difference:.8f})"
            )