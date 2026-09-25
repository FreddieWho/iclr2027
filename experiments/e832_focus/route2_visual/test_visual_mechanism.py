import inspect
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_mechanism as v


class ConstantBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.backbone = lambda x: torch.ones((x.shape[0], 1, 1, 1), device=x.device) + self.anchor


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = v.Mechanism("additive", input_size=224).eval()
        self.images = v.render_smoke(n=2)[0]

    def test_shape_native64_resized_to_configured_224(self):
        self.assertEqual(tuple(self.images.shape[1:]), (3, 64, 64))
        self.assertEqual(tuple(v.prep(self.images).shape), (2, 3, 224, 224))
        for mode in ("direct", "additive", "interaction", "representation"):
            model = v.Mechanism(mode, input_size=224).eval()
            with torch.no_grad():
                logits = model(self.images)
            self.assertEqual(tuple(logits.shape), (2,))
            self.assertTrue(torch.isfinite(logits).all())

    def test_four_head_contracts_are_distinct(self):
        features = {
            "global": torch.full((1, 512), 2.0),
            "red": torch.zeros((1, 512)),
            "blue": torch.zeros((1, 512)),
        }
        features["red"][0, 0] = 1.0
        features["blue"][0, 0] = 3.0
        direct = v.Mechanism("direct").eval()
        additive = v.Mechanism("additive").eval()
        representation = v.Mechanism("representation").eval()
        interaction = v.Mechanism("interaction").eval()
        with torch.no_grad():
            direct.head.weight.zero_(); direct.head.bias.zero_(); direct.head.weight[0, 0] = 1
            additive.head.weight.zero_(); additive.head.bias.zero_(); additive.head.weight[0, 0] = 1
            representation.head.weight.zero_(); representation.head.bias.zero_(); representation.head.weight[0, 0] = 1
            with mock.patch.object(v, "visible_segment_features", return_value=features):
                direct_logit = direct(self.images[:1])
                additive_logit = additive(self.images[:1])
                representation_logit = representation(self.images[:1])
                interaction_logit = interaction(self.images[:1])
        self.assertAlmostEqual(float(direct_logit), 2.0, places=5)
        self.assertAlmostEqual(float(additive_logit), 4.0, places=5)
        self.assertAlmostEqual(float(representation_logit), 1.0, places=5)
        self.assertEqual(tuple(interaction.head[0].weight.shape), (128, 1024))
        self.assertNotEqual(float(interaction_logit), float(representation_logit))

    def test_inference_signature_has_no_hidden_data(self):
        signature = inspect.signature(v.Mechanism.forward)
        self.assertEqual(list(signature.parameters), ["self", "images"])
        source = inspect.getsource(v.Mechanism.forward).lower()
        for forbidden in ("coord", "index", "oracle", "answer", "combination"):
            self.assertNotIn(forbidden, source)
        signature = inspect.signature(v.visible_segment_features)
        self.assertEqual(list(signature.parameters), ["model", "images", "mask_pool"])
        self.assertEqual(signature.parameters["mask_pool"].default, "nearest")

    def test_area_pool_preserves_visible_mass(self):
        thin = np.zeros((1, 3, 64, 64), np.float32)
        thin[:, :, 4:9, 8:57] = np.asarray([.9, .1, .1])[:, None, None]
        fake = ConstantBackbone().eval()
        with torch.no_grad():
            near = v.visible_segment_features(fake, thin, "nearest")
            area = v.visible_segment_features(fake, thin, "area")
        self.assertTrue(bool((near["red"] == 0).all()))
        self.assertGreater(float(area["red"].abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            v.visible_segment_features(fake, thin, "bogus")

    def test_missing_visible_color_uses_zero_mask(self):
        blank = np.zeros_like(self.images)
        fake = ConstantBackbone().eval()
        with torch.no_grad():
            features = v.visible_segment_features(fake, blank)
        self.assertTrue(torch.equal(features["red"], torch.zeros((2, 1))))
        self.assertTrue(torch.equal(features["blue"], torch.zeros((2, 1))))
        self.assertTrue(torch.equal(features["global"], torch.ones((2, 1))))

    def test_raw_tensor_input_path(self):
        model = v.Mechanism("interaction", input_size=224).eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(self.images))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA tensor-path regression")
    def test_cuda_tensor_input_path(self):
        model = v.Mechanism("interaction", input_size=224).cuda().eval()
        images = torch.from_numpy(self.images).cuda()
        with torch.no_grad():
            logits = model(images)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_renderer_red_blue_pixels_select_expected_visible_masks(self):
        red = np.zeros((1, 3, 64, 64), np.float32)
        blue = np.zeros_like(red)
        red[:, :, 20:40, 20:40] = np.asarray([.9, .1, .1])[:, None, None]
        blue[:, :, 20:40, 20:40] = np.asarray([.1, .1, .9])[:, None, None]
        with torch.no_grad():
            _, _ = v._image_nchw(red)
            raw, _ = v._image_nchw(red)
            red_mask = (raw[:, 0] > .15) & (raw[:, 1] < .35) & (raw[:, 2] < .35)
            raw_blue, _ = v._image_nchw(blue)
            blue_mask = (raw_blue[:, 2] > .15) & (raw_blue[:, 0] < .35) & (raw_blue[:, 1] < .35)
        self.assertEqual(int(red_mask.sum()), 20 * 20)
        self.assertEqual(int(blue_mask.sum()), 20 * 20)


if __name__ == "__main__":
    unittest.main()
