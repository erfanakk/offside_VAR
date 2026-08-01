import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline import gpu


class _Autocast:
    active = False

    def __enter__(self):
        self.active = True

    def __exit__(self, *_):
        self.active = False


class _Processor:
    def __init__(self, autocast):
        self.autocast = autocast

    def set_image(self, image):
        assert self.autocast.active
        return {"image": image}

    def set_text_prompt(self, state, prompt):
        assert self.autocast.active
        return {"boxes": np.array([[1, 2, 3, 4]]),
                "scores": np.array([0.9]),
                "masks": np.ones((1, 1, 2, 2))}


class Sam3PrecisionTest(unittest.TestCase):
    def test_cuda_inference_uses_bfloat16_autocast(self):
        autocast = _Autocast()
        fake_torch = SimpleNamespace(
            bfloat16=object(),
            cuda=SimpleNamespace(is_available=lambda: True),
            autocast=lambda device, dtype: autocast,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), \
             patch.object(gpu, "_get_sam3", return_value=_Processor(autocast)):
            people = gpu._detect_sam3(np.zeros((2, 2, 3), dtype=np.uint8), 0.5)

        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["bbox"].tolist(), [1, 2, 3, 4])
        self.assertEqual(people[0]["mask"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
