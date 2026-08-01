import sys
import tempfile
import unittest
from pathlib import Path
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


class _BFloatTensor:
    def __init__(self):
        self.casted = False

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        self.casted = True
        return self

    def numpy(self):
        if not self.casted:
            raise TypeError("Got unsupported ScalarType BFloat16")
        return np.array([0.9], dtype=np.float32)


class Sam3PrecisionTest(unittest.TestCase):
    def test_sam3d_helper_uses_configured_repo_not_jupyter_notebook(self):
        with tempfile.TemporaryDirectory() as root:
            notebook = Path(root) / "notebook"
            notebook.mkdir()
            (notebook / "utils.py").write_text(
                "def setup_sam_3d_body(**kwargs):\n"
                "    return kwargs\n",
                encoding="utf-8",
            )
            wrong_notebook = SimpleNamespace(setup_sam_3d_body=lambda **_: "wrong")
            with patch.object(gpu, "SAM3D_DIR", root), \
                 patch.dict(sys.modules, {"notebook.utils": wrong_notebook}):
                setup = gpu._load_sam3d_setup()

        self.assertEqual(setup(hf_repo_id="expected")["hf_repo_id"], "expected")

    def test_bfloat16_output_is_cast_before_numpy_conversion(self):
        output = gpu._to_numpy(_BFloatTensor())

        self.assertEqual(output.dtype, np.float32)
        self.assertAlmostEqual(float(output[0]), 0.9, places=5)

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
