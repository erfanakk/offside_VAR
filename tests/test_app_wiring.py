import unittest
from pathlib import Path


class SliderWiringTest(unittest.TestCase):
    def test_scrub_ignores_programmatic_slider_updates(self):
        source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")

        self.assertIn("frame_slider.input(on_scrub", source)
        self.assertNotIn("frame_slider.change(on_scrub", source)
        self.assertEqual(source.count("step_and_scrub(p, i,"), 2)


if __name__ == "__main__":
    unittest.main()
