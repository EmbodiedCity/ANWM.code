import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTest(unittest.TestCase):
    def test_entry_points_expose_help_without_loading_gpu_dependencies(self):
        with tempfile.TemporaryDirectory() as matplotlib_dir:
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = matplotlib_dir
            for script in ("train.py", "infer.py", "evaluate.py", "planning_eval.py"):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / script), "--help"],
                        cwd=ROOT,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
