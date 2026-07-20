import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/tools/prepare_airvln16.py"


class AirVlnPreparationTest(unittest.TestCase):
    def test_converter_writes_released_trajectory_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            output_root = temp_root / "output"
            split_root = temp_root / "splits"
            trajectory_id = "trajectory"
            trajectory_name = f"{trajectory_id}_processed"

            waypoint_root = source_root / "waypoints"
            rgb_root = source_root / "traj_obs" / trajectory_id / "rgb"
            depth_root = source_root / "traj_obs" / trajectory_id / "dep"
            waypoint_root.mkdir(parents=True)
            rgb_root.mkdir(parents=True)
            depth_root.mkdir(parents=True)
            (split_root / "train").mkdir(parents=True)
            (split_root / "test").mkdir(parents=True)

            reference_path = [
                [float(frame), 1.0, 2.0, 0.0, 0.0, 0.0] for frame in range(13)
            ]
            payload = {
                "episodes": [
                    {
                        "scene_id": 16,
                        "trajectory_id": trajectory_id,
                        "reference_path": reference_path,
                    }
                ]
            }
            (waypoint_root / "train.json").write_text(json.dumps(payload))
            (waypoint_root / "val_seen.json").write_text('{"episodes": []}')
            (split_root / "train/traj_names.txt").write_text(f"{trajectory_name}\n")
            (split_root / "test/traj_names.txt").write_text("")

            for frame in (11, 12):
                (rgb_root / f"rgb_obs_front_{frame}.png").write_bytes(b"rgb")
                np.save(
                    depth_root / f"dep_obs_front_{frame}.npy",
                    np.full((2, 3, 1), frame, dtype=np.float32),
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-root",
                    str(source_root),
                    "--output-root",
                    str(output_root),
                    "--split-root",
                    str(split_root),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            trajectory_root = output_root / trajectory_name
            with (trajectory_root / "traj_data.pkl").open("rb") as handle:
                trajectory = pickle.load(handle)

            self.assertEqual(trajectory["position"].shape, (2, 2))
            self.assertEqual(trajectory["point"].shape, (2, 3))
            self.assertEqual(trajectory["depth"].shape, (2, 2, 3))
            self.assertEqual(trajectory["pose"].shape, (2, 4, 4))
            np.testing.assert_array_equal(trajectory["timestamps"], [11.0, 12.0])
            np.testing.assert_array_equal(trajectory["images"], ["0", "1"])
            np.testing.assert_allclose(trajectory["K"][0, 0], 256.0)
            self.assertEqual((trajectory_root / "0.jpg").read_bytes(), b"rgb")


if __name__ == "__main__":
    unittest.main()
