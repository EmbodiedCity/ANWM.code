import pickle
import os
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from anwm.config import load_runtime_config
from anwm.model import CDiT_models


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTest(unittest.TestCase):
    def test_anwm_model_parameter_count(self):
        with torch.device("meta"):
            model = CDiT_models["CDiT-XL/2"](
                context_size=16,
                input_size=28,
                in_channels=4,
            )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()), 1_134_100_256
        )

    def test_anwm_configs_are_consistent(self):
        model_config = yaml.safe_load((ROOT / "config/anwm.yaml").read_text())
        eval_config = yaml.safe_load((ROOT / "config/eval_config.yaml").read_text())
        data_config = yaml.safe_load((ROOT / "config/data_config.yaml").read_text())
        real_eval_config = yaml.safe_load(
            (ROOT / "real/config/eval_config.yaml").read_text()
        )
        real_data_config = yaml.safe_load(
            (ROOT / "real/config/data_config.yaml").read_text()
        )

        self.assertEqual(model_config["run_name"], "anwm_cdit_airvln")
        self.assertEqual(model_config["context_size"], 16)
        self.assertEqual(eval_config["eval_context_size"], 16)
        self.assertNotIn("sekai_new", data_config)
        self.assertNotIn("sekai_new", eval_config["eval_datasets"])
        self.assertEqual(real_eval_config["trajectory_eval_context_size"], 16)
        self.assertEqual(
            real_data_config["sekai_new"]["metric_waypoint_spacing"], 0.025
        )

    def test_sekai_candidates_match_evaluation_index(self):
        split_dir = ROOT / "real/data_splits/sekai_new/test"
        with (split_dir / "trajectory_candidates.pkl").open("rb") as handle:
            candidates = pickle.load(handle)
        with (split_dir / "navigation_eval.pkl").open("rb") as handle:
            eval_index = pickle.load(handle)

        self.assertEqual(sorted(candidates), list(range(len(eval_index))))
        self.assertTrue(
            all(
                len(sample_candidates) == 5 for sample_candidates in candidates.values()
            )
        )

    def test_config_paths_do_not_depend_on_working_directory(self):
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                config = load_runtime_config("config/anwm.yaml")
        finally:
            os.chdir(original_cwd)

        self.assertEqual(Path(config["results_dir"]), ROOT / "logs")
        self.assertEqual(
            Path(config["datasets"]["airvln_16"]["data_folder"]),
            ROOT / "data/airvln_16",
        )


if __name__ == "__main__":
    unittest.main()
