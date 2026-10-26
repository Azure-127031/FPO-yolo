"""CLI regression checks; these do not execute a detector or require PyTorch."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"fpo_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntryPointTests(unittest.TestCase):
    def invoke(self, name, args):
        yolo = Mock()
        fake = types.ModuleType("ultralytics")
        fake.YOLO = yolo
        with patch.dict(sys.modules, {"ultralytics": fake}), patch.object(sys, "argv", [name, *args]):
            load_script(name).main()
        return yolo

    def test_help_does_not_import_model_dependencies(self):
        for name in ("train", "val", "check_model"):
            with self.subTest(script=name):
                # A blocked model import makes this check independent of the host environment.
                source = ("import sys, runpy; sys.modules['ultralytics'] = None; "
                          f"sys.argv = [{str(SCRIPTS / (name + '.py'))!r}, '--help']; "
                          f"runpy.run_path(sys.argv[0], run_name='__main__')")
                result = subprocess.run([sys.executable, "-B", "-c", source], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_validation_defaults_are_preserved(self):
        yolo = self.invoke("val", ["--weights", "best.pt"])
        yolo.assert_called_once_with("best.pt")
        options = yolo.return_value.val.call_args.kwargs
        self.assertEqual((options["batch"], options["half"], options["split"]), (16, True, "val"))
        self.assertEqual((options["conf"], options["iou"], options["max_det"]), (0.001, 0.7, 300))

    def test_aitod_evaluation_options_reach_validator(self):
        options = self.invoke("val", ["--weights", "best.pt", "--data", "aitod.yaml", "--split", "test",
                                      "--batch", "15", "--precision", "fp32", "--save-json"]).return_value.val.call_args.kwargs
        self.assertEqual((options["batch"], options["half"], options["split"]), (15, False, "test"))
        self.assertEqual(options["data"], "aitod.yaml")
        self.assertTrue(options["save_json"])

    def test_invalid_batch_is_rejected_before_import(self):
        for name in ("train", "val"):
            for batch in ("0", "-1"):
                with self.subTest(script=name, batch=batch), patch.dict(sys.modules, {"ultralytics": None}), \
                     patch.object(sys, "argv", [name, "--batch", batch] + (["--weights", "best.pt"] if name == "val" else [])), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    load_script(name).main()
                self.assertEqual(error.exception.code, 2)

    def test_training_defaults_and_explicit_override(self):
        for batch in (None, 8):
            with self.subTest(batch=batch):
                options = self.invoke("train", [] if batch is None else ["--batch", str(batch)]).return_value.train.call_args.kwargs
                self.assertEqual(options["batch"], 16 if batch is None else batch)
                self.assertEqual((options["optimizer"], options["epochs"], options["seed"]), ("SGD", 200, 0))

    def test_model_check_keeps_requested_scale(self):
        yolo = self.invoke("check_model", ["--scale", "n"])
        yolo.assert_called_once_with("yolo11n-fpo.yaml")
        yolo.return_value.info.assert_called_once_with(detailed=False, verbose=True, imgsz=640)


if __name__ == "__main__":
    unittest.main()
