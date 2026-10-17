"""Rutas del ejecutable: datos junto al .exe, recursos empaquetados y config.json creado al arrancar."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import app_paths
import main
from config_manager import DEFAULT_CONFIG, _get_project_root, load_config


class TestDevelopmentPaths(unittest.TestCase):
    def test_project_root_in_development(self):
        self.assertFalse(app_paths.is_frozen())
        self.assertEqual(app_paths.data_dir(), app_paths.PROJECT_ROOT)
        self.assertTrue(os.path.isdir(app_paths.resource_path("styles")))
        self.assertEqual(_get_project_root(), app_paths.PROJECT_ROOT)  # mismo comportamiento que antes


class TestFrozenPaths(unittest.TestCase):
    def frozen(self, exe_dir, meipass):
        return mock.patch.multiple(
            sys, frozen=True, executable=os.path.join(exe_dir, "MiraLink.exe"), _MEIPASS=meipass, create=True
        )

    def test_data_next_to_exe_and_resources_in_bundle(self):
        import app_logging
        import calibration
        import eye_tracker
        import gaze_session
        import theme_manager

        with tempfile.TemporaryDirectory() as exe_dir, tempfile.TemporaryDirectory() as bundle:
            with self.frozen(exe_dir, bundle):
                self.assertTrue(app_paths.is_frozen())
                self.assertEqual(app_paths.data_dir(), exe_dir)
                self.assertEqual(_get_project_root(), exe_dir)  # config.json junto al .exe
                self.assertEqual(calibration.default_calibration_path(), os.path.join(exe_dir, "calibration.json"))
                self.assertEqual(gaze_session.default_sessions_dir(), os.path.join(exe_dir, "sessions"))
                self.assertEqual(theme_manager._get_styles_dir(), os.path.join(bundle, "styles"))
                self.assertEqual(eye_tracker.default_model_path(), os.path.join(bundle, "models", "face_landmarker.task"))
        # el log se fija al importar app_logging: en desarrollo, logs/ del proyecto
        self.assertEqual(app_logging.LOG_DIR, os.path.join(app_paths.PROJECT_ROOT, "logs"))

    def test_double_click_opens_config_window(self):
        self.assertEqual(main.resolve_args([], frozen=True), ["--config"])
        self.assertEqual(main.resolve_args(["--dry-run"], frozen=True), ["--dry-run", "--config"])
        for mode in ("--record-session", "--self-test", "--check", "--tracker-debug", "--config"):
            self.assertEqual(main.resolve_args([mode], frozen=True), [mode])
        self.assertEqual(main.resolve_args([], frozen=False), [])  # python main.py: verificación del entorno


class TestEnsureConfigFile(unittest.TestCase):
    def test_created_with_defaults_once_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            self.assertEqual(main.ensure_config_file(path), (path, True))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["interaction"], DEFAULT_CONFIG["interaction"])
            data["interaction"]["smoothing_percent"] = 77  # el usuario lo edita
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.assertEqual(main.ensure_config_file(path), (path, False))
            self.assertEqual(load_config(path)["interaction"]["smoothing_percent"], 77)

    def test_streams_are_replaced_when_missing(self):
        with mock.patch.object(sys, "stdout", None), mock.patch.object(sys, "stderr", None):
            main.ensure_streams()
            print("no falla sin consola")
            out, err = sys.stdout, sys.stderr
        out.close()
        err.close()


if __name__ == "__main__":
    unittest.main()
