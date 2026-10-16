import os
import tempfile
import unittest

from config_manager import (
    DEFAULT_CONFIG,
    load_config,
    save_config,
)


class TestConfigManager(unittest.TestCase):
    def test_defaults_if_file_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cfg = load_config(path)
            self.assertEqual(cfg["version"], DEFAULT_CONFIG["version"])
            self.assertEqual(cfg["camera"]["device_index"], DEFAULT_CONFIG["camera"]["device_index"])
            self.assertEqual(cfg["interaction"]["dwell_hold_ms"], DEFAULT_CONFIG["interaction"]["dwell_hold_ms"])
            self.assertEqual(cfg["interaction"]["blink_click_ms"], DEFAULT_CONFIG["interaction"]["blink_click_ms"])
            self.assertEqual(cfg["interaction"]["blink_release_ms"], DEFAULT_CONFIG["interaction"]["blink_release_ms"])
            self.assertEqual(cfg["interaction"]["smoothing_percent"], DEFAULT_CONFIG["interaction"]["smoothing_percent"])
            self.assertEqual(cfg["ui"]["theme"], DEFAULT_CONFIG["ui"]["theme"])
            self.assertAlmostEqual(cfg["ui"]["font_scale"], DEFAULT_CONFIG["ui"]["font_scale"], places=2)

    def test_merge_with_partial_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{ "ui": { "theme": "alto_contraste_1" }, "interaction": { "smoothing_percent": 75 } }')

            cfg = load_config(path)
            self.assertEqual(cfg["ui"]["theme"], "alto_contraste_1")
            self.assertEqual(cfg["interaction"]["smoothing_percent"], 75)
            self.assertEqual(cfg["interaction"]["dwell_hold_ms"], DEFAULT_CONFIG["interaction"]["dwell_hold_ms"])

    def test_clamp_out_of_range(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    '{ "interaction": { "dwell_hold_ms": 5000, "blink_click_ms": 10, "blink_release_ms": 1000, "smoothing_percent": 200 }, '
                    '"ui": { "font_scale": 5.0 }, "camera": { "device_index": -3 } }'
                )

            cfg = load_config(path)
            self.assertEqual(cfg["interaction"]["dwell_hold_ms"], 3000)  # clamp to max
            self.assertEqual(cfg["interaction"]["blink_click_ms"], 150)  # clamp to min
            self.assertEqual(cfg["interaction"]["blink_release_ms"], 500)  # clamp to max
            self.assertEqual(cfg["interaction"]["smoothing_percent"], 100)  # clamp to max
            self.assertAlmostEqual(cfg["ui"]["font_scale"], 2.0, places=2)  # clamp to max
            self.assertEqual(cfg["camera"]["device_index"], 0)  # >= 0

    def test_invalid_json_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{ invalid json here')

            cfg = load_config(path)
            self.assertEqual(cfg["interaction"]["dwell_hold_ms"], DEFAULT_CONFIG["interaction"]["dwell_hold_ms"])
            self.assertEqual(cfg["ui"]["theme"], DEFAULT_CONFIG["ui"]["theme"])

    def test_roundtrip_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cfg = load_config(path)
            cfg["ui"]["theme"] = "alto_contraste_2"
            cfg["interaction"]["smoothing_percent"] = 60
            cfg["interaction"]["dwell_hold_ms"] = 1500
            cfg["interaction"]["blink_click_ms"] = 200
            cfg["interaction"]["blink_release_ms"] = 250
            cfg["ui"]["font_scale"] = 1.5

            saved = save_config(cfg, path)
            loaded = load_config(path)
            self.assertEqual(loaded["ui"]["theme"], "alto_contraste_2")
            self.assertEqual(loaded["interaction"]["smoothing_percent"], 60)
            self.assertEqual(loaded["interaction"]["dwell_hold_ms"], 1500)
            self.assertEqual(loaded["interaction"]["blink_click_ms"], 200)
            self.assertEqual(loaded["interaction"]["blink_release_ms"], 250)
            self.assertAlmostEqual(loaded["ui"]["font_scale"], 1.5, places=2)


class TestAdvancedInteractionKeys(unittest.TestCase):
    def test_defaults_and_clamps(self):
        from config_manager import _deep_merge, _validate_config

        inter = DEFAULT_CONFIG["interaction"]
        self.assertEqual(inter["dwell_radius_px"], 50)
        self.assertTrue(inter["hold_enabled"])
        self.assertEqual(inter["blink_max_ms"], 2000)
        self.assertEqual(inter["freeze_after_blink_ms"], 150)
        self.assertEqual(inter["hold_timeout_s"], 30)

        low = _validate_config(_deep_merge(DEFAULT_CONFIG, {"interaction": {"dwell_radius_px": 1, "blink_max_ms": 1, "freeze_after_blink_ms": -5, "hold_timeout_s": 0, "hold_enabled": "x"}}))
        self.assertEqual(low["interaction"]["dwell_radius_px"], 20)
        self.assertEqual(low["interaction"]["blink_max_ms"], 500)
        self.assertEqual(low["interaction"]["freeze_after_blink_ms"], 0)
        self.assertEqual(low["interaction"]["hold_timeout_s"], 5)
        self.assertTrue(low["interaction"]["hold_enabled"])
        high = _validate_config(_deep_merge(DEFAULT_CONFIG, {"interaction": {"dwell_radius_px": 999, "blink_max_ms": 99999, "freeze_after_blink_ms": 999, "hold_timeout_s": 999, "hold_enabled": False}}))
        self.assertEqual(high["interaction"]["dwell_radius_px"], 150)
        self.assertEqual(high["interaction"]["blink_max_ms"], 5000)
        self.assertEqual(high["interaction"]["freeze_after_blink_ms"], 500)
        self.assertEqual(high["interaction"]["hold_timeout_s"], 120)
        self.assertFalse(high["interaction"]["hold_enabled"])
