import os
import tempfile
import unittest

from theme_manager import list_themes, load_theme


class TestThemeManager(unittest.TestCase):
    def test_list_themes_has_three_and_default_first(self):
        themes = list_themes()
        self.assertGreaterEqual(len(themes), 3)
        self.assertEqual(themes[0]["id"], "default")
        ids = [t["id"] for t in themes]
        self.assertIn("alto_contraste_1", ids)
        self.assertIn("alto_contraste_2", ids)

    def test_visible_name_from_first_line(self):
        themes = list_themes()
        for t in themes:
            self.assertIsNotNone(t["name"])
            self.assertTrue(len(t["name"]) > 0)

    def test_load_theme_replaces_tokens_no_remaining(self):
        qss = load_theme("default", 1.0)
        self.assertNotIn("$FONT_SMALL", qss)
        self.assertNotIn("$FONT_BASE", qss)
        self.assertNotIn("$FONT_TITLE", qss)

    def test_load_theme_scale_2_0(self):
        qss = load_theme("default", 2.0)
        self.assertNotIn("$FONT_SMALL", qss)
        self.assertNotIn("$FONT_BASE", qss)
        self.assertNotIn("$FONT_TITLE", qss)
        self.assertIn("22pt", qss)
        self.assertIn("36pt", qss)
        self.assertIn("48px", qss)  # alto mínimo clicable sigue en px

    def test_load_theme_scale_1_0_values(self):
        qss = load_theme("default", 1.0)
        self.assertIn("11pt", qss)
        self.assertIn("18pt", qss)

    def test_font_tokens_never_below_one_point(self):
        qss = load_theme("default", 0.01)
        self.assertNotIn("0pt", qss.replace("10pt", "").replace("20pt", ""))
        self.assertIn("1pt", qss)

    def test_no_px_font_sizes_in_themes(self):
        import re

        for t in list_themes():
            qss = load_theme(t["id"], 1.0)
            self.assertIsNone(re.search(r"font-size:\s*\d+px", qss), t["id"])

    def test_unknown_theme_falls_back_to_default(self):
        qss_unknown = load_theme("no_existe", 1.0)
        qss_default = load_theme("default", 1.0)
        self.assertEqual(qss_unknown, qss_default)

    def test_new_theme_in_temp_dir_appears(self):
        with tempfile.TemporaryDirectory() as d:
            # copy existing themes
            import shutil

            proj_styles = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles")
            for fname in ["default.qss", "alto_contraste_1.qss", "alto_contraste_2.qss"]:
                src = os.path.join(proj_styles, fname)
                dst = os.path.join(d, fname)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            # add new
            with open(os.path.join(d, "nuevo_tema.qss"), "w", encoding="utf-8") as f:
                f.write("/* name: Nuevo Tema */\nQWidget{ font-size: $FONT_BASE; }\n")
            themes = list_themes(d)
            ids = [t["id"] for t in themes]
            self.assertIn("nuevo_tema", ids)
            self.assertEqual(themes[0]["id"], "default")  # default still first
            # load
            qss = load_theme("nuevo_tema", 1.0, styles_dir=d)
            self.assertNotIn("$FONT_BASE", qss)
