import logging
import os
import sys
import tempfile
import unittest

import app_logging


class TestAppLogging(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.old_handlers = list(self.root.handlers)
        self.old_level = self.root.level
        self.old_configured = app_logging._configured
        self.old_hook = sys.excepthook
        app_logging._configured = False

    def tearDown(self):
        for h in list(self.root.handlers):
            if h not in self.old_handlers:
                h.close()
                self.root.removeHandler(h)
        self.root.setLevel(self.old_level)
        app_logging._configured = self.old_configured
        app_logging.set_error_reporter(None)
        sys.excepthook = self.old_hook

    def test_writes_to_log_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "logs", "miralink_debug.log")
            self.assertEqual(app_logging.setup_logging(path, install_hooks=False), path)
            logging.getLogger("calibration_ui").info("Clic en 'Iniciar Calibración'")
            for h in self.root.handlers:
                h.flush()
            with open(path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("MiraLink iniciado", text)
            self.assertIn("Clic en 'Iniciar Calibración'", text)
            for h in list(self.root.handlers):
                if h not in self.old_handlers:
                    h.close()
                    self.root.removeHandler(h)

    def test_uncaught_exception_is_reported_visibly(self):
        shown = []
        app_logging.set_error_reporter(lambda title, detail: shown.append((title, detail)))
        try:
            raise ValueError("fallo en un slot")
        except ValueError:
            info = sys.exc_info()
        stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            app_logging._excepthook(*info)
        finally:
            sys.stderr.close()
            sys.stderr = stderr
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown[0][0], "Error inesperado")
        self.assertIn("fallo en un slot", shown[0][1])
        self.assertIn("miralink_debug.log", shown[0][1])


if __name__ == "__main__":
    unittest.main()
