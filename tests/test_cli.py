import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from docqa import AppError, main


class CliTest(unittest.TestCase):
    def test_doctor_explains_missing_ollama_without_traceback(self):
        errors = io.StringIO()
        with patch("docqa.check_ollama", side_effect=AppError("Ollama не отвечает")):
            with redirect_stderr(errors):
                code = main(["doctor"])
        self.assertEqual(code, 1)
        self.assertIn("Ollama не отвечает", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_doctor_prints_success_after_model_check(self):
        output = io.StringIO()
        with patch("docqa.check_ollama"):
            with redirect_stdout(output):
                code = main(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("qwen3:1.7b", output.getvalue())
        self.assertIn("embeddinggemma", output.getvalue())


if __name__ == "__main__":
    unittest.main()
