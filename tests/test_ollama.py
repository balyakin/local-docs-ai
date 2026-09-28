import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from docqa import AppError, check_ollama, embed_texts


def response(body):
    return io.BytesIO(json.dumps(body).encode("utf-8"))


class OllamaTest(unittest.TestCase):
    def test_check_requires_both_models(self):
        def respond(request, timeout):
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/tags")
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(timeout, 180)
            return response({"models": [{"name": "embeddinggemma:latest"}]})

        with patch("docqa.urlopen", side_effect=respond):
            with self.assertRaisesRegex(AppError, "qwen3:1.7b"):
                check_ollama()

    def test_embed_sends_batch_without_truncation(self):
        def respond(request, timeout):
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/embed")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(json.loads(request.data), {
                "model": "embeddinggemma", "input": ["кот", "собака"], "truncate": False
            })
            return response({"model": "embeddinggemma", "embeddings": [[1.0, 0.0], [0.0, 1.0]]})

        with patch("docqa.urlopen", side_effect=respond):
            vectors = embed_texts(["кот", "собака"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])

    def test_embed_reports_invalid_server_response(self):
        with patch("docqa.urlopen", return_value=response({"embeddings": []})):
            with self.assertRaisesRegex(AppError, "ответ Ollama"):
                embed_texts(["текст"])

    def test_connection_error_explains_what_to_start(self):
        with patch("docqa.urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(AppError, "Ollama не отвечает"):
                check_ollama()


if __name__ == "__main__":
    unittest.main()
