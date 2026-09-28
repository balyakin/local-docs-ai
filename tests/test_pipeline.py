import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from docqa import AppError, answer_question, build_index, find_chunks, load_index, scan_documents


def response(body):
    return io.BytesIO(json.dumps(body).encode("utf-8"))


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "documents"
        self.docs.mkdir()
        (self.docs / "cats.md").write_text("Коты мурлыкают.\n", encoding="utf-8")
        (self.docs / "dogs.txt").write_text("Собаки лают.\n", encoding="utf-8")
        self.index_file = self.root / "data" / "index.json"
        self.chat_payload = None

    def ollama(self, request, timeout):
        self.assertEqual(timeout, 180)
        payload = json.loads(request.data)
        if request.full_url.endswith("/api/embed"):
            vectors = []
            for text in payload["input"]:
                vectors.append([1.0, 0.0] if "Коты" in text or "мурлыкает" in text else [0.0, 1.0])
            return response({"model": "embeddinggemma", "embeddings": vectors})
        if request.full_url.endswith("/api/chat"):
            self.chat_payload = payload
            return response({"message": {"role": "assistant", "content": "Коты [1]."}})
        self.fail(f"Unexpected request: {request.full_url}")

    def test_index_search_and_answer_show_real_source_lines(self):
        with patch("docqa.urlopen", side_effect=self.ollama):
            index = build_index(self.docs, self.index_file)
            self.assertEqual(len(index["chunks"]), 2)
            self.assertTrue(self.index_file.is_file())
            hits = find_chunks("Кто мурлыкает?", load_index(self.docs, self.index_file), limit=1)
            self.assertEqual([(h["path"], h["start_line"], h["end_line"]) for h in hits],
                             [("cats.md", 1, 1)])
            answer, sources = answer_question("Кто мурлыкает?", self.docs, self.index_file)

        self.assertEqual(answer, "Коты [1].")
        self.assertEqual(sources[0]["path"], "cats.md")
        self.assertIn("Коты мурлыкают.", sources[0]["text"])
        self.assertEqual(self.chat_payload["model"], "qwen3:1.7b")
        self.assertFalse(self.chat_payload["stream"])
        self.assertIn("cats.md:1-1", self.chat_payload["messages"][1]["content"])

    def test_changed_documents_require_reindex(self):
        with patch("docqa.urlopen", side_effect=self.ollama):
            build_index(self.docs, self.index_file)
            (self.docs / "cats.md").write_text("Коты мяукают.\n", encoding="utf-8")
            with self.assertRaisesRegex(AppError, "Обновите индекс"):
                load_index(self.docs, self.index_file)

    def test_empty_question_and_empty_documents_are_explained(self):
        (self.root / "empty").mkdir()
        with self.assertRaisesRegex(AppError, "нет.*md.*txt"):
            build_index(self.root / "empty", self.index_file)
        with patch("docqa.urlopen", side_effect=self.ollama):
            build_index(self.docs, self.index_file)
            with self.assertRaisesRegex(AppError, "Напишите вопрос"):
                answer_question("   ", self.docs, self.index_file)

    def test_valid_json_with_missing_vectors_is_rejected_before_search(self):
        _, fingerprint = scan_documents(self.docs)
        self.index_file.parent.mkdir()
        self.index_file.write_text(json.dumps({
            "version": 1, "embedding_model": "embeddinggemma", "fingerprint": fingerprint,
            "chunks": [{"path": "cats.md", "start_line": 1, "end_line": 1, "text": "Коты"}],
        }), encoding="utf-8")
        with self.assertRaisesRegex(AppError, "Индекс повреждён"):
            load_index(self.docs, self.index_file)

    def test_empty_model_answer_is_explained(self):
        def no_chat_text(request, timeout):
            if request.full_url.endswith("/api/chat"):
                return response({"message": {"role": "assistant", "content": ""}})
            return self.ollama(request, timeout)

        with patch("docqa.urlopen", side_effect=no_chat_text):
            build_index(self.docs, self.index_file)
            with self.assertRaisesRegex(AppError, "пустой ответ"):
                answer_question("Кто мурлыкает?", self.docs, self.index_file)

    def test_failed_reindex_keeps_last_complete_index(self):
        with patch("docqa.urlopen", side_effect=self.ollama):
            build_index(self.docs, self.index_file)
        before = self.index_file.read_bytes()
        with patch("docqa.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(AppError):
                build_index(self.docs, self.index_file)
        self.assertEqual(self.index_file.read_bytes(), before)

    def test_index_write_failure_has_clear_error(self):
        with patch("docqa.urlopen", side_effect=self.ollama):
            with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(AppError, "Не удалось сохранить"):
                    build_index(self.docs, self.index_file)


if __name__ == "__main__":
    unittest.main()
