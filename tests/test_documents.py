import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docqa import AppError, scan_documents, split_document


class DocumentsTest(unittest.TestCase):
    def test_scan_uses_supported_files_and_detects_same_size_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.md").write_text("кот\n", encoding="utf-8")
            (root / "skip.pdf").write_bytes(b"ignored")
            (root / "sub").mkdir()
            (root / "sub" / "b.txt").write_text("собака\n", encoding="utf-8")

            files, first = scan_documents(root)
            self.assertEqual([path for path, _ in files], ["a.md", "sub/b.txt"])
            (root / "a.md").write_text("пес\n", encoding="utf-8")
            _, second = scan_documents(root)
            self.assertNotEqual(first, second)

    def test_split_keeps_real_line_numbers_and_caps_long_lines(self):
        chunks = split_document("notes/a.md", "первый\nвторой\nтретий", max_chars=15)
        self.assertEqual(
            [(c["text"], c["start_line"], c["end_line"]) for c in chunks],
            [("первый\nвторой", 1, 2), ("третий", 3, 3)],
        )
        long_line = split_document("x.txt", "abcdefghij", max_chars=4)
        self.assertEqual([c["text"] for c in long_line], ["abcd", "efgh", "ij"])
        self.assertEqual([(c["start_line"], c["end_line"]) for c in long_line], [(1, 1)] * 3)

    def test_scan_rejects_more_than_one_megabyte(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "large.txt").write_bytes(b"a" * 1_000_001)
            with self.assertRaisesRegex(AppError, "1 МБ"):
                scan_documents(root)

    def test_markdown_sections_keep_their_own_source_ranges(self):
        text = "# Памятка\n\nВступление.\n\n## Цена\n\n500 рублей.\n\n## Адрес\n\nУлица Лесная.\n"
        chunks = split_document("example.md", text)
        self.assertEqual(
            [(c["start_line"], c["end_line"]) for c in chunks],
            [(1, 4), (5, 8), (9, 11)],
        )
        self.assertIn("500 рублей", chunks[1]["text"])

    def test_unreadable_file_has_a_human_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "secret.md").write_text("Текст", encoding="utf-8")
            with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(AppError, "Не удалось прочитать.*secret.md"):
                    scan_documents(root)


if __name__ == "__main__":
    unittest.main()
