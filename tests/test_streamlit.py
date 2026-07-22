import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"


class StreamlitEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.created_chunks = not CHUNKS_FILE.exists()
        CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if cls.created_chunks:
            row = {
                "chunk_id": "test",
                "source_id": "test",
                "source_title": "Test source",
                "category": "test",
                "page": 1,
                "text": "Test passage.",
            }
            CHUNKS_FILE.write_text(json.dumps(row) + "\n")

    @classmethod
    def tearDownClass(cls):
        if cls.created_chunks:
            CHUNKS_FILE.unlink(missing_ok=True)

    def test_cloud_entrypoint_renders_without_page_config_error(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
        app.run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.title[0].value, "Credit Risk Advisor")
        self.assertEqual(len(app.text_input), 1)


if __name__ == "__main__":
    unittest.main()
