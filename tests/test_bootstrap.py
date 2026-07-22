import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_empty_knowledge_base_is_not_ready(self):
        with TemporaryDirectory() as tmp:
            chunks = Path(tmp) / "chunks.jsonl"
            chunks.touch()
            with patch.object(bootstrap, "CHUNKS_FILE", chunks):
                self.assertFalse(bootstrap.knowledge_base_ready())

    def test_bootstrap_builds_missing_knowledge_base_once(self):
        with TemporaryDirectory() as tmp:
            chunks = Path(tmp) / "chunks.jsonl"

            def build():
                chunks.write_text(json.dumps({"chunk_id": "test"}) + "\n")

            with (
                patch.object(bootstrap, "CHUNKS_FILE", chunks),
                patch.object(bootstrap, "load_secrets_into_env"),
                patch.object(bootstrap, "build_knowledge_base", side_effect=build) as mocked,
            ):
                bootstrap.bootstrap()
                bootstrap.bootstrap()

            mocked.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
