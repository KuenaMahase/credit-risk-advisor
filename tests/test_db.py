import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from monitoring import db
from rag.types import AnswerResult, RewriteMetrics


class MonitoringDatabaseTests(unittest.TestCase):
    def test_conversation_records_cost_and_latency_breakdown(self):
        result = AnswerResult(
            answer="The applicable risk weight is 100%.",
            sources=[],
            search_mode="rerank",
            model="gpt-4o-mini",
            prompt_tokens=700,
            completion_tokens=40,
            total_tokens=740,
            response_time=1.25,
            cost=0.000129,
        )
        rewrite = RewriteMetrics(
            query="risk weight for unrated corporate exposures",
            tokens=25,
            cost=0.000004,
            time=0.4,
        )

        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "advisor.db"
            with patch.object(db, "DB_FILE", database):
                conversation_id = db.save_conversation(
                    "RW for unrated corps?", result, rewrite=rewrite
                )
                db.update_conversation_judge(
                    conversation_id,
                    120,
                    0.00003,
                    judge_time=0.8,
                    response_time=2.6,
                )
                db.save_feedback(conversation_id, "user", score=1)

            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM conversations").fetchone()
            feedback = conn.execute("SELECT * FROM feedback").fetchone()
            conn.close()

        self.assertEqual(row["rewritten_query"], rewrite["query"])
        self.assertEqual(row["answer_time"], 1.25)
        self.assertEqual(row["judge_time"], 0.8)
        self.assertEqual(row["response_time"], 2.6)
        self.assertEqual(row["judge_tokens"], 120)
        self.assertAlmostEqual(
            row["cost"] + row["rewrite_cost"] + row["judge_cost"],
            0.000163,
        )
        self.assertEqual(feedback["score"], 1)


if __name__ == "__main__":
    unittest.main()
