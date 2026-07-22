import unittest

from rag.search import _reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_document_ranked_in_both_lists_wins(self):
        a = {"chunk_id": "a"}
        b = {"chunk_id": "b"}
        c = {"chunk_id": "c"}

        results = _reciprocal_rank_fusion([[a, b], [c, b]], k=10)

        self.assertEqual(results[0]["chunk_id"], "b")
        self.assertEqual({row["chunk_id"] for row in results}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
