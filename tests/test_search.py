import unittest

from rag.search import _reciprocal_rank_fusion
from rag.types import Chunk


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id="basel-iii",
        source_title="Basel III",
        category="credit-risk",
        page=1,
        text=f"Passage {chunk_id}",
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_document_ranked_in_both_lists_wins(self):
        a = chunk("a")
        b = chunk("b")
        c = chunk("c")

        results = _reciprocal_rank_fusion([[a, b], [c, b]], k=10)

        self.assertEqual(results[0]["chunk_id"], "b")
        self.assertEqual({row["chunk_id"] for row in results}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
