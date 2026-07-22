import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from rag.llm import build_context, calculate_cost, get_model_pricing
from rag.types import Chunk


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


class LlmUtilitiesTests(unittest.TestCase):
    def test_gpt_4o_mini_cost(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage, "gpt-4o-mini"), 0.75)

    def test_custom_model_requires_explicit_prices(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "No pricing configured"):
                get_model_pricing("custom-model")

    def test_custom_model_uses_environment_prices(self):
        env = {
            "LLM_PRICE_PER_M_INPUT": "1.25",
            "LLM_PRICE_PER_M_OUTPUT": "5.0",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_model_pricing("custom-model"), (1.25, 5.0))

    def test_context_contains_auditable_source_and_page(self):
        context = build_context(
            [
                Chunk(
                    chunk_id="basel-17-1",
                    source_id="basel-iii",
                    source_title="Basel III",
                    category="credit-risk",
                    page=17,
                    text="Risk weight 100%.",
                )
            ]
        )
        self.assertIn("[Basel III, p.17]", context)
        self.assertIn("Risk weight 100%.", context)


if __name__ == "__main__":
    unittest.main()
