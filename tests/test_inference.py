import unittest
from src.inference import generate_rag_answer
from unittest.mock import MagicMock

class TestInference(unittest.TestCase):
    def test_generate_rag_answer(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []  # Empty docs
        answer = generate_rag_answer("Test query", mock_retriever)
        self.assertIn("not available", answer.lower())  # Expected fallback

if __name__ == "__main__":
    unittest.main()