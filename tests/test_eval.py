import unittest
from unittest.mock import patch

import rag.eval as eval_module


class FakeRetrievalService:
    def retrieve(self, query):
        raise RuntimeError("stubbed in later task")


class TestEvalModule(unittest.TestCase):
    def test_eval_module_can_import_without_chat_model(self):
        self.assertTrue(hasattr(eval_module, "load_test_queries"))
        self.assertTrue(hasattr(eval_module, "PolicySectionResolver"))


if __name__ == "__main__":
    unittest.main()