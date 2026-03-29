import unittest
from main import validate_llm_output

class TestValidateLLMOutput(unittest.TestCase):

    def setUp(self):
        self.valid_data = {
            "summary": "A valid summary.",
            "tags": ["tech", "ai"],
            "entities": [{"name": "OpenAI", "type": "org"}],
            "urgency_score": 2
        }

    def test_valid_output(self):
        self.assertIsNone(validate_llm_output(self.valid_data))

    def test_missing_fields(self):
        invalid_data = self.valid_data.copy()
        del invalid_data["summary"]
        result = validate_llm_output(invalid_data)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("Missing fields:"))
        self.assertIn("'summary'", result)

    def test_tags_not_list(self):
        invalid_data = self.valid_data.copy()
        invalid_data["tags"] = "tech, ai"
        result = validate_llm_output(invalid_data)
        self.assertEqual(result, "tags must be a list")

    def test_entities_not_list(self):
        invalid_data = self.valid_data.copy()
        invalid_data["entities"] = {"name": "OpenAI", "type": "org"}
        result = validate_llm_output(invalid_data)
        self.assertEqual(result, "entities must be a list")

    def test_invalid_urgency_score_type(self):
        invalid_data = self.valid_data.copy()
        invalid_data["urgency_score"] = "2"
        result = validate_llm_output(invalid_data)
        self.assertEqual(result, "urgency_score must be 1-3, got 2")

    def test_invalid_urgency_score_value_high(self):
        invalid_data = self.valid_data.copy()
        invalid_data["urgency_score"] = 4
        result = validate_llm_output(invalid_data)
        self.assertEqual(result, "urgency_score must be 1-3, got 4")

    def test_invalid_urgency_score_value_low(self):
        invalid_data = self.valid_data.copy()
        invalid_data["urgency_score"] = 0
        result = validate_llm_output(invalid_data)
        self.assertEqual(result, "urgency_score must be 1-3, got 0")

if __name__ == '__main__':
    unittest.main()
