# tests/test_chatbot.py
import unittest
from src.chatbot import predict_disease, load_model

class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.model = load_model('../models/symptom_checker_model.pkl')

    def test_predict_disease(self):
        symptoms = "headache and fever"
        disease = predict_disease(self.model, symptoms)
        self.assertIsInstance(disease, str)
        self.assertTrue(len(disease) > 0)

if __name__ == '__main__':
    unittest.main()
