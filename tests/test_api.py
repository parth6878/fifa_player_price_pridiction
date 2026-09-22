import unittest
import sys
from pathlib import Path

# Add repo root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.app import app, VALIDATION_BOUNDS

class TestFIFAValuationAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.valid_payload = {
            "POT": 95,
            "OVA": 86,
            "Age": 18,
            "Attacking": 450,
            "Defending": 80,
            "Power": 75,
            "Movement": 90,
            "Goalkeeping": 35,
            "Mentality": 90,
            "Skill": 5,
            "Height": 180,
            "Weight": 70,
            "Preferred Foot": "Left",
            "Best Position": "RW",
            "Club": "FC Barcelona"
        }

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("status"), "healthy")

    def test_clubs_endpoint(self):
        response = self.client.get("/api/clubs")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("clubs", data)
        self.assertGreater(len(data["clubs"]), 0)
        self.assertIn("FC Barcelona", data["clubs"])

    def test_predict_success(self):
        response = self.client.post("/predict", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predicted_value_eur", data)
        self.assertIsInstance(data["predicted_value_eur"], float)
        self.assertGreater(data["predicted_value_eur"], 1_000_000)

    def test_predict_out_of_bounds_age(self):
        # Age too high
        bad_payload = dict(self.valid_payload, Age=55)
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Age must be between", response.get_json().get("error", ""))

        # Age too low
        bad_payload = dict(self.valid_payload, Age=12)
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Age must be between", response.get_json().get("error", ""))

    def test_predict_out_of_bounds_ova_pot(self):
        # OVA > 99
        bad_payload = dict(self.valid_payload, OVA=105)
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("OVA", response.get_json().get("error", ""))

        # POT < 40
        bad_payload = dict(self.valid_payload, POT=35)
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("POT", response.get_json().get("error", ""))

    def test_predict_invalid_preferred_foot(self):
        bad_payload = dict(self.valid_payload, **{"Preferred Foot": "Both"})
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Preferred Foot must be either 'Left' or 'Right'", response.get_json().get("error", ""))

    def test_predict_invalid_position(self):
        bad_payload = dict(self.valid_payload, **{"Best Position": "WATERBOY"})
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Best Position", response.get_json().get("error", ""))

    def test_predict_missing_fields(self):
        incomplete_payload = {"POT": 85, "OVA": 80}
        response = self.client.post("/predict", json=incomplete_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Missing required fields", response.get_json().get("error", ""))

if __name__ == "__main__":
    unittest.main()
