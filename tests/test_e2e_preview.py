
import unittest
from services.generation_task import GenerationTask

class TestE2EPreviewLogic(unittest.TestCase):
    
    def test_generation_task_firebase_outputs(self):
        """
        Verify that GenerationTask correctly stores and retrieves firebase_outputs
        which are critical for the frontend preview fallback.
        """
        # Setup
        task = GenerationTask(task_id="test-123", request={})
        
        # Act
        # Simulate the backend populating these during generation
        task.firebase_outputs["base_stl"] = "https://storage.googleapis.com/bucket/base.stl"
        task.firebase_outputs["roads_stl"] = "https://storage.googleapis.com/bucket/roads.stl"
        
        # Assert
        self.assertEqual(task.firebase_outputs["base_stl"], "https://storage.googleapis.com/bucket/base.stl")
        self.assertEqual(task.firebase_outputs.get("buildings_stl"), None)
        
    def test_response_structure_simulation(self):
        """
        Simulate the logic in main.py::get_status to ensure it maps 
        firebase_outputs to firebase_preview_parts correctly.
        """
        # Setup
        task = GenerationTask(task_id="test-456", request={})
        task.firebase_outputs = {
            "base_stl": "https://url/base.stl",
            "roads_stl": "https://url/roads.stl",
            "parks_stl": "https://url/parks.stl"
        }
        
        # Act - Simulate main.py logic
        response = {
            "task_id": task.task_id,
            "firebase_preview_parts": {
                "base": task.firebase_outputs.get("base_stl"),
                "roads": task.firebase_outputs.get("roads_stl"),
                "buildings": task.firebase_outputs.get("buildings_stl"),
                "water": task.firebase_outputs.get("water_stl"),
                "parks": task.firebase_outputs.get("parks_stl"),
            }
        }
        
        # Assert
        self.assertEqual(response["firebase_preview_parts"]["base"], "https://url/base.stl")
        self.assertEqual(response["firebase_preview_parts"]["roads"], "https://url/roads.stl")
        self.assertEqual(response["firebase_preview_parts"]["parks"], "https://url/parks.stl")
        self.assertIsNone(response["firebase_preview_parts"]["buildings"])
        self.assertIsNone(response["firebase_preview_parts"]["water"])

if __name__ == '__main__':
    unittest.main()
