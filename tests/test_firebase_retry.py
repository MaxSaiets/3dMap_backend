
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.firebase_service import FirebaseService

class TestFirebaseRetry(unittest.TestCase):
    
    @patch('services.firebase_service.storage')
    @patch('services.firebase_service.time.sleep', return_value=None) # Skip sleep
    @patch('pathlib.Path.exists', return_value=True)
    def test_upload_retry_failure(self, mock_exists, mock_sleep, mock_storage):
        # Setup - Mock storage blob to ALWAYS raise exception
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Configure upload_from_filename to raise Exception every time
        mock_blob.upload_from_filename.side_effect = Exception("Simulated Network Error")
        
        mock_bucket.blob.return_value = mock_blob
        FirebaseService._bucket = mock_bucket
        FirebaseService._initialized = True
        
        # Act
        with patch('builtins.print') as mocked_print:
            url = FirebaseService.upload_file("dummy_path.stl")
        
        # Assert
        # Should return None after retries
        self.assertIsNone(url)
        # Should have called upload (1 initial + 5 retries = 6 calls)
        self.assertEqual(mock_blob.upload_from_filename.call_count, 6)
        
    @patch('services.firebase_service.storage')
    @patch('services.firebase_service.time.sleep', return_value=None)
    @patch('pathlib.Path.exists', return_value=True)
    def test_upload_retry_success(self, mock_exists, mock_sleep, mock_storage):
        # Setup - Fail twice, then succeed
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.public_url = "http://success.url"
        
        # Side effect: Exception, Exception, Success (None)
        mock_blob.upload_from_filename.side_effect = [Exception("Fail 1"), Exception("Fail 2"), None]
        
        mock_bucket.blob.return_value = mock_blob
        FirebaseService._bucket = mock_bucket
        FirebaseService._initialized = True
        
        # Act
        url = FirebaseService.upload_file("dummy_path.stl")
        
        # Assert
        self.assertEqual(url, "http://success.url")
        # Should have called upload 3 times (1 initial + 2 retries)
        self.assertEqual(mock_blob.upload_from_filename.call_count, 3)

if __name__ == '__main__':
    unittest.main()
