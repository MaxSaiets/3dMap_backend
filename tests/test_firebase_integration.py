
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.firebase_service import FirebaseService

class TestFirebaseIntegration(unittest.TestCase):
    
    @patch('services.firebase_service.firebase_admin')
    @patch('services.firebase_service.credentials')
    @patch('services.firebase_service.storage')
    def test_initialize_success(self, mock_storage, mock_creds, mock_admin):
        # Setup
        mock_admin._apps = {}
        os.environ["FIREBASE_STORAGE_BUCKET"] = "test-bucket"
        
        # Act
        FirebaseService._initialized = False
        FirebaseService.initialize()
        
        # Assert
        mock_admin.initialize_app.assert_called_once()
        self.assertTrue(FirebaseService._initialized)
        
    @patch('services.firebase_service.storage')
    def test_configure_cors(self, mock_storage):
        # Setup
        mock_bucket = MagicMock()
        mock_storage.bucket.return_value = mock_bucket
        FirebaseService._bucket = mock_bucket
        FirebaseService._initialized = True
        
        # Act
        FirebaseService.configure_cors()
        
        # Assert
        # Check if cors was set
        self.assertTrue(hasattr(mock_bucket, 'cors'))
        cors_config = mock_bucket.cors
        self.assertEqual(len(cors_config), 1)
        self.assertEqual(cors_config[0]['origin'], ["*"])
        mock_bucket.patch.assert_called_once()
        
    @patch('services.firebase_service.storage')
    @patch('pathlib.Path.exists')
    def test_upload_file(self, mock_exists, mock_storage):
        # Setup
        mock_exists.return_value = True
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.public_url = "https://storage.googleapis.com/test-bucket/test.stl"
        
        FirebaseService._bucket = mock_bucket
        FirebaseService._initialized = True
        
        # Act
        url = FirebaseService.upload_file("path/to/test.stl")
        
        # Assert
        mock_bucket.blob.assert_called_with("test.stl")
        mock_blob.upload_from_filename.assert_called_with("path/to/test.stl", content_type="model/stl")
        mock_blob.make_public.assert_called_once()
        self.assertEqual(url, "https://storage.googleapis.com/test-bucket/test.stl")

if __name__ == '__main__':
    unittest.main()
