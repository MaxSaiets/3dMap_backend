
import firebase_admin
from firebase_admin import credentials, storage
import os
import datetime
import time
import random
from pathlib import Path
from typing import Optional

class FirebaseService:
    _initialized = False
    _bucket = None

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return

        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")

        if not bucket_name:
            print("[WARN] FIREBASE_STORAGE_BUCKET not set. Firebase upload disabled.")
            return
        
        if not os.path.exists(cred_path):
             print(f"[WARN] Firebase credentials not found at {cred_path}. Firebase upload disabled.")
             return

        try:
            cred = credentials.Certificate(cred_path)
            # Check if likely already initialized
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })
            
            cls._bucket = storage.bucket()
            cls._initialized = True
            print(f"[INFO] Firebase initialized with bucket: {bucket_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Firebase: {e}")

    @classmethod
    def upload_file(cls, file_path: str, remote_path: Optional[str] = None) -> Optional[str]:
        if not cls._initialized:
            cls.initialize()
            if not cls._initialized:
                return None

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"[ERROR] File to upload not found: {file_path}")
            return None

        if remote_path is None:
            remote_path = file_path_obj.name

        # Retry configuration
        max_retries = 5
        base_delay = 1.0  # seconds

        for attempt in range(max_retries + 1):
            try:
                blob = cls._bucket.blob(remote_path)
                
                # Detect content type
                content_type = "application/octet-stream"
                if file_path.endswith(".3mf"):
                    content_type = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"
                elif file_path.endswith(".stl"):
                    content_type = "model/stl"
                
                if attempt > 0:
                    print(f"[INFO] Upload attempt {attempt + 1}/{max_retries + 1} for {file_path}...")
                else:
                    print(f"[INFO] Uploading {file_path} to gs://{cls._bucket.name}/{remote_path}...")
                
                # Set a large chunk size to help with reliability on some connections, or keep default.
                # Default is usually fine, but explicit timeouts (if supported by library version) would be good.
                blob.upload_from_filename(file_path, content_type=content_type, timeout=600)
                blob.make_public()
                
                url = blob.public_url
                print(f"[INFO] Upload successful. Public URL: {url}")
                return url
            
            except Exception as e:
                print(f"[WARN] Upload failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                    print(f"[INFO] Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"[ERROR] All upload attempts failed for {file_path}")
                    return None

    @classmethod
    def configure_cors(cls):
        """
        Configures CORS for the storage bucket to allow access from any origin.
        This is required for the frontend to download files directly from Firebase Storage.
        """
        if not cls._initialized:
            cls.initialize()
            if not cls._initialized:
                return

        try:
            print(f"[INFO] Configuring CORS for bucket {cls._bucket.name}...")
            # Allow all origins, typical for public previews
            cors_configuration = [
                {
                    "origin": ["*"],
                    "responseHeader": ["Content-Type", "x-goog-resumable"],
                    "method": ["GET", "HEAD", "DELETE", "PUT", "POST", "OPTIONS"],
                    "maxAgeSeconds": 3600
                }
            ]
            cls._bucket.cors = cors_configuration
            cls._bucket.patch()
            print(f"[INFO] CORS configured successfully for {cls._bucket.name}")
        except Exception as e:
            print(f"[ERROR] Failed to configure CORS: {e}")
