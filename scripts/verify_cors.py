
import firebase_admin
from firebase_admin import credentials, storage
import os
import sys
from dotenv import load_dotenv

# Load env from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

def verify_cors():
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
    if not os.path.exists(cred_path):
        # try looking in backend root
        cred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cred_path)
    
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    
    print(f"Credential path: {cred_path}")
    print(f"Bucket name: {bucket_name}")

    if not bucket_name or not os.path.exists(cred_path):
        print("Missing credentials or bucket name.")
        return

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': bucket_name
    })

    bucket = storage.bucket()
    bucket.reload() # Refresh metadata
    
    print("\n=== Current CORS Configuration ===")
    if hasattr(bucket, 'cors') and bucket.cors:
        for rule in bucket.cors:
            print(rule)
    else:
        print("No CORS configuration found (None).")
        
    print("\n=== Testing Configuration Update ===")
    try:
        cors_configuration = [
            {
                "origin": ["*"],
                "responseHeader": ["Content-Type", "x-goog-resumable"],
                "method": ["GET", "HEAD", "DELETE", "PUT", "POST", "OPTIONS"],
                "maxAgeSeconds": 3600
            }
        ]
        bucket.cors = cors_configuration
        bucket.patch()
        print("CORS configuration re-applied successfully.")
        
        bucket.reload()
        print("Verified New Config:", bucket.cors)
        
    except Exception as e:
        print(f"Error updating CORS: {e}")

if __name__ == "__main__":
    verify_cors()
