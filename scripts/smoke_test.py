import requests
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def run_smoke_tests():
    print(f"🚀 Starting smoke tests on {BASE_URL}...")
    
    # 1. Check Health Endpoint
    try:
        health_resp = requests.get(f"{BASE_URL}/health", timeout=10)
        health_data = health_resp.json()
        if health_resp.status_code == 200 and health_data.get("status") == "healthy":
            print("✅ Health check passed!")
        else:
            print(f"❌ Health check failed: {health_data}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        sys.exit(1)

    # 2. Check Prediction Endpoint
    # We'll use a local test image from our processed data if available
    test_img_path = Path("data/processed/test/cat")
    if not test_img_path.exists():
        print("⚠️ Test data not found, skipping prediction smoke test...")
        return

    sample_img = next(test_img_path.glob("*.jpg"))
    print(f"📸 Testing prediction with: {sample_img}")
    
    try:
        with open(sample_img, "rb") as f:
            files = {"file": f}
            pred_resp = requests.post(f"{BASE_URL}/predict", files=files, timeout=15)
        
        if pred_resp.status_code == 200:
            result = pred_resp.json()
            print(f"✅ Prediction passed: Found {result['label']} (Prob: {result['probability']})")
        else:
            print(f"❌ Prediction failed with status {pred_resp.status_code}: {pred_resp.text}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during prediction test: {e}")
        sys.exit(1)

    print("\n🎉 All smoke tests passed!")

if __name__ == "__main__":
    # Wait a few seconds for the server to spin up if running in a combined script
    time.sleep(2)
    run_smoke_tests()
