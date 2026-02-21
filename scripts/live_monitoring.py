import requests
import os
import json
import time
from pathlib import Path

# Config
API_URL = "http://localhost:8000/predict"
TEST_DATA_DIR = Path("data/processed/test")

def simulate_production_monitoring(num_samples=20):
    """
    Simulates collecting 'real' requests from production and comparing
    them against ground truth labels provided later by human labellers.
    """
    print(f"🕵️ Starting Post-Deployment Performance Tracking...")
    
    logs = []
    correct = 0
    total = 0
    
    # Classes map: folder name is the ground truth
    classes = ["cat", "dog"]
    
    for label in classes:
        image_folder = TEST_DATA_DIR / label
        images = list(image_folder.glob("*.jpg"))[:num_samples//2]
        
        for img_path in images:
            with open(img_path, "rb") as f:
                # Call the live API
                response = requests.post(API_URL, files={"file": f})
                
                if response.status_code == 200:
                    prediction = response.json()
                    pred_label = prediction["label"]
                    is_correct = (pred_label == label)
                    
                    if is_correct: correct += 1
                    total += 1
                    
                    logs.append({
                        "timestamp": time.time(),
                        "filename": img_path.name,
                        "true_label": label,
                        "pred_label": pred_label,
                        "correct": is_correct
                    })
                else:
                    print(f"❌ Failed request for {img_path.name}")

    # Calculate Production Accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Save monitoring report
    report = {
        "report_time": time.ctime(),
        "total_samples": total,
        "correct_predictions": correct,
        "production_accuracy": round(accuracy, 4),
        "detailed_logs": logs
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/production_monitoring.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"📝 Monitoring Report saved to reports/production_monitoring.json")
    print(f"📊 Production Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    simulate_production_monitoring()
