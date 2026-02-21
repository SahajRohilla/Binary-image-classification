# Cats vs Dogs – MLOps Project
**M1: Model Development & Experiment Tracking**

---

## 📁 Project Structure

```
Mlops/
├── data/
│   ├── raw/               ← Original Kaggle dataset (Cat/, Dog/)
│   └── processed/         ← Pre-processed 224×224 images (DVC-tracked)
│       ├── train/
│       ├── val/
│       └── test/
├── models/                ← Saved .h5 model weights
├── reports/               ← Plots: loss curve, accuracy curve, confusion matrix, ROC
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── EDA.ipynb
├── .dvc/
│   └── config
├── dvc.yaml               ← DVC pipeline stages
├── params.yaml            ← All hyperparameters
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialise Git + DVC
git init
dvc init
git add .
git commit -m "Initial project setup"
```

---

## 📦 Dataset

Place the Kaggle dataset under `data/raw/` with this structure:
```
data/raw/
├── Cat/   (*.jpg images)
└── Dog/   (*.jpg images)
```

Or set up your `~/.kaggle/kaggle.json` API key, then run:
```bash
python src/data_preprocessing.py
```

---

## 🔄 DVC Pipeline

```bash
# Run the full pipeline (preprocess → train → evaluate)
dvc repro

# Track processed data and models with DVC
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw dataset with DVC"
```

---

## 🏋️ Training

```bash
# Train with default params (CNN, 15 epochs)
python src/train.py

# Train logistic regression baseline
python src/train.py --arch logistic --epochs 10
```

---

## 📊 Experiment Tracking (MLflow)

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri mlruns

# Open browser at: http://127.0.0.1:5000
```

MLflow logs per run:
| Category   | Items logged                                      |
|------------|---------------------------------------------------|
| Parameters | architecture, lr, batch size, augmentation, etc.  |
| Metrics    | train/val loss & accuracy (per epoch), test AUC  |
| Artifacts  | `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `.h5` model |

---

## 🧪 Evaluation

```bash
python src/evaluate.py
# or specify a model path:
python src/evaluate.py --model models/final_cnn_model.h5
```

Produces:
- `reports/eval_confusion_matrix.png`
- `reports/roc_curve.png`
- Logs metrics to MLflow evaluation run

---

## 🔧 Hyperparameter Tuning

Edit `params.yaml` and re-run `dvc repro` to track experiments automatically.

---

---

## 📈 M5: Monitoring & Observability

The API now includes built-in logging and metrics tracking!

### 📊 Checking Metrics
While the API is running, hit the `/metrics` endpoint to see real-time stats:
```bash
curl http://localhost:8000/metrics
```
Expected output:
```json
{
  "total_requests": 42,
  "predictions_served": 40,
  "error_count": 2,
  "error_rate": 0.0476
}
```

### 🕵️ Post-Deployment Tracking
To simulate post-deployment performance evaluation (tracking accuracy in the wild):
```bash
python scripts/live_monitoring.py
```
This generates `reports/production_monitoring.json`.

---

## 📂 Final Submission Package

To submit your assignment, ensure your directory includes the following core artifacts:

| Module | Core Files |
|--------|------------|
| **M1** | `params.yaml`, `src/train.py`, `dvc.yaml`, `mlruns/` |
| **M2** | `app/main.py`, `Dockerfile`, `models/final_cnn_model.h5` |
| **M3** | `tests/test_pipeline.py`, `.github/workflows/ci.yml` |
| **M4** | `docker-compose.yml`, `k8s/deployment.yaml`, `scripts/smoke_test.py` |
| **M5** | `app/main.py`(updated), `scripts/live_monitoring.py`, `reports/` |

### 🏁 Final Steps:
1. Ensure all tests pass: `pytest`
2. Run training and evaluation to generate latest reports.
3. Build the Docker image one last time to verify it's fresh.
4. Zip the entire `Mlops/` folder (excluding `venv/`, `data/raw/`, and huge `__pycache__` folders).

---

## 📋 Full Grading Checklist (100% Score)

- [x] M1: Model Building, DVC & MLflow (10M)
- [x] M2: FastAPI Packaging & Docker (10M)
- [x] M3: Automated CI (Tests & Image Build) (10M)
- [x] M4: CD Pipeline & Kubernetes Manifests (10M)
- [x] M5: Observability & Production Monitoring (10M)
