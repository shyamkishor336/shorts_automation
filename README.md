# HITL AI Multimedia Pipeline

**Design and Evaluation of an Adaptive Human-in-the-Loop AI Multimedia Pipeline with ML-Based Intervention Prediction**

MSc IT Dissertation — Shyam Kishor Pandit | University of the West of Scotland | Banner ID: B01804169

This system produces short educational YouTube videos (~48–60 seconds) from text prompts across three research modes: fully automated (Mode A), human-in-the-loop (Mode B), and ML-predicted intervention (Mode C).

---

## 1. Project Overview

The pipeline has four stages: (1) Script Generation via Gemini 2.5 Flash, (2) Audio Synthesis via Edge-TTS, (3) Visual Generation via multi-provider rotation (Modal, fal.ai, HF, GCP, Kaggle, Ken Burns), and (4) Video Assembly via FFmpeg. The system records 25+ ML features per stage attempt and trains a Random Forest classifier to predict when human review is needed.

The experiment uses 20 fixed prompts × 5 repetitions × 2 modes = 200 runs, generating ~1,600 video clips.

---

## 2. One-Time Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- FFmpeg (must be on PATH)

### Python Environment

```bash
cd shorts_automation
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Environment Variables

```bash
cp .env.template .env
# Edit .env and fill in all API keys
```

### Database

```bash
createdb hitl_pipeline
python backend/database.py
```

### Frontend

```bash
cd frontend
npm install
```

---

## 3. Deploying Modal (One Time Only)

Modal provides real-time CogVideoX-2b inference (~550 clips allocation).

```bash
pip install modal
modal setup        # Follow browser authentication
modal deploy modal_cogvideox_endpoint.py
```

Copy the printed endpoint URL into `.env`:
```
MODAL_ENDPOINT_URL=https://your-name--cogvideox-hitl-generate.modal.run
```

---

## 4. GCP Setup ($300 Free Credits — ~500 clips)

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and sign up (new accounts receive $300 free credit)
2. Create a VM instance:
   - **Compute Engine → VM Instances → Create Instance**
   - Machine type: `n1-standard-4` (4 vCPU, 15 GB RAM)
   - GPU: NVIDIA T4 (1×)
   - Boot disk: **Deep Learning on Linux**, 50 GB SSD
   - Zone: `us-central1-a` (cheapest with T4 availability)
3. SSH into the VM:
   ```bash
   gcloud compute ssh hitl-cogvideox-vm --zone=us-central1-a
   ```
4. Populate the queue. Start pipeline runs; GCP jobs appear in `data/gcp_queue.json` automatically.
5. Upload script and queue to VM:
   ```bash
   gcloud compute scp gcp_cogvideox_batch.py data/gcp_queue.json \
     hitl-cogvideox-vm:~/ --zone=us-central1-a
   ```
6. Run batch generation on VM:
   ```bash
   python gcp_cogvideox_batch.py
   ```
7. Download results:
   ```bash
   mkdir -p data/gcp_results
   gcloud compute scp "hitl-cogvideox-vm:~/gcp_outputs/*" data/gcp_results/ \
     --zone=us-central1-a
   gcloud compute scp hitl-cogvideox-vm:~/gcp_results.json data/ --zone=us-central1-a
   ```
8. Copy videos to their run directories: `data/outputs/[run_id]/scenes/`
9. For each completed job, mark it complete:
   ```bash
   curl -X POST http://localhost:8000/providers/queue/complete \
     -H "Content-Type: application/json" \
     -d '{"platform": "gcp", "job_id": "...", "output_path": "..."}'
   ```
10. **IMPORTANT — Stop the VM to avoid charges:**
    ```bash
    gcloud compute instances stop hitl-cogvideox-vm --zone=us-central1-a
    ```

---

## 5. Kaggle Setup (30 hrs/week free — ~400 clips)

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. Upload `kaggle_cogvideox_notebook.py`
3. Enable GPU: **Settings → Accelerator → GPU T4 x2** (or P100)
4. Upload `data/kaggle_queue.json` as a dataset named `hitl-kaggle-queue`
5. **Run all cells**
6. Download from `/kaggle/working/`:
   - All `.mp4` files (generated clips)
   - `kaggle_results.json` (completion manifest)
7. Copy `.mp4` files to `data/outputs/[run_id]/scenes/`
8. Mark jobs complete via:
   ```bash
   curl -X POST http://localhost:8000/providers/queue/complete \
     -H "Content-Type: application/json" \
     -d '{"platform": "kaggle", "job_id": "...", "output_path": "..."}'
   ```

---

## 6. Running the Experiment

### Start the Backend

```bash
cd shorts_automation
uvicorn backend.main:app --reload
```

API available at [http://localhost:8000](http://localhost:8000)
Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### Start the Frontend

```bash
cd frontend
npm run dev
```

Frontend available at [http://localhost:5173](http://localhost:5173)

### Experiment Order

1. **Start Mode A runs first** (fully automated, establishes baseline):
   - Click "Start New Run" → Select each of the 20 prompts → Mode A
   - Runs complete without human intervention

2. **Run Mode B** (requires human review in browser):
   - Click "Start New Run" → Select prompts → Mode B
   - Navigate to the **Review** tab
   - For each pending stage, inspect the output and click Accept or Reject
   - Rejected stages are retried automatically

3. **Mode C** (after ML training — see step 8):
   - Click "Start New Run" → Mode C

---

## 7. Exporting Training Data

After completing Mode B runs:

```bash
curl http://localhost:8000/export/csv -o hitl_training_data.csv
```

Or navigate to [http://localhost:8000/export/csv](http://localhost:8000/export/csv) in your browser to download the CSV.

The CSV contains all Mode B stage attempts with 25+ features and `human_decision` labels.

---

## 8. Training the Classifier

After Mode B data collection is complete (minimum ~100 reviewed stages recommended):

```bash
python backend/ml/train_classifier.py
```

This trains Logistic Regression, Random Forest, and XGBoost for each of the 4 stages, prints a comparison table, and saves the best model to `backend/ml/model_{stage}.pkl`.

To evaluate saved models:

```bash
python backend/ml/evaluate_classifier.py
```

Outputs confusion matrices, ROC curves, and `backend/ml/evaluation_results.json`.

---

## 9. Provider Credit Status

- Check the **Provider Status** panel on the dashboard at [http://localhost:5173](http://localhost:5173)
- **Green** = active and within budget
- **Orange** = low (< 20% remaining)
- **Red** = exhausted

To reset a provider at the start of a new billing month:
```bash
curl -X POST http://localhost:8000/providers/modal/reset
```

Or click the **↺** button next to the provider in the UI.

### Provider Allocation Summary

| Provider | Clips | Method | Notes |
|----------|-------|--------|-------|
| Modal.com | 550 | Real-time API | ~$30/month |
| Kaggle | 400 | Batch (manual) | 30 hrs/week free |
| GCP | 500 | Batch (manual) | $300 free credit |
| fal.ai | 100 | Real-time API | Free signup credits |
| Hugging Face | ∞ | Real-time API | Slow, overflow only |
| Ken Burns | ∞ | Local FFmpeg | Always available fallback |

---

## File Structure

```
shorts_automation/
├── backend/                    FastAPI backend
│   ├── main.py                 API entry point
│   ├── config.py               Settings from .env
│   ├── database.py             SQLAlchemy + PostgreSQL
│   ├── models.py               ORM models
│   ├── pipeline/               Stage implementations
│   ├── providers/              Video generation providers
│   ├── features/               ML feature extraction
│   ├── routers/                API route handlers
│   └── ml/                     Classifier training/evaluation
├── frontend/                   React + Vite + TypeScript UI
├── data/
│   ├── prompts.json            20 fixed experiment prompts
│   ├── provider_budget.json    Provider usage tracking
│   └── outputs/[run_id]/       Per-run outputs
├── modal_cogvideox_endpoint.py Deployed to Modal.com
├── kaggle_cogvideox_notebook.py Run manually on Kaggle
├── gcp_cogvideox_batch.py      Run on GCP VM
├── .env.template               API key template
└── requirements.txt
```
