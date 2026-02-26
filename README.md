# MediGraph

> Drug-Target Interaction Prediction and Drug Repurposing Platform

## Overview

MediGraph is a full-stack AI-powered platform for predicting drug-target interactions (DTI) and identifying drug repurposing opportunities. It emphasizes interpretability, reproducibility, and real-time data visualization.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React + Vite |
| Backend/Auth | Firebase (Auth + Firestore) |
| ML Pipeline | Python (scikit-learn, RDKit) |
| Deployment | Netlify (Frontend) |

## Project Structure

```
MediGraph/
├── src/                    # React frontend
│   ├── components/         # React components
│   ├── hooks/              # Custom React hooks
│   ├── services/           # Firebase services
│   └── styles/             # CSS styles
├── firebase/               # Firebase configuration
├── ml_pipeline/            # Python ML pipeline
├── firestore.rules         # Security rules
└── firestore.indexes.json  # Database indexes
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.9+
- Firebase CLI
- Firebase project with Auth + Firestore enabled

### Setup

1. **Clone and install frontend dependencies:**
   ```bash
   cd MediGraph
   npm install
   ```

2. **Install ML pipeline dependencies:**
   ```bash
   cd ml_pipeline
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Firebase credentials
   ```

4. **Deploy Firestore rules:**
   ```bash
   firebase deploy --only firestore:rules,firestore:indexes
   ```

5. **Run the frontend:**
   ```bash
   npm run dev
   ```

## ML Pipeline Usage

```bash
# Train a model
python ml_pipeline/run_pipeline.py train --data data/training.csv --model random_forest

# Run predictions
python ml_pipeline/run_pipeline.py run --model random_forest --model-path ./models/rf_model.pkl
```

## Architecture

- **Frontend**: Real-time data from Firestore listeners
- **ML Pipeline**: Batch processing writes predictions to Firestore
- **Repurposing**: Derived deterministically from DTI predictions + target-disease associations

## License

MIT
