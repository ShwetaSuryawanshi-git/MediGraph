# MediGraph: Complete Codebase Walkthrough

# 1. Machine Learning Pipeline (`ml_pipeline/`)
This directory is the "Brains" of the project. It handles data gathering, mathematical modeling, and pushing results to the cloud.

### Data Gathering & Processing
`chembl_fetcher.py` & `chembl_drugbank_fetcher.py`
    *   Purpose: These scripts establish API connections to public pharmaceutical databases (like ChEMBL). They scrape raw biological data (Drugs, disease interactions, properties).
* `feature_extraction.py`
    * Purpose: Machine learning models cannot read "English" chemical names. This script uses RDKit to convert chemical molecule definitions into binary mathematical "fingerprints" (features) that algorithms can understand.
*   *`config.py`*
    *Purpose: A central file holding all hardcoded variables (like your `HIGH_CONFIDENCE_THRESHOLD = 0.85` and batch size limits).

### Modeling & Training
*   `train_ddi_model.py` & `train_real_model.py`
    *   Purpose: The algorithms. These scripts ingest the RDKit fingerprints and train classical algorithms (like Scikit-Learn **Random Forests**) to recognize patterns and predict if a Drug will successfully bind to a specific Target.
*   `evaluate_model.py`
    *  Purpose: Generates statistical graphs. After running a model, it mathematically checks the accuracy, producing Precision, Recall, Confusion Matrices, and ROC Curve charts (found in the `evaluation/` folder).

### Reasoning & Cloud Sync
*   **`explainability.py`**
    *   **Purpose:** The **SHAP** generator. This script looks at the model's prediction and reverse-engineers it, mathematically proving *which* specific molecular structure caused the drug to bind.
*   **`repurposing.py`**
    *   **Purpose:** The "Discovery" engine. It takes high-confidence DTI predictions, checks if the target disease differs from the drug's approved uses, and flags it as a "Repurposing Candidate".
*   **`run_pipeline.py` & `run_ddi_pipeline.py`**
    *   **Purpose:** The "Master Switches." Running these scripts triggers the entire sequence: fetch data $\rightarrow$ train model $\rightarrow$ predict $\rightarrow$ explain $\rightarrow$ publish.
*   **`firebase_client.py`** & **`service-account.json`**
    *   **Purpose:** This securely logs into your Firebase database as an Admin and automatically pushes the final predictions, nodes, edges, and SHAP stats into the cloud.

---

## 2. Explainable AI Backend (`backend/`)
*This directory acts as the literature researcher. It grounds the UI in real medicine.*

*   **`app.py`**
    *   **Purpose:** A lightweight Python **FastAPI** web server. It keeps port `8000` open to listen for queries from your React frontend. 
*   **`rag_pipeline.py`**
    *   **Purpose:** Uses **Retrieval-Augmented Generation (RAG)**. When it gets a request (e.g., "Explain Drug A and Target B"), it actively queries the **PubMed (Entrez)** database for live medical paper abstracts, injects them into an **OpenAI LLM**, and returns a professional, cited medical explanation back to the user.

---

## 3. Web Interface & Frontend (`src/`)
*This is everything the user clicks on and interacts with, built using React and Vite.*

### Core Setup
*   **`App.jsx` & `index.jsx`**
    *   **Purpose:** The entry points of the website. They load the UI and check if the user is authenticated out-of-the-gate.
*   **`styles/index.css`**
    *   **Purpose:** Handles all the colors, animations, glass-morphism effects, and layout aesthetics.

### Hooks & Services (The Data Plumbers)
*   **`services/firestoreService.js` / `authService.js`**
    *   **Purpose:** The direct logic that talks to Firebase cloud asking to read data or log users in via Firebase Auth.
*   **`hooks/useFirestoreData.jsx`**
    *   **Purpose:** A specialized React listener. It keeps an open socket to Firebase. When the ML Pipeline pushes a new prediction, this hook detects it instantly and forces the website to visually update in real-time without reloading the page.

### Visual Components (`components/`)
*   **`Dashboard/Dashboard.jsx`**
    *   **Purpose:** Displays high-level stats (e.g., "Total Predictions Made" or "Active Repurposing Opportunities").
*   **`Graph/GraphView.jsx`** & **`GraphCanvas.jsx`**
    *   **Purpose:** The interactive Knowledge Graph. Parses the cloud data into visual circle Nodes and Edges, allowing the researcher to physically drag and map biological pathways.
*   **`Repurposing/RepurposingView.jsx`**
    *   **Purpose:** An organized dashboard listing all the Repurposing Candidates found by `repurposing.py`, complete with their evidence panels.
*   **`Auth/AuthPage.jsx`**
    *   **Purpose:** The login/registration screens relying on Firebase identity.

---

## 4. Root Configuration Files
*These files configure the software environment that wraps the project.*

*   **`package.json`** & **`package-lock.json`**: Lists every NPM Javascript package required to run the React frontend (like `react`, `firebase`, `vite`).
*   **`ml_pipeline/requirements.txt`**: Lists every Python package required to run the backend (like `scikit-learn`, `rdkit`, `firebase-admin`, `shap`).
*   **`vite.config.js`**: Instructs the Vite bundler on how to compile your React code for maximum speed.
*   **`netlify.toml`**: Instructions for the Netlify Cloud server telling it exactly how to build and host your website (`npm run build`).
*   **`firebase/firestore.rules` & `firestore.indexes.json`**: Secure configurations instructing Google Firebase who is allowed to read vs write to the cloud database.
