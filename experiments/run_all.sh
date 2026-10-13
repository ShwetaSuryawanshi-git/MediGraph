#!/usr/bin/env bash
# =============================================================================
# MediGraph: Complete End-to-End Experimental Reproduction Suite
# IEEE Format Research Study / NCSET Conference Submission
# Order of Execution: Phase 1 > Phase 3 > Phase 2 > Phase 4 > Phase 5
# =============================================================================

set -e

# Setup runtime environment
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python"

echo "============================================================================="
echo "  MediGraph: Reproducible Benchmark & Explainability Experiments"
echo "============================================================================="
echo "Python binary: ${PYTHON_BIN}"
echo "Project root:  ${PROJECT_ROOT}"
echo "Script dir:    ${SCRIPT_DIR}"
echo "Current time:  $(date)"
echo "============================================================================="

# Phase 0: Verification
echo ""
echo ">>> [Phase 0] Verifying Experimental Environment & Dependencies..."
"${PYTHON_BIN}" -c "
import rdkit, sklearn, shap, xgboost, Bio, torch, transformers, google.genai
from tdc.multi_pred import DTI
print('Environment check passed: all required scientific packages loaded successfully.')
"

# Phase 1: DTI Benchmark (RQ1)
echo ""
echo "============================================================================="
echo ">>> [Phase 1] Running DTI Benchmark (RQ1: Random & Cold Splits)..."
echo "============================================================================="
"${PYTHON_BIN}" "${SCRIPT_DIR}/phase1_dti_benchmark.py"

# Phase 3: RAG Evaluation (RQ2)
echo ""
echo "============================================================================="
echo ">>> [Phase 3] Running PubMed RAG Evaluation (RQ2: Factual Grounding)..."
echo "============================================================================="
"${PYTHON_BIN}" "${SCRIPT_DIR}/phase3_rag_benchmark.py"

# Phase 2: SHAP Validity (RQ3)
echo ""
echo "============================================================================="
echo ">>> [Phase 2] Running SHAP Validity Benchmark (RQ3: Faithfulness & Stability)..."
echo "============================================================================="
"${PYTHON_BIN}" "${SCRIPT_DIR}/phase2_shap_benchmark.py"

# Phase 4: Retrospective Repurposing (RQ4)
echo ""
echo "============================================================================="
echo ">>> [Phase 4] Running Retrospective Repurposing Recovery Benchmark (RQ4)..."
echo "============================================================================="
"${PYTHON_BIN}" "${SCRIPT_DIR}/phase4_repurposing_benchmark.py"

echo ""
echo "============================================================================="
echo "ALL EXPERIMENTAL PHASES COMPLETED SUCCESSFULLY!"
echo "============================================================================="
