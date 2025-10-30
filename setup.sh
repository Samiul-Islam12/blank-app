#!/bin/bash
# Setup script for PDF RAG Evaluation System on Linux/Mac

set -e

echo "========================================"
echo "PDF RAG Evaluation System Setup"
echo "========================================"
echo

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    echo "Please make sure you're in the correct directory."
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "[1/3] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi
python3 --version
echo

echo "[2/3] Upgrading pip..."
python3 -m pip install --upgrade pip || echo "WARNING: Could not upgrade pip, continuing anyway..."
echo

echo "[3/3] Installing requirements..."
python3 -m pip install -r requirements.txt
echo

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "To run the application, execute:"
echo "    streamlit run streamlit_app.py"
echo
