# Development Environment Setup Guide

This guide walks you through setting up a complete local development environment for contributing to FastAPI.

## Prerequisites
- **Python**: Python 3.8, 3.9, 3.10, 3.11, or 3.12 installed.
- **Git**: Installed and configured with your name and email.
- **Package Manager**: We recommend using `pdm` or standard Python `venv` with `pip`.

## Step-by-Step Installation

### 1. Clone the Repository
Clone your fork of the repository:
```bash
git clone https://github.com/<your-username>/fastapi.git
cd fastapi
```

### 2. Configure Remote Upstream
Add the official repository as upstream:
```bash
git remote add upstream https://github.com/fastapi/fastapi.git
git fetch upstream
```

### 3. Create and Activate a Virtual Environment
Using standard Python `venv`:
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
Install FastAPI in editable mode along with all development and test dependencies:
```bash
pip install -e ".[all,dev,test]"
```
Or if using PDM:
```bash
pdm install --dev
```

### 5. Verify the Installation
Run the test suite to make sure everything works:
```bash
pytest
```
To run tests with code coverage:
```bash
pytest --cov=fastapi --cov-report=term-missing
```
To run a specific test file:
```bash
pytest tests/test_tutorial/test_first_steps.py
```

## Running the Documentation Locally
FastAPI documentation uses `mkdocs-material`. To preview docs live with hot reloading:
```bash
mkdocs serve
```
Then visit `http://127.0.0.1:8000` in your browser.
To serve a specific translation (e.g. Spanish):
```bash
mkdocs serve -f docs/es/mkdocs.yml
```
