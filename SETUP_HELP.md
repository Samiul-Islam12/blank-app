# 🔧 Setup Help Guide

## Quick Fix for "requirements.txt not found" Error

If you're seeing this error:
```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
```

### Step-by-Step Fix

#### 1. Check Your Location
First, verify you're in the correct directory:

**Windows (PowerShell):**
```powershell
cd  # Shows current directory
dir # Lists files in current directory
```

**Linux/Mac (Terminal):**
```bash
pwd # Shows current directory
ls  # Lists files in current directory
```

You should see these files:
- `requirements.txt`
- `streamlit_app.py`
- `README.md`
- `setup.bat` (Windows) or `setup.sh` (Linux/Mac)

#### 2. If Files Are Missing

**Option A: Re-download the Repository**
The easiest solution is to properly download the complete repository:

```bash
# Using git (recommended):
git clone <repository-url>
cd <repository-name>

# Verify files are present:
ls  # or 'dir' on Windows
```

**Option B: Use the Setup Script**
If you have the repository but just need to set up:

**Windows:**
```powershell
.\setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Option C: Manual Creation**
Create a file named `requirements.txt` in your project directory with this content:

```
streamlit
langchain
langchain-community
langchain-huggingface
PyPDF2
pypdf
faiss-cpu
sentence-transformers
huggingface_hub>=0.20.0
transformers
torch
numpy
pandas
plotly
scikit-learn
nltk
bert-score
evaluate
rouge-score
sacrebleu
```

#### 3. Update pip (Recommended)

Before installing packages, update pip:

**Windows:**
```powershell
python -m pip install --upgrade pip
```

**Linux/Mac:**
```bash
python3 -m pip install --upgrade pip
```

#### 4. Install Dependencies

**Windows:**
```powershell
python -m pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3 -m pip install -r requirements.txt
```

#### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The app should open automatically at `http://localhost:8501`

---

## Other Common Setup Issues

### Python Not Found

**Error:** `'python' is not recognized as an internal or external command`

**Fix:**
1. Install Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Restart your terminal/PowerShell

### Permission Errors on Linux/Mac

**Error:** `Permission denied`

**Fix:**
```bash
# For setup script:
chmod +x setup.sh

# For pip install (if needed):
python3 -m pip install --user -r requirements.txt
```

### Module Not Found After Installation

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Fix:**
Make sure you're using the same Python environment where you installed packages:
```bash
# Check Python location:
which python3  # Linux/Mac
where python   # Windows

# Reinstall in correct environment:
python3 -m pip install -r requirements.txt
```

### Torch Installation Issues

The `torch` package is large (~2GB). If installation fails:

**Option 1:** Install torch separately first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Option 2:** Use CPU-only version (faster download):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Getting Help

If you're still having issues:

1. **Check you have all required files:**
   - `requirements.txt`
   - `streamlit_app.py`
   - `README.md`

2. **Verify Python version:**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

3. **Check pip is working:**
   ```bash
   python -m pip --version
   ```

4. **Try installing packages one by one** to identify which package is causing issues:
   ```bash
   pip install streamlit
   pip install langchain
   # etc.
   ```

5. **Check the full error message** and search for it online or report it as an issue.

---

## Quick Start Checklist

- [ ] Downloaded/cloned the complete repository
- [ ] In the correct directory (contains requirements.txt)
- [ ] Python 3.8+ installed
- [ ] pip updated to latest version
- [ ] All dependencies installed
- [ ] HuggingFace API token ready
- [ ] Ready to run: `streamlit run streamlit_app.py`

---

**Need more help?** Check the main [README.md](README.md) for detailed documentation.
