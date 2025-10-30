# 🪟 Quick Fix for Windows Users

## Your Error
```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
```

## What Happened
You're in directory: `C:\Users\User\Downloads\blank-app-main (1)`  
But the `requirements.txt` file is not there.

## Quick Fix (3 Steps)

### Step 1: Re-download the Project
The file might not have been included in your download. Download the complete repository again.

### Step 2: Navigate to the Correct Folder
```powershell
cd "C:\Users\User\Downloads\blank-app-main (1)"
dir
```

You should see these files:
- ✅ requirements.txt
- ✅ streamlit_app.py
- ✅ setup.bat
- ✅ README.md

### Step 3: Run the Setup Script
```powershell
.\setup.bat
```

This will:
- ✅ Check if requirements.txt exists
- ✅ Update pip to version 25.3
- ✅ Install all dependencies
- ✅ Verify everything is working

## Alternative: Manual Installation

If setup.bat doesn't work, do this manually:

```powershell
# 1. Update pip (fixes the version notice)
python -m pip install --upgrade pip

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

## Still Not Working?

### Check if requirements.txt exists:
```powershell
dir requirements.txt
```

### If file doesn't exist, create it:
Create a new file named `requirements.txt` with this content:
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

Then run:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Need More Help?
See `SETUP_HELP.md` for detailed troubleshooting.

---

**Ready to run?**
```powershell
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501` 🚀
