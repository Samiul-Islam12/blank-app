# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites
- Python 3.8 or higher
- HuggingFace account and API token

## Step 1: Get HuggingFace API Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token (Read access is sufficient)
3. Copy the token - you'll need it in Step 4

## Step 2: Install Dependencies

### Option A: Using setup script (Linux/Mac)
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Option B: Manual installation
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Step 4: Configure the Application

In the sidebar:
1. **Enter your HuggingFace API Token**
2. **Select models** to evaluate (start with Mistral-7B)
3. **Set number of trials** (3 is recommended)

## Step 5: Upload PDFs

1. Go to **"Document Upload"** tab
2. Click **"Browse files"** and select your PDF(s)
3. Click **"Process PDFs and Create Vector Store"**
4. Wait for processing to complete (✅ appears when done)

## Step 6: Run Evaluation

1. Go to **"Evaluation"** tab
2. Enter a **test question** about your PDF content
3. Provide a **reference answer** (what the correct answer should be)
4. Click **"Run Evaluation on Selected Models"**
5. Wait for all trials to complete

## Step 7: View Results

1. Go to **"Results & Visualizations"** tab
2. Explore:
   - Summary statistics table
   - Latency comparison chart
   - Quality metrics comparison
   - Error metrics analysis
   - Comprehensive radar chart
   - Detailed trial results

## Step 8: Export Results (Optional)
Click **"Download Results as CSV"** to save your evaluation data

---

## Example Usage

### Sample Question
```
What is the main topic discussed in the document?
```

### Sample Reference Answer
```
The document discusses machine learning techniques for natural language processing, specifically focusing on transformer architectures and their applications.
```

### Expected Output
The system will:
- Generate answers from each selected model
- Run multiple trials for consistency
- Calculate 10+ evaluation metrics
- Display comparative visualizations
- Show detailed statistics

---

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Invalid API token"
- Check your HuggingFace token is correct
- Ensure you have read access enabled
- Try regenerating the token

### "Vector store not created"
- Ensure PDFs uploaded successfully
- Check PDF files are not corrupted
- Try with a smaller PDF first

### Slow response times
- This is normal for large language models
- 7B parameter models take 3-10 seconds per query
- Consider using fewer models or trials

### Out of memory
- Try smaller models
- Reduce number of PDFs
- Restart the application

---

## What's Next?

- **Upload different PDFs** to test various content types
- **Compare multiple models** to find the best for your use case
- **Adjust trial numbers** for statistical significance
- **Experiment with questions** of varying complexity
- **Export and analyze** results in external tools

---

## Need Help?

- Check `README.md` for detailed documentation
- See `example_evaluation.md` for sample test cases
- Review evaluation metrics explanations in the app sidebar

**Happy evaluating! 🎉**
