# 📊 RAG Evaluation Guide

This guide explains how to use the comprehensive evaluation features in the RAG system.

## 🎯 Overview

The evaluation system automatically tracks and measures the quality of RAG responses across multiple dimensions:

- **Performance Metrics**: Latency, trial scores
- **Quality Metrics**: Relevance, completeness, BERTScore, BLEU, METEOR
- **Reliability Metrics**: Hallucination detection, irrelevance scoring

## 🚀 Quick Start

### Step 1: Enable Evaluation

1. Load your model and process documents
2. In the **Chat tab**, check the **"Enable Evaluation"** checkbox in the top right
3. Ask questions - each response will be automatically evaluated

### Step 2: View Results

1. Switch to the **"Evaluation Dashboard"** tab
2. View summary statistics and interactive charts
3. Download data as CSV for external analysis

## 📈 Understanding the Metrics

### Trial Score (0-100)

The overall performance score that combines multiple metrics:

- **80-100**: Excellent - High quality, relevant, complete answer
- **60-80**: Good - Decent quality with minor issues
- **40-60**: Fair - Acceptable but needs improvement
- **0-40**: Poor - Significant quality issues

**What affects it:**
- Answer relevance (25%)
- Context similarity (15%)
- Completeness (20%)
- BERTScore F1 (15%)
- BLEU score (10%)
- METEOR score (10%)
- Hallucination penalty (-5%)

### Latency

Response time in seconds from query submission to answer generation.

**Interpretation:**
- < 2s: Fast
- 2-5s: Normal
- 5-10s: Slow
- > 10s: Very slow (may need optimization)

### Cosine Similarity Scores

Measures semantic similarity using embeddings (0-1 scale):

1. **Answer-Context Similarity**: How grounded the answer is in source documents
   - > 0.7: Well-grounded
   - 0.5-0.7: Moderately grounded
   - < 0.5: Poorly grounded (risk of hallucination)

2. **Question-Context Similarity**: How relevant retrieved documents are
   - > 0.7: Highly relevant context
   - 0.5-0.7: Moderately relevant
   - < 0.5: Irrelevant context retrieved

3. **Answer Relevance**: How well the answer addresses the question
   - > 0.8: Highly relevant
   - 0.6-0.8: Relevant
   - < 0.6: Off-topic

### BERTScore F1

Semantic similarity using BERT embeddings (0-1 scale):

- **F1**: Harmonic mean of precision and recall
- **Precision**: What fraction of the answer is relevant
- **Recall**: What fraction of expected content is covered

**Interpretation:**
- > 0.9: Excellent semantic match
- 0.8-0.9: Good match
- 0.7-0.8: Moderate match
- < 0.7: Poor match

*Note: Requires a reference answer to compute*

### BLEU Score

Measures n-gram overlap between answer and reference (0-1 scale):

**Interpretation:**
- > 0.5: Excellent overlap
- 0.3-0.5: Good overlap
- 0.1-0.3: Moderate overlap
- < 0.1: Poor overlap

*Note: Requires a reference answer to compute*

### METEOR Score

Considers unigram precision/recall with synonym matching (0-1 scale):

**Interpretation:**
- > 0.6: Excellent
- 0.4-0.6: Good
- 0.2-0.4: Fair
- < 0.2: Poor

*Note: Requires a reference answer to compute*

### Hallucination Score

Risk of fabricated information (0-1 scale, **lower is better**):

**Risk Levels:**
- < 0.3: Low risk (similarity > 0.7)
- 0.3-0.5: Medium risk (similarity 0.5-0.7)
- > 0.5: High risk (similarity < 0.5)

**What it measures:**
- How much the answer deviates from source documents
- Presence of uncertainty phrases ("I don't know", "not mentioned")

### Irrelevance Score

Measure of off-topic content (0-1 scale, **lower is better**):

**Interpretation:**
- < 0.2: Highly relevant
- 0.2-0.4: Mostly relevant
- 0.4-0.6: Partially relevant
- > 0.6: Largely irrelevant

### Completeness Score

How thorough and complete the answer is (0-1 scale):

**Components:**
- Length ratio compared to context (30%)
- Semantic coverage of context (70%)

**Interpretation:**
- > 0.8: Very complete
- 0.6-0.8: Complete
- 0.4-0.6: Partially complete
- < 0.4: Incomplete

## 📊 Using the Dashboard

### Summary Statistics

At the top of the dashboard, you'll see:
- **Total Trials**: Number of evaluated queries
- **Average Latency**: Mean response time
- **Average Trial Score**: Overall performance
- **Average Relevance**: Mean relevance score
- **Average Completeness**: Mean completeness
- **Average Hallucination**: Mean hallucination risk
- **Average Context Similarity**: Mean context grounding
- **Average Irrelevance**: Mean off-topic score

### Visualizations

#### 1. Trial Scores Comparison
Bar chart showing trial score (0-100) for each query. Higher is better.

**Use case:** Identify which queries got the best/worst responses

#### 2. Latency Comparison
Bar chart showing response time for each query. Lower is better.

**Use case:** Identify performance bottlenecks

#### 3. Quality Metrics Comparison
Grouped bar chart showing relevance, completeness, and context similarity.

**Use case:** Compare multiple quality dimensions simultaneously

#### 4. Risk Metrics
Grouped bar chart showing hallucination and irrelevance scores. Lower is better.

**Use case:** Identify answers with reliability issues

#### 5. NLP Metrics (if reference answers provided)
Grouped bar chart showing BERTScore, BLEU, and METEOR.

**Use case:** Compare answers against gold standard references

### Data Table

Detailed table with all metrics for each trial. Useful for:
- Identifying patterns
- Finding outliers
- Detailed analysis

### Export Data

Click **"Download Evaluation Data (CSV)"** to export all metrics for:
- External analysis in Excel, Python, R
- Creating custom visualizations
- Long-term tracking
- Sharing results

## 🔬 Advanced Use Cases

### 1. Model Comparison

Compare different models on the same questions:

1. Load Model A (e.g., Qwen 2.5 - 3B)
2. Ask questions with evaluation enabled
3. Note the model name in results
4. Unload Model A
5. Load Model B (e.g., Mistral 7B)
6. Ask the same questions
7. Compare trial scores, latency, and quality metrics in the dashboard

### 2. Prompt Optimization

Test different question formulations:

1. Ask the same question in different ways
2. Compare relevance and completeness scores
3. Identify which phrasing produces better results

### 3. Document Quality Assessment

Evaluate if your documents are suitable for RAG:

- Low context similarity → Documents may not contain relevant information
- High irrelevance scores → Retrieved chunks may be off-topic
- Adjust chunk size or retrieval parameters

### 4. System Optimization

Identify bottlenecks:

- High latency → Consider smaller model or hardware upgrade
- Low relevance → Adjust retrieval parameters (k value)
- High hallucination → Reduce temperature or add constraints

### 5. Benchmark Testing

Create a test set of questions:

1. Prepare reference answers
2. Run all questions through the system
3. Review BERTScore, BLEU, and METEOR metrics
4. Calculate average trial score as system benchmark

## 🛠️ Tips & Best Practices

### Getting Reliable Metrics

1. **Use Evaluation Consistently**: Enable it for all queries in a session
2. **Ask Similar Complexity Questions**: Mix of simple and complex
3. **Track Multiple Trials**: At least 5-10 queries for meaningful comparisons
4. **Note the Context**: Document types affect metrics

### Interpreting Results

1. **Don't Optimize for One Metric**: Balance across multiple dimensions
2. **Consider Trade-offs**: Lower latency may come with reduced quality
3. **Context Matters**: Technical docs vs. casual writing affects scores
4. **Hallucination > 0.5**: Investigate the answer carefully

### When to Re-evaluate

- After changing models
- After adjusting chunk size or overlap
- After modifying retrieval parameters (k value)
- After adding new documents
- For model comparison studies

## 📝 Metric Limitations

### What These Metrics DON'T Measure

- **Factual Correctness**: Metrics measure similarity and relevance, not truth
- **User Satisfaction**: Subjective quality may differ from metrics
- **Domain Expertise**: Requires human verification for specialized domains
- **Creative Quality**: For tasks requiring creativity beyond retrieval

### Reference-Based Metrics Caveat

BERTScore, BLEU, and METEOR require reference answers:
- Only use when you have ground truth
- Manual evaluation still necessary for novel questions
- Good for benchmarking, not runtime evaluation

## 🔄 Clearing and Resetting

To start fresh:

1. Click **"Clear Evaluations"** button in the dashboard
2. This resets trial counter and removes all stored evaluations
3. Chat history remains intact (stored separately)

## 📊 Example Analysis Workflow

1. **Collect Data**: Ask 10 questions with evaluation enabled
2. **Review Dashboard**: Check summary statistics
3. **Identify Issues**: Look for patterns in low-scoring trials
4. **Investigate**: Review specific low-score answers and sources
5. **Adjust**: Modify system parameters or documents
6. **Re-test**: Ask same questions and compare results
7. **Export**: Download data and track improvements over time

## 🆘 Troubleshooting

### Evaluation is Slow

- First query initializes the evaluator (takes a few seconds)
- Subsequent evaluations are faster
- Disable evaluation for casual usage

### Metrics Show "N/A"

- BERTScore, BLEU, METEOR require reference answers
- Provide reference answers in evaluation code if needed

### Unexpected Low Scores

- Check if context documents contain relevant information
- Review hallucination scores for potential issues
- Verify model is loaded correctly
- Try reformulating the question

---

**For more information, see the main README.md or evaluation_metrics.py source code.**
