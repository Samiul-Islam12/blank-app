# Example Evaluation Scenarios

This file provides example questions and reference answers you can use to test the RAG system.

## Example 1: Technical Documentation

### Sample PDFs
Upload technical documentation, API guides, or software manuals.

### Test Questions & Reference Answers

**Question 1:**
```
What are the main features of this system?
```

**Reference Answer:**
```
[Provide a summary based on your actual PDF content]
```

---

**Question 2:**
```
How do I install and configure the application?
```

**Reference Answer:**
```
[Provide installation steps from your PDF]
```

---

## Example 2: Research Papers

### Sample PDFs
Upload academic papers, research articles, or scientific publications.

### Test Questions & Reference Answers

**Question 1:**
```
What methodology was used in this research?
```

**Reference Answer:**
```
[Describe the methodology from the paper]
```

---

**Question 2:**
```
What were the main findings or conclusions?
```

**Reference Answer:**
```
[Summarize the findings from the paper]
```

---

## Example 3: Business Documents

### Sample PDFs
Upload reports, presentations, or policy documents.

### Test Questions & Reference Answers

**Question 1:**
```
What are the key performance indicators mentioned?
```

**Reference Answer:**
```
[List KPIs from the document]
```

---

**Question 2:**
```
What recommendations were made?
```

**Reference Answer:**
```
[Summarize recommendations from the document]
```

---

## Tips for Creating Good Test Cases

1. **Be Specific**: Questions should have clear, factual answers in the PDFs
2. **Avoid Ambiguity**: Reference answers should be concise and accurate
3. **Test Different Aspects**: 
   - Factual recall
   - Summarization
   - Comparison
   - Analysis
4. **Vary Complexity**: Mix simple and complex questions
5. **Multiple Trials**: Run 3-5 trials to assess consistency

## Interpreting Results

### Good Performance Indicators
- BLEU > 0.3
- METEOR > 0.4
- BERTScore F1 > 0.7
- Cosine Similarity > 0.6
- Completeness > 0.5
- Hallucination < 0.3
- Irrelevance < 0.4
- Latency < 5 seconds

### Model Comparison
- Compare models across all metrics
- Look for consistency across trials (low std deviation)
- Balance quality vs speed for your use case
- Consider hallucination rates for critical applications

## Sample Workflow

1. Upload 2-3 related PDFs
2. Create 5 test questions with reference answers
3. Run evaluation with all 3 models (Mistral, Qwen, LLaMa)
4. Set trials to 3 for statistical significance
5. Analyze results in visualization tab
6. Export CSV for detailed analysis
7. Choose best model for your specific needs
