"""
Evaluation metrics for RAG system
"""
import time
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
import re

# Download required NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class RAGEvaluator:
    """Comprehensive RAG evaluation metrics"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.smoothing = SmoothingFunction().method1
        
    def calculate_latency(self, start_time: float, end_time: float) -> float:
        """Calculate response latency in seconds"""
        return round(end_time - start_time, 3)
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(float(similarity), 4)
    
    def calculate_bertscore(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BERTScore (F1, Precision, Recall)"""
        try:
            P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
            return {
                "precision": round(float(P.mean()), 4),
                "recall": round(float(R.mean()), 4),
                "f1": round(float(F1.mean()), 4)
            }
        except Exception as e:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            score = sentence_bleu([reference_tokens], candidate_tokens, 
                                 smoothing_function=self.smoothing)
            return round(score, 4)
        except Exception as e:
            return 0.0
    
    def calculate_meteor_score(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score"""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            score = meteor_score([reference_tokens], candidate_tokens)
            return round(score, 4)
        except Exception as e:
            return 0.0
    
    def detect_hallucination(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Detect potential hallucinations by checking if answer content
        is grounded in the provided contexts
        """
        # Combine all contexts
        combined_context = " ".join(contexts)
        
        # Calculate semantic similarity between answer and context
        similarity = self.calculate_cosine_similarity(answer, combined_context)
        
        # Check for factual grounding
        # Lower similarity might indicate hallucination
        hallucination_score = 1.0 - similarity
        
        # Additional check: look for phrases that indicate uncertainty
        uncertain_phrases = [
            "i don't know", "i'm not sure", "i cannot find",
            "not mentioned", "no information", "unclear"
        ]
        
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertain_phrases)
        
        return {
            "hallucination_score": round(hallucination_score, 4),
            "context_similarity": round(similarity, 4),
            "has_uncertainty": has_uncertainty,
            "risk_level": "Low" if similarity > 0.7 else "Medium" if similarity > 0.5 else "High"
        }
    
    def calculate_relevance(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Calculate relevance scores:
        - Answer relevance to question
        - Context relevance to question
        """
        # Answer relevance to question
        answer_relevance = self.calculate_cosine_similarity(question, answer)
        
        # Context relevance to question (average across all contexts)
        context_relevances = [
            self.calculate_cosine_similarity(question, ctx) 
            for ctx in contexts
        ]
        avg_context_relevance = round(np.mean(context_relevances), 4) if context_relevances else 0.0
        
        return {
            "answer_relevance": answer_relevance,
            "context_relevance": avg_context_relevance,
            "irrelevance_score": round(1.0 - answer_relevance, 4)
        }
    
    def calculate_completeness(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Estimate answer completeness based on:
        - Answer length relative to context
        - Coverage of context information
        """
        combined_context = " ".join(contexts)
        
        # Length-based completeness
        answer_length = len(answer.split())
        context_length = len(combined_context.split())
        length_ratio = min(answer_length / max(context_length * 0.3, 1), 1.0)
        
        # Semantic coverage
        coverage = self.calculate_cosine_similarity(answer, combined_context)
        
        # Combined completeness score
        completeness_score = (length_ratio * 0.3 + coverage * 0.7)
        
        return {
            "completeness_score": round(completeness_score, 4),
            "answer_length": answer_length,
            "context_length": context_length,
            "coverage": round(coverage, 4)
        }
    
    def comprehensive_evaluation(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str],
        reference_answer: str = None,
        latency: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on a RAG response
        """
        results = {
            "latency": latency,
            "question": question,
            "answer": answer,
            "num_contexts": len(contexts)
        }
        
        # Cosine similarity metrics
        context_text = " ".join(contexts)
        results["answer_context_similarity"] = self.calculate_cosine_similarity(answer, context_text)
        results["question_context_similarity"] = self.calculate_cosine_similarity(question, context_text)
        
        # Relevance metrics
        relevance = self.calculate_relevance(question, answer, contexts)
        results.update(relevance)
        
        # Hallucination detection
        hallucination = self.detect_hallucination(answer, contexts)
        results.update(hallucination)
        
        # Completeness
        completeness = self.calculate_completeness(answer, contexts)
        results.update(completeness)
        
        # If reference answer is provided, calculate additional metrics
        if reference_answer:
            bertscore = self.calculate_bertscore(reference_answer, answer)
            results["bertscore_f1"] = bertscore["f1"]
            results["bertscore_precision"] = bertscore["precision"]
            results["bertscore_recall"] = bertscore["recall"]
            
            results["bleu_score"] = self.calculate_bleu_score(reference_answer, answer)
            results["meteor_score"] = self.calculate_meteor_score(reference_answer, answer)
        
        return results
    
    def calculate_trial_score(self, evaluation_results: Dict[str, Any], trial_number: int) -> float:
        """
        Calculate an overall trial score (0-100) based on multiple metrics
        Trial number is used for tracking but doesn't affect scoring
        """
        weights = {
            "answer_relevance": 0.25,
            "context_similarity": 0.15,
            "completeness_score": 0.20,
            "bertscore_f1": 0.15,
            "bleu_score": 0.10,
            "meteor_score": 0.10,
        }
        
        # Penalty for hallucination
        hallucination_penalty = evaluation_results.get("hallucination_score", 0) * 0.05
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in evaluation_results:
                score += evaluation_results[metric] * weight
        
        # Apply hallucination penalty
        score = max(0, score - hallucination_penalty)
        
        # Convert to 0-100 scale
        trial_score = round(score * 100, 2)
        
        return trial_score


def format_evaluation_summary(evaluation: Dict[str, Any]) -> str:
    """Format evaluation results for display"""
    summary = f"""
    📊 **Evaluation Summary**
    
    **Performance Metrics:**
    - ⏱️ Latency: {evaluation.get('latency', 0):.3f}s
    - 🎯 Answer Relevance: {evaluation.get('answer_relevance', 0):.4f}
    - 📝 Completeness: {evaluation.get('completeness_score', 0):.4f}
    - 🔗 Context Similarity: {evaluation.get('context_similarity', 0):.4f}
    
    **Quality Metrics:**
    - 🎭 BERTScore F1: {evaluation.get('bertscore_f1', 'N/A')}
    - 📐 BLEU Score: {evaluation.get('bleu_score', 'N/A')}
    - ☄️ METEOR Score: {evaluation.get('meteor_score', 'N/A')}
    
    **Reliability:**
    - ⚠️ Hallucination Risk: {evaluation.get('risk_level', 'Unknown')}
    - 🎲 Irrelevance Score: {evaluation.get('irrelevance_score', 0):.4f}
    """
    return summary
