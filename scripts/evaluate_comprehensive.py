"""
Comprehensive Evaluation Script for Real Estate Document Intelligence System

Computes all required metrics:
1. Recall@K (K=1,3,5)
2. Top-K Accuracy
3. MRR (Mean Reciprocal Rank)
4. nDCG (Normalized Discounted Cumulative Gain)
5. Entity Coverage Score
6. Paraphrase Robustness Score
7. Hallucination Rate
8. False Positive Rate
9. Stage-wise Latency Breakdown

Usage:
    python scripts/evaluate_comprehensive.py --api-url http://localhost:8000
"""

import json
import time
import argparse
import requests
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from collections import defaultdict
import re
import math


@dataclass
class LatencyStats:
    """Stage-wise latency statistics."""
    embedding_times: List[float] = field(default_factory=list)
    retrieval_times: List[float] = field(default_factory=list)
    reranking_times: List[float] = field(default_factory=list)
    formatting_times: List[float] = field(default_factory=list)
    total_times: List[float] = field(default_factory=list)
    
    def add(self, breakdown: Dict):
        """Add latency breakdown from a query."""
        self.embedding_times.append(breakdown.get('embedding_ms', 0))
        self.retrieval_times.append(breakdown.get('retrieval_ms', 0))
        self.reranking_times.append(breakdown.get('reranking_ms', 0))
        self.formatting_times.append(breakdown.get('formatting_ms', 0))
        self.total_times.append(breakdown.get('total_ms', 0))
    
    def get_stats(self) -> Dict:
        """Get statistics for each stage."""
        def compute_stats(times: List[float], name: str) -> Dict:
            if not times:
                return {f'{name}_avg': 0, f'{name}_p50': 0, f'{name}_p95': 0}
            arr = np.array(times)
            return {
                f'{name}_avg': round(np.mean(arr), 2),
                f'{name}_p50': round(np.percentile(arr, 50), 2),
                f'{name}_p95': round(np.percentile(arr, 95), 2),
                f'{name}_min': round(np.min(arr), 2),
                f'{name}_max': round(np.max(arr), 2)
            }
        
        stats = {}
        stats.update(compute_stats(self.embedding_times, 'embedding'))
        stats.update(compute_stats(self.retrieval_times, 'retrieval'))
        stats.update(compute_stats(self.reranking_times, 'reranking'))
        stats.update(compute_stats(self.formatting_times, 'formatting'))
        stats.update(compute_stats(self.total_times, 'total'))
        return stats


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: int
    query: str
    category: str
    expected_keywords: List[str]
    expected_entities: List[str]
    results: List[Dict]
    ranks: List[int]  # Ranks where relevant results found (1-indexed)
    latency_ms: float
    latency_breakdown: Dict
    cached: bool
    
    @property
    def found_in_top_1(self) -> bool:
        return 1 in self.ranks
    
    @property
    def found_in_top_3(self) -> bool:
        return any(r <= 3 for r in self.ranks)
    
    @property
    def found_in_top_5(self) -> bool:
        return any(r <= 5 for r in self.ranks)
    
    @property
    def first_relevant_rank(self) -> Optional[int]:
        return min(self.ranks) if self.ranks else None


@dataclass
class ComprehensiveMetrics:
    """All evaluation metrics."""
    # Recall metrics
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    
    # Accuracy metrics
    top_1_accuracy: float = 0.0
    top_3_accuracy: float = 0.0
    top_5_accuracy: float = 0.0
    
    # Ranking metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    
    # Quality metrics
    entity_coverage: float = 0.0
    paraphrase_robustness: float = 0.0
    hallucination_rate: float = 0.0
    false_positive_rate: float = 0.0
    
    # Latency metrics
    latency_stats: Dict = field(default_factory=dict)
    cache_stats: Dict = field(default_factory=dict)
    
    # Targets met
    targets: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ComprehensiveEvaluator:
    """
    Evaluates the document retrieval system with comprehensive metrics.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.questions = self._load_questions()
        self.paraphrase_questions = self._load_paraphrases()
        self.negative_questions = self._load_negative_queries()
        self.latency_stats = LatencyStats()
        self.results: List[EvaluationResult] = []
    
    def _load_questions(self) -> List[Dict]:
        """Load evaluation questions from JSON file."""
        questions_path = Path(__file__).parent.parent / "tests" / "evaluation_questions.json"
        
        if not questions_path.exists():
            print(f"Warning: Questions file not found at {questions_path}")
            return self._default_questions()
        
        with open(questions_path, 'r') as f:
            questions = json.load(f)
        
        # Ensure all questions have expected_entities
        for q in questions:
            if 'expected_entities' not in q:
                q['expected_entities'] = q.get('expected_keywords', [])
        
        return questions
    
    def _default_questions(self) -> List[Dict]:
        """Default evaluation questions for real estate."""
        return [
            {
                "id": 1,
                "query": "What is the total square footage of the property?",
                "expected_keywords": ["square feet", "sq ft", "sqft", "area"],
                "expected_entities": ["square footage", "property size"],
                "category": "property_details"
            },
            {
                "id": 2,
                "query": "How many bedrooms does the property have?",
                "expected_keywords": ["bedroom", "bedrooms", "bed"],
                "expected_entities": ["bedroom count"],
                "category": "property_details"
            },
            {
                "id": 3,
                "query": "What is the asking price?",
                "expected_keywords": ["$", "price", "cost", "asking"],
                "expected_entities": ["price", "cost"],
                "category": "pricing"
            },
            {
                "id": 4,
                "query": "Are there any schools nearby?",
                "expected_keywords": ["school", "education", "elementary", "high school"],
                "expected_entities": ["school", "education"],
                "category": "amenities"
            },
            {
                "id": 5,
                "query": "Does the property have a garage?",
                "expected_keywords": ["garage", "parking", "car"],
                "expected_entities": ["garage", "parking"],
                "category": "features"
            },
            {
                "id": 6,
                "query": "What year was the house built?",
                "expected_keywords": ["built", "year", "constructed"],
                "expected_entities": ["year built", "construction date"],
                "category": "property_details"
            },
            {
                "id": 7,
                "query": "Is there a swimming pool?",
                "expected_keywords": ["pool", "swimming"],
                "expected_entities": ["swimming pool"],
                "category": "features"
            },
            {
                "id": 8,
                "query": "What are the HOA fees?",
                "expected_keywords": ["hoa", "fee", "association", "monthly"],
                "expected_entities": ["HOA", "fees"],
                "category": "pricing"
            }
        ]
    
    def _load_paraphrases(self) -> List[Dict]:
        """Load paraphrase variations for robustness testing."""
        paraphrase_path = Path(__file__).parent.parent / "tests" / "paraphrase_questions.json"
        
        if paraphrase_path.exists():
            with open(paraphrase_path, 'r') as f:
                return json.load(f)
        
        # Default paraphrases if file not found
        return [
            {
                "original_query": "What is the total square footage of the property?",
                "paraphrases": [
                    "How big is the property?",
                    "What's the size of the house?",
                    "Tell me the square footage",
                    "How many sq ft is the home?"
                ],
                "expected_keywords": ["square feet", "sq ft", "sqft", "area"]
            },
            {
                "original_query": "How many bedrooms does the property have?",
                "paraphrases": [
                    "Number of bedrooms?",
                    "How many beds?",
                    "Bedroom count in the house?"
                ],
                "expected_keywords": ["bedroom", "bedrooms", "bed"]
            },
            {
                "original_query": "What is the asking price?",
                "paraphrases": [
                    "How much does it cost?",
                    "What's the price?",
                    "List price of the property?"
                ],
                "expected_keywords": ["$", "price", "cost", "asking"]
            }
        ]
    
    def _load_negative_queries(self) -> List[Dict]:
        """Load queries whose answers should NOT exist in documents (for FP testing)."""
        negative_path = Path(__file__).parent.parent / "tests" / "negative_queries.json"
        
        if negative_path.exists():
            with open(negative_path, 'r') as f:
                return json.load(f)
        
        # Default negative queries if file not found
        return [
            {"id": 1, "query": "What cryptocurrency payment options are accepted?", "category": "negative"},
            {"id": 2, "query": "What is the restaurant menu?", "category": "negative"},
            {"id": 3, "query": "Who won the 2024 election?", "category": "negative"},
            {"id": 4, "query": "What are the airline flight schedules?", "category": "negative"},
            {"id": 5, "query": "How do I fix my car engine?", "category": "negative"}
        ]
    
    def _check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _search(self, query: str, top_k: int = 5) -> tuple:
        """Execute search query."""
        try:
            start = time.time()
            response = requests.post(
                f"{self.api_url}/api/search",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return (
                    data.get("results", []),
                    latency,
                    data.get("cached", False),
                    data.get("latency_breakdown", {})
                )
            else:
                print(f"Search error: {response.status_code}")
                return [], latency, False, {}
                
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return [], 0, False, {}
    
    def _check_relevance(self, result: Dict, expected_keywords: List[str]) -> bool:
        """Check if a result contains expected keywords."""
        text = result.get("text", "").lower()
        return any(keyword.lower() in text for keyword in expected_keywords)
    
    def _find_relevant_ranks(self, results: List[Dict], expected_keywords: List[str]) -> List[int]:
        """Find all ranks where relevant results appear (1-indexed)."""
        ranks = []
        for i, result in enumerate(results):
            if self._check_relevance(result, expected_keywords):
                ranks.append(i + 1)
        return ranks
    
    def _calculate_ndcg(self, ranks: List[int], k: int = 5) -> float:
        """Calculate nDCG@k."""
        if not ranks:
            return 0.0
        
        # Create relevance array (1 for relevant, 0 for irrelevant)
        relevance = [0] * k
        for r in ranks:
            if r <= k:
                relevance[r - 1] = 1
        
        # DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
        
        # Ideal DCG (all relevant at top)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _count_entity_mentions(self, results: List[Dict], expected_entities: List[str]) -> float:
        """Calculate entity coverage score."""
        if not expected_entities or not results:
            return 0.0
        
        all_text = " ".join(r.get("text", "") for r in results).lower()
        mentioned = sum(1 for entity in expected_entities if entity.lower() in all_text)
        return mentioned / len(expected_entities)
    
    def evaluate_single(self, question: Dict, top_k: int = 5) -> EvaluationResult:
        """Evaluate a single question."""
        query = question["query"]
        expected_keywords = question.get("expected_keywords", [])
        expected_entities = question.get("expected_entities", expected_keywords)
        category = question.get("category", "general")
        
        results, latency, cached, breakdown = self._search(query, top_k=top_k)
        
        # Track latency breakdown
        if breakdown:
            self.latency_stats.add(breakdown)
        
        ranks = self._find_relevant_ranks(results, expected_keywords)
        
        return EvaluationResult(
            query_id=question.get("id", 0),
            query=query,
            category=category,
            expected_keywords=expected_keywords,
            expected_entities=expected_entities,
            results=results,
            ranks=ranks,
            latency_ms=latency,
            latency_breakdown=breakdown,
            cached=cached
        )
    
    def evaluate_paraphrase_robustness(self) -> float:
        """
        Evaluate system consistency across paraphrased versions of same question.
        Returns score between 0-1 indicating consistency.
        """
        consistency_scores = []
        
        for paraphrase_set in self.paraphrase_questions:
            original = paraphrase_set.get("original_query", paraphrase_set.get("original", ""))
            paraphrases = paraphrase_set["paraphrases"]
            keywords = paraphrase_set["expected_keywords"]
            
            # Get results for original
            orig_results, _, _, _ = self._search(original)
            orig_found = any(self._check_relevance(r, keywords) for r in orig_results[:3])
            
            # Get results for each paraphrase
            paraphrase_found = []
            for para in paraphrases:
                para_results, _, _, _ = self._search(para)
                found = any(self._check_relevance(r, keywords) for r in para_results[:3])
                paraphrase_found.append(found)
            
            # Consistency: all should have same result as original
            if paraphrase_found:
                consistency = sum(1 for pf in paraphrase_found if pf == orig_found) / len(paraphrase_found)
                consistency_scores.append(consistency)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 1.0
    
    def evaluate_false_positive_rate(self) -> float:
        """
        Evaluate false positive rate using negative queries.
        Returns percentage of negative queries that inappropriately returned relevant-looking results.
        """
        false_positives = 0
        
        for neg_query in self.negative_questions:
            results, _, _, _ = self._search(neg_query["query"], top_k=3)
            
            # Consider it a false positive if any result has high similarity score
            for r in results:
                score = r.get("similarity_score", 0)
                if score > 0.5:  # High confidence result for irrelevant query
                    false_positives += 1
                    break
        
        return false_positives / len(self.negative_questions) if self.negative_questions else 0.0
    
    def evaluate_hallucination_rate(self, results: List[EvaluationResult]) -> float:
        """
        Estimate hallucination rate based on low-confidence results being returned.
        In a pure retrieval system, hallucination would be returning irrelevant content.
        """
        hallucination_count = 0
        total_results = 0
        
        for eval_result in results:
            for r in eval_result.results[:3]:  # Check top-3
                total_results += 1
                score = r.get("similarity_score", 0)
                # Low score but still returned = potentially misleading
                if score < 0.3 and not self._check_relevance(r, eval_result.expected_keywords):
                    hallucination_count += 1
        
        return hallucination_count / total_results if total_results > 0 else 0.0
    
    def run_evaluation(self, num_runs: int = 1, verbose: bool = True) -> ComprehensiveMetrics:
        """Run comprehensive evaluation."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 70)
        
        if not self._check_health():
            print("ERROR: API is not healthy. Please ensure the server is running.")
            return ComprehensiveMetrics()
        
        print(f"\nAPI URL: {self.api_url}")
        print(f"Questions: {len(self.questions)}")
        print(f"Runs per question: {num_runs}")
        print("-" * 70)
        
        all_results = []
        
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            
            for i, question in enumerate(self.questions):
                result = self.evaluate_single(question)
                all_results.append(result)
                
                status = "✓" if result.found_in_top_3 else "✗"
                if verbose:
                    print(f"  [{i+1:2d}] {status} {question['query'][:50]}... "
                          f"(rank: {result.first_relevant_rank or '-'}, {result.latency_ms:.0f}ms)")
        
        self.results = all_results
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_results)
        
        # Print report
        self._print_report(metrics)
        
        return metrics
    
    def _calculate_metrics(self, results: List[EvaluationResult]) -> ComprehensiveMetrics:
        """Calculate all comprehensive metrics."""
        total = len(results)
        metrics = ComprehensiveMetrics()
        
        if total == 0:
            return metrics
        
        # === Recall@K ===
        metrics.recall_at_1 = sum(1 for r in results if r.found_in_top_1) / total
        metrics.recall_at_3 = sum(1 for r in results if r.found_in_top_3) / total
        metrics.recall_at_5 = sum(1 for r in results if r.found_in_top_5) / total
        
        # === Top-K Accuracy (same as recall for single relevant item) ===
        metrics.top_1_accuracy = metrics.recall_at_1
        metrics.top_3_accuracy = metrics.recall_at_3
        metrics.top_5_accuracy = metrics.recall_at_5
        
        # === MRR (Mean Reciprocal Rank) ===
        reciprocal_ranks = [1 / r.first_relevant_rank if r.first_relevant_rank else 0 for r in results]
        metrics.mrr = float(np.mean(reciprocal_ranks))
        
        # === nDCG ===
        ndcg_3_scores = [self._calculate_ndcg(r.ranks, k=3) for r in results]
        ndcg_5_scores = [self._calculate_ndcg(r.ranks, k=5) for r in results]
        metrics.ndcg_at_3 = float(np.mean(ndcg_3_scores))
        metrics.ndcg_at_5 = float(np.mean(ndcg_5_scores))
        
        # === Entity Coverage ===
        entity_scores = [
            self._count_entity_mentions(r.results[:3], r.expected_entities) 
            for r in results
        ]
        metrics.entity_coverage = float(np.mean(entity_scores))
        
        # === Paraphrase Robustness ===
        metrics.paraphrase_robustness = self.evaluate_paraphrase_robustness()
        
        # === Hallucination Rate ===
        metrics.hallucination_rate = self.evaluate_hallucination_rate(results)
        
        # === False Positive Rate ===
        metrics.false_positive_rate = self.evaluate_false_positive_rate()
        
        # === Latency Statistics ===
        metrics.latency_stats = self.latency_stats.get_stats()
        
        # === Targets ===
        metrics.targets = {
            'recall_at_3_target_90': metrics.recall_at_3 >= 0.90,
            'recall_at_1_target_75': metrics.recall_at_1 >= 0.75,
            'mrr_above_70': metrics.mrr >= 0.70,
            'hallucination_below_10': metrics.hallucination_rate <= 0.10,
            'false_positive_below_20': metrics.false_positive_rate <= 0.20
        }
        
        return metrics
    
    def _print_report(self, metrics: ComprehensiveMetrics):
        """Print comprehensive evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        
        print("\n📊 RECALL METRICS")
        print("-" * 50)
        print(f"  Recall@1:  {metrics.recall_at_1:.1%}  (target: ≥75%)")
        print(f"  Recall@3:  {metrics.recall_at_3:.1%}  (target: ≥90%)")
        print(f"  Recall@5:  {metrics.recall_at_5:.1%}")
        
        print("\n🎯 ACCURACY METRICS")
        print("-" * 50)
        print(f"  Top-1 Accuracy:  {metrics.top_1_accuracy:.1%}")
        print(f"  Top-3 Accuracy:  {metrics.top_3_accuracy:.1%}")
        print(f"  Top-5 Accuracy:  {metrics.top_5_accuracy:.1%}")
        
        print("\n📈 RANKING METRICS")
        print("-" * 50)
        print(f"  MRR:       {metrics.mrr:.3f}")
        print(f"  nDCG@3:    {metrics.ndcg_at_3:.3f}")
        print(f"  nDCG@5:    {metrics.ndcg_at_5:.3f}")
        
        print("\n🔬 QUALITY METRICS")
        print("-" * 50)
        print(f"  Entity Coverage:        {metrics.entity_coverage:.1%}")
        print(f"  Paraphrase Robustness:  {metrics.paraphrase_robustness:.1%}")
        print(f"  Hallucination Rate:     {metrics.hallucination_rate:.1%}")
        print(f"  False Positive Rate:    {metrics.false_positive_rate:.1%}")
        
        print("\n⏱️  STAGE-WISE LATENCY (ms)")
        print("-" * 50)
        lat = metrics.latency_stats
        if lat:
            print(f"  {'Stage':<15} {'Avg':>10} {'P50':>10} {'P95':>10}")
            print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
            print(f"  {'Embedding':<15} {lat.get('embedding_avg', 0):>10.1f} "
                  f"{lat.get('embedding_p50', 0):>10.1f} {lat.get('embedding_p95', 0):>10.1f}")
            print(f"  {'Retrieval':<15} {lat.get('retrieval_avg', 0):>10.1f} "
                  f"{lat.get('retrieval_p50', 0):>10.1f} {lat.get('retrieval_p95', 0):>10.1f}")
            print(f"  {'Reranking':<15} {lat.get('reranking_avg', 0):>10.1f} "
                  f"{lat.get('reranking_p50', 0):>10.1f} {lat.get('reranking_p95', 0):>10.1f}")
            print(f"  {'Total':<15} {lat.get('total_avg', 0):>10.1f} "
                  f"{lat.get('total_p50', 0):>10.1f} {lat.get('total_p95', 0):>10.1f}")
        
        print("\n✅ TARGET COMPLIANCE")
        print("-" * 50)
        for target, met in metrics.targets.items():
            status = "✅ PASS" if met else "❌ FAIL"
            print(f"  {status}: {target.replace('_', ' ')}")
        
        print("\n" + "=" * 70)
    
    def save_report(self, metrics: ComprehensiveMetrics, output_path: str = "comprehensive_evaluation.json"):
        """Save evaluation report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation of Document Intelligence System")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per query")
    parser.add_argument("--output", default="comprehensive_evaluation.json", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(api_url=args.api_url)
    metrics = evaluator.run_evaluation(num_runs=args.runs, verbose=args.verbose)
    
    if metrics:
        evaluator.save_report(metrics, args.output)


if __name__ == "__main__":
    main()
