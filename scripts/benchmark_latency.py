"""
Latency Benchmark Script for Real Estate Document Intelligence System

Measures:
- Stage-wise latency breakdown (embedding, retrieval, reranking, generation)
- Before/after optimization comparison
- Cache performance impact
- Re-ranking latency vs accuracy trade-off

Usage:
    python scripts/benchmark_latency.py --api-url http://localhost:8000
"""

import time
import argparse
import requests
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    query: str
    total_ms: float
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    formatting_ms: float = 0.0
    cached: bool = False
    use_reranking: bool = False


@dataclass
class BenchmarkResults:
    """Benchmark results summary."""
    # Baseline (no optimizations)
    baseline_avg_ms: float = 0.0
    baseline_p50_ms: float = 0.0
    baseline_p95_ms: float = 0.0
    baseline_p99_ms: float = 0.0
    
    # With embedding cache
    cached_avg_ms: float = 0.0
    cached_p50_ms: float = 0.0
    cached_p95_ms: float = 0.0
    
    # With reranking
    reranking_avg_ms: float = 0.0
    reranking_overhead_ms: float = 0.0
    
    # Stage breakdown (averages)
    embedding_avg_ms: float = 0.0
    retrieval_avg_ms: float = 0.0
    reranking_avg_ms: float = 0.0
    formatting_avg_ms: float = 0.0
    
    # Improvement metrics
    improvement_percent: float = 0.0
    target_50_percent_met: bool = False
    
    # Cache stats
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LatencyBenchmark:
    """
    Latency benchmark for document intelligence system.
    """
    
    SAMPLE_QUERIES = [
        "What is the price of the property?",
        "How many bedrooms?",
        "What schools are nearby?",
        "Is there a garage?",
        "What is the square footage?",
        "Are there any HOA fees?",
        "What amenities are available?",
        "What year was it built?",
        "Is there a swimming pool?",
        "What is the address?",
        "How many bathrooms does the house have?",
        "What is the lot size?",
        "Is there a basement?",
        "What type of heating system?",
        "What is the property tax?",
    ]
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.measurements: List[LatencyMeasurement] = []
    
    def _search(
        self,
        query: str,
        use_reranking: bool = False,
        top_k: int = 5
    ) -> Optional[Dict]:
        """Execute search and return response with latency breakdown."""
        try:
            start = time.time()
            response = requests.post(
                f"{self.api_url}/api/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "use_reranking": use_reranking
                },
                timeout=30
            )
            external_latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                data['external_latency_ms'] = external_latency
                return data
            return None
            
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def _clear_cache(self):
        """Clear cache to measure cold performance."""
        try:
            # Try to clear via stats endpoint or a dedicated clear endpoint
            # For now, we'll just run with fresh queries
            pass
        except Exception:
            pass
    
    def measure_baseline(self, num_iterations: int = 3) -> List[LatencyMeasurement]:
        """Measure baseline latency (cold cache, no reranking)."""
        print("\n📊 Measuring baseline latency (cold cache)...")
        measurements = []
        
        # Use unique queries to avoid cache hits
        queries = [f"{q} #{i}" for i, q in enumerate(self.SAMPLE_QUERIES)]
        
        for iteration in range(num_iterations):
            for query in queries:
                result = self._search(query, use_reranking=False)
                
                if result:
                    breakdown = result.get('latency_breakdown', {})
                    measurement = LatencyMeasurement(
                        query=query,
                        total_ms=result.get('query_time_ms', 0),
                        embedding_ms=breakdown.get('embedding_ms', 0),
                        retrieval_ms=breakdown.get('retrieval_ms', 0),
                        reranking_ms=breakdown.get('reranking_ms', 0),
                        formatting_ms=breakdown.get('formatting_ms', 0),
                        cached=result.get('cached', False),
                        use_reranking=False
                    )
                    measurements.append(measurement)
        
        self.measurements.extend(measurements)
        return measurements
    
    def measure_cached(self, num_iterations: int = 3) -> List[LatencyMeasurement]:
        """Measure cached latency (warm cache)."""
        print("\n📊 Measuring cached latency (warm cache)...")
        measurements = []
        
        # First pass to populate cache
        for query in self.SAMPLE_QUERIES:
            self._search(query, use_reranking=False)
        
        # Measure cached responses
        for iteration in range(num_iterations):
            for query in self.SAMPLE_QUERIES:
                result = self._search(query, use_reranking=False)
                
                if result:
                    breakdown = result.get('latency_breakdown', {})
                    measurement = LatencyMeasurement(
                        query=query,
                        total_ms=result.get('query_time_ms', 0),
                        embedding_ms=breakdown.get('embedding_ms', 0),
                        retrieval_ms=breakdown.get('retrieval_ms', 0),
                        reranking_ms=breakdown.get('reranking_ms', 0),
                        formatting_ms=breakdown.get('formatting_ms', 0),
                        cached=result.get('cached', False),
                        use_reranking=False
                    )
                    measurements.append(measurement)
        
        self.measurements.extend(measurements)
        return measurements
    
    def measure_with_reranking(self, num_iterations: int = 2) -> List[LatencyMeasurement]:
        """Measure latency with re-ranking enabled."""
        print("\n📊 Measuring latency with re-ranking...")
        measurements = []
        
        # Use unique queries to avoid cache hits
        queries = [f"{q} rerank#{i}" for i, q in enumerate(self.SAMPLE_QUERIES[:5])]
        
        for iteration in range(num_iterations):
            for query in queries:
                result = self._search(query, use_reranking=True)
                
                if result:
                    breakdown = result.get('latency_breakdown', {})
                    measurement = LatencyMeasurement(
                        query=query,
                        total_ms=result.get('query_time_ms', 0),
                        embedding_ms=breakdown.get('embedding_ms', 0),
                        retrieval_ms=breakdown.get('retrieval_ms', 0),
                        reranking_ms=breakdown.get('reranking_ms', 0),
                        formatting_ms=breakdown.get('formatting_ms', 0),
                        cached=result.get('cached', False),
                        use_reranking=True
                    )
                    measurements.append(measurement)
        
        self.measurements.extend(measurements)
        return measurements
    
    def run_benchmark(self, iterations: int = 3) -> BenchmarkResults:
        """Run full benchmark suite."""
        print("\n" + "=" * 70)
        print("LATENCY BENCHMARK")
        print("=" * 70)
        
        # Check API health
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                print("ERROR: API is not healthy")
                return BenchmarkResults()
        except Exception as e:
            print(f"ERROR: Cannot connect to API: {e}")
            return BenchmarkResults()
        
        print(f"\nAPI URL: {self.api_url}")
        print(f"Iterations: {iterations}")
        print(f"Queries: {len(self.SAMPLE_QUERIES)}")
        print("-" * 70)
        
        results = BenchmarkResults()
        
        # 1. Measure baseline (cold cache)
        baseline_measurements = self.measure_baseline(iterations)
        baseline_times = [m.total_ms for m in baseline_measurements if not m.cached]
        
        if baseline_times:
            results.baseline_avg_ms = np.mean(baseline_times)
            results.baseline_p50_ms = np.percentile(baseline_times, 50)
            results.baseline_p95_ms = np.percentile(baseline_times, 95)
            results.baseline_p99_ms = np.percentile(baseline_times, 99)
            
            # Stage breakdown from baseline
            results.embedding_avg_ms = np.mean([m.embedding_ms for m in baseline_measurements if not m.cached])
            results.retrieval_avg_ms = np.mean([m.retrieval_ms for m in baseline_measurements if not m.cached])
            results.formatting_avg_ms = np.mean([m.formatting_ms for m in baseline_measurements if not m.cached])
        
        # 2. Measure cached performance
        cached_measurements = self.measure_cached(iterations)
        cached_times = [m.total_ms for m in cached_measurements if m.cached]
        
        if cached_times:
            results.cached_avg_ms = np.mean(cached_times)
            results.cached_p50_ms = np.percentile(cached_times, 50)
            results.cached_p95_ms = np.percentile(cached_times, 95)
            results.cache_hit_rate = len(cached_times) / len(cached_measurements)
        
        # 3. Measure with reranking
        try:
            rerank_measurements = self.measure_with_reranking(iterations)
            rerank_times = [m.total_ms for m in rerank_measurements]
            rerank_overhead = [m.reranking_ms for m in rerank_measurements]
            
            if rerank_times:
                results.reranking_avg_ms = np.mean(rerank_times)
                results.reranking_overhead_ms = np.mean(rerank_overhead)
                results.reranking_avg_ms = np.mean([m.reranking_ms for m in rerank_measurements])
        except Exception as e:
            print(f"Re-ranking benchmark skipped: {e}")
        
        # Calculate improvement
        if results.baseline_avg_ms > 0 and results.cached_avg_ms > 0:
            results.improvement_percent = (
                (results.baseline_avg_ms - results.cached_avg_ms) / results.baseline_avg_ms
            ) * 100
            results.target_50_percent_met = results.improvement_percent >= 50
        
        # Print report
        self._print_report(results)
        
        return results
    
    def _print_report(self, results: BenchmarkResults):
        """Print benchmark report."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        
        print("\n📊 STAGE-WISE LATENCY BREAKDOWN (Cold Cache)")
        print("-" * 50)
        print(f"  Embedding:    {results.embedding_avg_ms:>8.1f} ms")
        print(f"  Retrieval:    {results.retrieval_avg_ms:>8.1f} ms")
        print(f"  Formatting:   {results.formatting_avg_ms:>8.1f} ms")
        print(f"  {'─' * 25}")
        print(f"  Total:        {results.baseline_avg_ms:>8.1f} ms")
        
        print("\n📈 LATENCY COMPARISON")
        print("-" * 50)
        print(f"  {'Metric':<20} {'Cold Cache':>15} {'Warm Cache':>15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15}")
        print(f"  {'Average':.<20} {results.baseline_avg_ms:>14.1f}ms {results.cached_avg_ms:>14.1f}ms")
        print(f"  {'P50 (Median)':.<20} {results.baseline_p50_ms:>14.1f}ms {results.cached_p50_ms:>14.1f}ms")
        print(f"  {'P95':.<20} {results.baseline_p95_ms:>14.1f}ms {results.cached_p95_ms:>14.1f}ms")
        
        print("\n🚀 OPTIMIZATION IMPACT")
        print("-" * 50)
        print(f"  Improvement:       {results.improvement_percent:>6.1f}%")
        print(f"  Cache Hit Rate:    {results.cache_hit_rate:>6.1%}")
        print(f"  Target 50% Met:    {'✅ YES' if results.target_50_percent_met else '❌ NO'}")
        
        if results.reranking_overhead_ms > 0:
            print("\n⚖️  RE-RANKING TRADE-OFF")
            print("-" * 50)
            print(f"  Re-ranking Overhead:  {results.reranking_overhead_ms:>6.1f} ms")
            print(f"  Total with Reranking: {results.reranking_avg_ms:>6.1f} ms")
            rerank_cost_percent = (results.reranking_overhead_ms / results.baseline_avg_ms) * 100 if results.baseline_avg_ms > 0 else 0
            print(f"  Latency Cost:         {rerank_cost_percent:>6.1f}%")
        
        print("\n" + "=" * 70)
    
    def save_results(self, results: BenchmarkResults, output_path: str = "latency_benchmark.json"):
        """Save results to JSON file."""
        # Include individual measurements as well
        output = {
            'summary': results.to_dict(),
            'measurements': [asdict(m) for m in self.measurements]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Latency Benchmark for Document Intelligence System")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--output", default="latency_benchmark.json", help="Output file path")
    
    args = parser.parse_args()
    
    benchmark = LatencyBenchmark(api_url=args.api_url)
    results = benchmark.run_benchmark(iterations=args.iterations)
    benchmark.save_results(results, args.output)


if __name__ == "__main__":
    main()
