"""
Central Analysis Module
Compute weighted consistency metrics across models and topics.
"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def mean_offdiag_and_pairs(S: np.ndarray) -> Tuple[float, int]:
    """Compute mean off-diagonal similarity and pair count."""
    if S is None or not isinstance(S, np.ndarray) or S.size == 0:
        return float("nan"), 0
    n = S.shape[0]
    if n <= 1:
        return float("nan"), 0
    mask = ~np.eye(n, dtype=bool)
    vals = S[mask]
    pairs = (n * (n - 1)) // 2
    return float(np.nanmean(vals)), int(pairs)


def compute_central_analysis(
    all_similarity_matrices: Dict[str, Dict[str, np.ndarray]],
    output_dir: str = "output/analysis"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Compute central analysis metrics from similarity matrices.
    
    Args:
        all_similarity_matrices: Dict[model][topic] -> similarity matrix
        output_dir: Directory to save output files
        
    Returns:
        Tuple of (per_model_per_topic, model_overall_weighted, 
                  topic_across_models_weighted, benchmark)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Per-model / per-topic within-model mean similarity
    rows = []
    for model, topic_map in all_similarity_matrices.items():
        for topic, S in (topic_map or {}).items():
            ms, pairs = mean_offdiag_and_pairs(S)
            rows.append({
                "model": model,
                "topic": topic,
                "mean_response_similarity": ms,
                "pair_count": pairs,
            })
    
    per_model_per_topic = (
        pd.DataFrame(rows)
        .sort_values(["model", "topic"])
        .reset_index(drop=True)
    )
    per_model_per_topic_csv = os.path.join(output_dir, "per_model_per_topic.csv")
    per_model_per_topic.to_csv(per_model_per_topic_csv, index=False)
    
    # 2) Weighted mean per model across topics
    overall_rows = []
    for model, g in per_model_per_topic.groupby("model", dropna=False):
        x = g["mean_response_similarity"].to_numpy()
        w = g["pair_count"].to_numpy()
        mask = (~np.isnan(x)) & (w > 0)
        if mask.any():
            weighted_mean = float(np.average(x[mask], weights=w[mask]))
            total_pairs = int(w[mask].sum())
            topics_count = int(mask.sum())
        else:
            weighted_mean, total_pairs, topics_count = float("nan"), 0, 0
        overall_rows.append({
            "model": model,
            "weighted_mean_similarity": weighted_mean,
            "topics_count": topics_count,
            "total_pairs": total_pairs,
        })
    
    model_overall_weighted = (
        pd.DataFrame(overall_rows)
        .sort_values("weighted_mean_similarity", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    model_overall_weighted_csv = os.path.join(output_dir, "model_overall_weighted.csv")
    model_overall_weighted.to_csv(model_overall_weighted_csv, index=False)
    
    # 3) Response-weighted topic consistency across models
    topic_rows = []
    for topic, g in per_model_per_topic.groupby("topic", dropna=False):
        x = g["mean_response_similarity"].to_numpy()
        w = g["pair_count"].to_numpy()
        mask = (~np.isnan(x)) & (w > 0)
        if mask.any():
            topic_weighted = float(np.average(x[mask], weights=w[mask]))
            models_contributing = int(mask.sum())
            total_pairs = int(w[mask].sum())
        else:
            topic_weighted, models_contributing, total_pairs = float("nan"), 0, 0
        topic_rows.append({
            "topic": topic,
            "response_weighted_topic_consistency": topic_weighted,
            "models_contributing": models_contributing,
            "total_response_pairs": total_pairs,
        })
    
    topic_across_models_weighted = (
        pd.DataFrame(topic_rows)
        .sort_values(["response_weighted_topic_consistency"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    topic_across_models_weighted_csv = os.path.join(output_dir, "topic_across_models_weighted.csv")
    topic_across_models_weighted.to_csv(topic_across_models_weighted_csv, index=False)
    
    # 4) Benchmark = arithmetic mean of models' weighted means
    benchmark = float(model_overall_weighted["weighted_mean_similarity"].mean(skipna=True))
    
    # Write summary.txt
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Benchmark (mean of models' weighted means): {benchmark:.6f}\n")
        f.write("\nTop 5 models by weighted mean similarity:\n")
        f.write(model_overall_weighted.head(5).to_string(index=False))
        f.write("\n\nTop 5 topics by response-weighted consistency:\n")
        f.write(topic_across_models_weighted.head(5).to_string(index=False))
    
    # Write README.md
    readme = """# ConsistencyAI Central Analysis Outputs

**Benchmark:** mean of the models' weighted mean similarities (computed from similarity matrices).
This follows the paper's convention of taking the arithmetic mean of model-level
weighted means as the study-wide benchmark threshold.

## Files Written
- **per_model_per_topic.csv** — Within-model mean response similarity for every (model, topic),
  plus `pair_count` used as weights (pair_count = N*(N-1)/2 for an N×N matrix).
- **model_overall_weighted.csv** — Weighted mean similarity per model across topics
  (weights = `pair_count` per (model, topic)), with totals.
- **topic_across_models_weighted.csv** — Response-weighted topic consistency across models
  (weights = `pair_count` for each (model, topic)).
- **summary.txt** — Benchmark value and top-5 leaderboards for models and topics.

## Notes
- Off-diagonal mean is used within each NxN similarity matrix.
- Missing/degenerate matrices are ignored in weighted averages.
- Outputs are sorted for readability.
"""
    
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    
    print(f"Central analysis complete!")
    print(f"\nFiles saved to: {output_dir}/")
    print(f"   - per_model_per_topic.csv")
    print(f"   - model_overall_weighted.csv")
    print(f"   - topic_across_models_weighted.csv")
    print(f"   - summary.txt")
    print(f"   - README.md")
    
    return per_model_per_topic, model_overall_weighted, topic_across_models_weighted, benchmark


def print_central_analysis_summary(
    model_overall_weighted: pd.DataFrame,
    topic_across_models_weighted: pd.DataFrame,
    benchmark: float
):
    """Print a formatted summary of central analysis results."""
    print("=" * 80)
    print("CENTRAL ANALYSIS RESULTS")
    print("=" * 80)
    print()
    print(f"Benchmark (mean of models' weighted means): {benchmark:.6f}")
    print()
    print("-" * 80)
    print("Top Models by Weighted Mean Similarity:")
    print("-" * 80)
    print()
    print(model_overall_weighted.head(5).to_string(index=False))
    print()
    print("-" * 80)
    print("Top Topics by Response-Weighted Consistency:")
    print("-" * 80)
    print()
    print(topic_across_models_weighted.head(5).to_string(index=False))
    print()
    print("=" * 80)

