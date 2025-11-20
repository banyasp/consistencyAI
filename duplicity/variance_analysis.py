"""Compare control variance vs. main experiment variance."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def compute_within_model_variance(
    control_results: Dict[str, Dict[str, Dict[str, str]]],
    control_similarities: Dict[str, Dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Calculate variance for each model across repetitions (control experiment).

    For each model:
    - Extract similarity matrix for the control topic (10x10 repetitions)
    - Calculate mean off-diagonal similarity
    - Calculate std dev of off-diagonal similarities

    Args:
        control_results: Results dict from control experiment
        control_similarities: Similarity matrices from control experiment

    Returns:
        DataFrame with columns: [model, mean_similarity, std_similarity, num_reps]
    """
    variance_data = []

    for model_name, topics in control_similarities.items():
        # Should only have one topic in control experiment
        if len(topics) == 0:
            continue

        topic_name = list(topics.keys())[0]
        similarity_matrix = topics[topic_name]

        # Calculate off-diagonal similarities
        n = similarity_matrix.shape[0]
        if n <= 1:
            continue

        # Create mask for off-diagonal elements
        mask = ~np.eye(n, dtype=bool)
        off_diag_values = similarity_matrix[mask]

        # Calculate statistics
        mean_sim = float(np.mean(off_diag_values))
        std_sim = float(np.std(off_diag_values))

        variance_data.append({
            "model": model_name,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "num_reps": n
        })

    return pd.DataFrame(variance_data)


def compute_across_persona_variance(
    main_results: Dict[str, Dict[str, Dict[str, str]]],
    main_similarities: Dict[str, Dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Calculate variance for each model across personas (main experiment).

    For each model/topic:
    - Get existing NxN similarity matrix (N personas)
    - Calculate mean off-diagonal similarity
    - Calculate std dev of off-diagonal similarities
    - Aggregate across all topics for model-level metric

    Args:
        main_results: Results dict from main experiment
        main_similarities: Similarity matrices from main experiment

    Returns:
        DataFrame with columns: [model, mean_similarity, std_similarity, num_topics, avg_personas]
    """
    # First, calculate per-topic statistics
    per_topic_data = []

    for model_name, topics in main_similarities.items():
        for topic_name, similarity_matrix in topics.items():
            # Calculate off-diagonal similarities
            n = similarity_matrix.shape[0]
            if n <= 1:
                continue

            # Create mask for off-diagonal elements
            mask = ~np.eye(n, dtype=bool)
            off_diag_values = similarity_matrix[mask]

            # Calculate statistics
            mean_sim = float(np.mean(off_diag_values))
            std_sim = float(np.std(off_diag_values))

            per_topic_data.append({
                "model": model_name,
                "topic": topic_name,
                "mean_similarity": mean_sim,
                "std_similarity": std_sim,
                "num_personas": n
            })

    per_topic_df = pd.DataFrame(per_topic_data)

    # Aggregate across topics for each model
    model_variance = []

    for model_name, group in per_topic_df.groupby("model"):
        # Average the mean similarities across topics
        avg_mean_sim = group["mean_similarity"].mean()
        # Average the std deviations across topics
        avg_std_sim = group["std_similarity"].mean()
        num_topics = len(group)
        avg_personas = group["num_personas"].mean()

        model_variance.append({
            "model": model_name,
            "mean_similarity": avg_mean_sim,
            "std_similarity": avg_std_sim,
            "num_topics": num_topics,
            "avg_personas": avg_personas
        })

    return pd.DataFrame(model_variance)


def create_variance_comparison_visualizations(
    control_variance: pd.DataFrame,
    persona_variance: pd.DataFrame,
    output_dir: str = "output/variance_comparison"
) -> None:
    """Create comprehensive variance comparison plots.

    Visualizations:
    1. Bar chart: Control vs. Persona mean similarity per model
    2. Scatter plot: Control mean (X) vs. Persona mean (Y)
    3. Heatmap: Both variance types across models
    4. Box plots: Distribution of similarities

    Args:
        control_variance: DataFrame from compute_within_model_variance
        persona_variance: DataFrame from compute_across_persona_variance
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Merge dataframes for comparison
    control_var = control_variance.copy()
    control_var["experiment"] = "Control (Within-Model)"
    control_var = control_var.rename(columns={"mean_similarity": "similarity"})

    persona_var = persona_variance.copy()
    persona_var["experiment"] = "Main (Across-Persona)"
    persona_var = persona_var.rename(columns={"mean_similarity": "similarity"})

    # === VISUALIZATION 1: Bar Chart Comparison ===
    fig, ax = plt.subplots(figsize=(14, 8))

    # Merge on model for side-by-side comparison
    merged = control_variance.merge(
        persona_variance,
        on="model",
        suffixes=("_control", "_persona")
    )

    x = np.arange(len(merged))
    width = 0.35

    bars1 = ax.bar(x - width/2, merged["mean_similarity_control"],
                   width, label="Control (Within-Model)", color="steelblue")
    bars2 = ax.bar(x + width/2, merged["mean_similarity_persona"],
                   width, label="Main (Across-Persona)", color="coral")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title("Control vs. Persona Variance: Mean Similarity Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["model"], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_bar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === VISUALIZATION 2: Scatter Plot ===
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(merged["mean_similarity_control"],
               merged["mean_similarity_persona"],
               s=100, alpha=0.6, c="purple")

    # Add model labels
    for idx, row in merged.iterrows():
        ax.annotate(row["model"],
                   (row["mean_similarity_control"], row["mean_similarity_persona"]),
                   fontsize=8, alpha=0.7, xytext=(5, 5),
                   textcoords="offset points")

    # Add diagonal line (y=x)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.4, zorder=0, label="y=x")

    ax.set_xlabel("Control Variance (Within-Model Similarity)", fontsize=12)
    ax.set_ylabel("Persona Variance (Across-Persona Similarity)", fontsize=12)
    ax.set_title("Control vs. Persona Variance: Scatter Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === VISUALIZATION 3: Heatmap ===
    # Prepare data for heatmap
    heatmap_data = merged[[
        "model",
        "mean_similarity_control",
        "std_similarity_control",
        "mean_similarity_persona",
        "std_similarity_persona"
    ]].set_index("model")

    heatmap_data.columns = [
        "Control Mean",
        "Control Std",
        "Persona Mean",
        "Persona Std"
    ]

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
                cbar_kws={"label": "Similarity Value"}, ax=ax)
    ax.set_title("Variance Metrics Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === VISUALIZATION 4: Box Plots ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Control variance distribution
    axes[0].boxplot([control_variance["mean_similarity"]],
                    labels=["Control (Within-Model)"])
    axes[0].set_ylabel("Mean Cosine Similarity", fontsize=12)
    axes[0].set_title("Control Variance Distribution", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Persona variance distribution
    axes[1].boxplot([persona_variance["mean_similarity"]],
                    labels=["Main (Across-Persona)"])
    axes[1].set_ylabel("Mean Cosine Similarity", fontsize=12)
    axes[1].set_title("Persona Variance Distribution", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Variance Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/variance_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f" Visualizations saved to: {output_dir}/")


def generate_variance_report(
    control_variance: pd.DataFrame,
    persona_variance: pd.DataFrame,
    output_path: str = "output/variance_comparison/report.txt"
) -> str:
    """Generate text report with key findings.

    Report includes:
    - Summary statistics for control variance
    - Summary statistics for persona variance
    - Top 5 most consistent models (high control similarity, low std)
    - Top 5 most persona-sensitive models (high difference between persona and control)
    - Interesting patterns

    Args:
        control_variance: DataFrame from compute_within_model_variance
        persona_variance: DataFrame from compute_across_persona_variance
        output_path: Path to save report

    Returns:
        Report text as string
    """
    # Merge dataframes
    merged = control_variance.merge(
        persona_variance,
        on="model",
        suffixes=("_control", "_persona")
    )

    # Calculate difference (persona - control)
    merged["persona_sensitivity"] = (
        merged["mean_similarity_persona"] - merged["mean_similarity_control"]
    )

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("VARIANCE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Control variance summary
    report.append("CONTROL VARIANCE (Within-Model Consistency)")
    report.append("-" * 80)
    report.append(f"Mean similarity across all models: {control_variance['mean_similarity'].mean():.4f}")
    report.append(f"Std deviation across models: {control_variance['mean_similarity'].std():.4f}")
    report.append(f"Min similarity: {control_variance['mean_similarity'].min():.4f}")
    report.append(f"Max similarity: {control_variance['mean_similarity'].max():.4f}")
    report.append("")

    # Persona variance summary
    report.append("PERSONA VARIANCE (Across-Persona Consistency)")
    report.append("-" * 80)
    report.append(f"Mean similarity across all models: {persona_variance['mean_similarity'].mean():.4f}")
    report.append(f"Std deviation across models: {persona_variance['mean_similarity'].std():.4f}")
    report.append(f"Min similarity: {persona_variance['mean_similarity'].min():.4f}")
    report.append(f"Max similarity: {persona_variance['mean_similarity'].max():.4f}")
    report.append("")

    # Top 5 most internally consistent models
    report.append("TOP 5 MOST INTERNALLY CONSISTENT MODELS")
    report.append("(High control similarity = Low randomness)")
    report.append("-" * 80)
    top_consistent = control_variance.nlargest(5, "mean_similarity")
    for idx, row in top_consistent.iterrows():
        report.append(f"{idx+1}. {row['model']}: {row['mean_similarity']:.4f} "
                     f"(std: {row['std_similarity']:.4f})")
    report.append("")

    # Top 5 most persona-sensitive models
    report.append("TOP 5 MOST PERSONA-SENSITIVE MODELS")
    report.append("(High persona variance - control variance = Strong persona effect)")
    report.append("-" * 80)
    top_sensitive = merged.nlargest(5, "persona_sensitivity")
    for idx, row in top_sensitive.iterrows():
        report.append(f"{idx+1}. {row['model']}: "
                     f"Persona={row['mean_similarity_persona']:.4f}, "
                     f"Control={row['mean_similarity_control']:.4f}, "
                     f"Diff={row['persona_sensitivity']:.4f}")
    report.append("")

    # Interesting patterns
    report.append("INTERPRETATION")
    report.append("-" * 80)
    report.append("• High control variance (low similarity) = Model is noisy/random")
    report.append("• Low control variance (high similarity) = Model is consistent")
    report.append("• High persona variance (low similarity) = Strong persona differentiation")
    report.append("• Low persona variance (high similarity) = Persona-agnostic responses")
    report.append("")
    report.append("IDEAL PROFILE:")
    report.append("• High control similarity (consistent)")
    report.append("• Lower persona similarity (persona-aware)")
    report.append("• This indicates: Consistent behavior but responds differently to personas")
    report.append("")

    report.append("=" * 80)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_text = "\n".join(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f" Report saved to: {output_path}")
    return report_text
