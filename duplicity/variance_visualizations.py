"""
Variance visualization module for ConsistencyAI v2.

This module provides functions for creating visualizations that compare
control experiment variance (within-model consistency) against main experiment
variance (across-persona sensitivity).

The key insight: Models should be consistent with themselves (high control similarity)
while still adapting to different personas (lower persona similarity = persona-aware).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def prepare_comparison_data(
    control_similarities: Dict,
    persona_similarities: Dict
) -> pd.DataFrame:
    """
    Prepare data for variance comparison visualizations.

    Args:
        control_similarities: Dict mapping model names to dict of topics to similarity matrices
        persona_similarities: Dict mapping model names to dict of topics to similarity matrices

    Returns:
        DataFrame with columns: model, control_mean, persona_mean, sensitivity_gap
    """
    data = []

    for model in control_similarities.keys():
        if model in persona_similarities:
            # Flatten all similarity values across all topics for this model
            control_values = []
            for topic, matrix in control_similarities[model].items():
                # Extract upper triangle values (excluding diagonal) from similarity matrix
                control_values.extend(matrix[np.triu_indices_from(matrix, k=1)])
            
            persona_values = []
            for topic, matrix in persona_similarities[model].items():
                # Extract upper triangle values (excluding diagonal) from similarity matrix
                persona_values.extend(matrix[np.triu_indices_from(matrix, k=1)])
            
            # Compute means
            control_mean = np.mean(control_values) if control_values else 0.0
            persona_mean = np.mean(persona_values) if persona_values else 0.0
            sensitivity_gap = control_mean - persona_mean

            data.append({
                'model': model,
                'control_mean': control_mean,
                'persona_mean': persona_mean,
                'sensitivity_gap': sensitivity_gap
            })

    return pd.DataFrame(data)


def create_quadrant_plot(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
) -> None:
    """
    Create a quadrant plot showing model positioning in consistency/persona-sensitivity space.

    The ideal zone is high consistency (high control similarity) and persona-aware
    (lower persona similarity, showing larger sensitivity gap).

    Args:
        df: DataFrame from prepare_comparison_data()
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot colored by sensitivity gap
    scatter = ax.scatter(
        df['control_mean'],
        df['persona_mean'],
        c=df['sensitivity_gap'],
        cmap='RdYlGn',
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )

    # Add diagonal line (control = persona)
    min_val = min(df['control_mean'].min(), df['persona_mean'].min())
    max_val = max(df['control_mean'].max(), df['persona_mean'].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.3, label='Control = Persona')

    # Add quadrant divisions
    control_median = df['control_mean'].median()
    persona_median = df['persona_mean'].median()
    ax.axhline(y=persona_median, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=control_median, color='gray', linestyle=':', alpha=0.5)

    # Label ideal zone
    ax.text(
        0.98, 0.02,
        'IDEAL ZONE:\nHigh Consistency\n+ Persona-Aware',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )

    # Annotate models
    for idx, row in df.iterrows():
        ax.annotate(
            row['model'].split('/')[-1],  # Just model name
            (row['control_mean'], row['persona_mean']),
            fontsize=8,
            alpha=0.7
        )

    ax.set_xlabel('Control Similarity (Within-Model Consistency)', fontsize=12)
    ax.set_ylabel('Persona Similarity (Across-Persona)', fontsize=12)
    ax.set_title('Model Positioning: Consistency vs Persona-Sensitivity',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sensitivity Gap (Control - Persona)', fontsize=10)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def create_sensitivity_ranking(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 12),
    dpi: int = 300
) -> None:
    """
    Create a horizontal bar chart ranking models by persona-awareness.

    Green bars = more persona-aware (positive gap)
    Red bars = persona-agnostic (negative gap)

    Args:
        df: DataFrame from prepare_comparison_data()
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
    """
    df_sorted = df.sort_values('sensitivity_gap')

    fig, ax = plt.subplots(figsize=figsize)

    # Color bars based on positive/negative gap
    colors = ['green' if x > 0 else 'red' for x in df_sorted['sensitivity_gap']]

    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['sensitivity_gap'], color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.split('/')[-1] for m in df_sorted['model']], fontsize=9)
    ax.set_xlabel('Sensitivity Gap (Control - Persona)', fontsize=12)
    ax.set_title('Model Persona-Awareness Ranking', fontsize=14, fontweight='bold')

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Add legend
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Persona-Aware')
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Persona-Agnostic')
    ax.legend(handles=[green_patch, red_patch])

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def create_landscape_zones(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 300
) -> None:
    """
    Create a 2D landscape with colored zones showing different model behaviors.

    Zones:
    - Ideal (lightgreen): High consistency + Persona-aware
    - Problematic (lightcoral): Low consistency + Persona-agnostic
    - Other zones colored accordingly

    Args:
        df: DataFrame from prepare_comparison_data()
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate medians for zone divisions
    control_median = df['control_mean'].median()
    persona_median = df['persona_mean'].median()

    # Define zones with Rectangle patches
    # Zone 1: Low Control, High Persona (bottom-left) - yellow
    zone1 = Rectangle(
        (df['control_mean'].min(), df['persona_mean'].min()),
        control_median - df['control_mean'].min(),
        persona_median - df['persona_mean'].min(),
        facecolor='yellow', alpha=0.2, label='Low Consistency, Persona-Aware'
    )
    ax.add_patch(zone1)

    # Zone 2: High Control, Low Persona (top-right) - lightgreen (IDEAL)
    zone2 = Rectangle(
        (control_median, persona_median),
        df['control_mean'].max() - control_median,
        df['persona_mean'].max() - persona_median,
        facecolor='lightgreen', alpha=0.2, label='IDEAL: Consistent & Persona-Aware'
    )
    ax.add_patch(zone2)

    # Zone 3: Low Control, Low Persona (bottom-right) - lightcoral
    zone3 = Rectangle(
        (control_median, df['persona_mean'].min()),
        df['control_mean'].max() - control_median,
        persona_median - df['persona_mean'].min(),
        facecolor='lightcoral', alpha=0.2, label='Low Consistency, Persona-Agnostic'
    )
    ax.add_patch(zone3)

    # Zone 4: High Control, High Persona (top-left) - lightyellow
    zone4 = Rectangle(
        (df['control_mean'].min(), persona_median),
        control_median - df['control_mean'].min(),
        df['persona_mean'].max() - persona_median,
        facecolor='lightyellow', alpha=0.2, label='Consistent, Persona-Agnostic'
    )
    ax.add_patch(zone4)

    # Scatter plot of models
    scatter = ax.scatter(
        df['control_mean'],
        df['persona_mean'],
        c=df['sensitivity_gap'],
        cmap='RdYlGn',
        s=200,
        alpha=0.8,
        edgecolors='black',
        linewidth=1.5,
        zorder=10
    )

    # Annotate models
    for idx, row in df.iterrows():
        ax.annotate(
            row['model'].split('/')[-1],
            (row['control_mean'], row['persona_mean']),
            fontsize=8,
            alpha=0.9,
            zorder=11
        )

    ax.set_xlabel('Control Similarity (Consistency)', fontsize=12)
    ax.set_ylabel('Persona Similarity', fontsize=12)
    ax.set_title('Model Landscape with Behavioral Zones', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sensitivity Gap', fontsize=10)

    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def create_distribution_violin(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 300
) -> None:
    """
    Create split violin plots comparing control vs persona distributions.

    Args:
        df: DataFrame from prepare_comparison_data()
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for violin plot
    control_data = df['control_mean'].values
    persona_data = df['persona_mean'].values

    # Create violin plot
    parts = ax.violinplot(
        [control_data, persona_data],
        positions=[1, 2],
        widths=0.7,
        showmeans=True,
        showmedians=True
    )

    # Color the violin parts
    for i, pc in enumerate(parts['bodies']):
        if i == 0:
            pc.set_facecolor('blue')
        else:
            pc.set_facecolor('orange')
        pc.set_alpha(0.6)

    # Overlay box plots for quartiles
    bp = ax.boxplot(
        [control_data, persona_data],
        positions=[1, 2],
        widths=0.3,
        patch_artist=True,
        showfliers=False
    )

    for patch, color in zip(bp['boxes'], ['blue', 'orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Control\n(Within-Model)', 'Persona\n(Across-Persona)'], fontsize=11)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_title('Distribution Comparison: Control vs Persona Variance',
                 fontsize=14, fontweight='bold')

    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def create_consistency_variability_plot(
    control_similarities: Dict,
    persona_similarities: Dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
) -> None:
    """
    Create scatter plot showing consistency vs variability trade-off.

    X-axis: Average similarity (higher = more consistent)
    Y-axis: Standard deviation (higher = more variable)

    Args:
        control_similarities: Dict mapping model names to dict of topics to similarity matrices
        persona_similarities: Dict mapping model names to dict of topics to similarity matrices
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
    """
    data = []

    for model in control_similarities.keys():
        if model in persona_similarities:
            # Flatten all similarity values across all topics for this model
            control_values = []
            for topic, matrix in control_similarities[model].items():
                # Extract upper triangle values (excluding diagonal) from similarity matrix
                control_values.extend(matrix[np.triu_indices_from(matrix, k=1)])
            
            persona_values = []
            for topic, matrix in persona_similarities[model].items():
                # Extract upper triangle values (excluding diagonal) from similarity matrix
                persona_values.extend(matrix[np.triu_indices_from(matrix, k=1)])
            
            control_mean = np.mean(control_values) if control_values else 0.0
            persona_mean = np.mean(persona_values) if persona_values else 0.0

            # Calculate overall average and variability
            all_sims = control_values + persona_values
            avg_similarity = np.mean(all_sims) if all_sims else 0.0
            avg_variability = np.std(all_sims) if all_sims else 0.0
            sensitivity_gap = control_mean - persona_mean

            data.append({
                'model': model,
                'avg_similarity': avg_similarity,
                'avg_variability': avg_variability,
                'sensitivity_gap': sensitivity_gap
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(
        df['avg_similarity'],
        df['avg_variability'],
        c=df['sensitivity_gap'],
        cmap='RdYlGn',
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )

    # Annotate models
    for idx, row in df.iterrows():
        ax.annotate(
            row['model'].split('/')[-1],
            (row['avg_similarity'], row['avg_variability']),
            fontsize=8,
            alpha=0.7
        )

    # Highlight top 3 most consistent (high similarity, low variability)
    df_sorted_consistent = df.sort_values('avg_variability')
    for i in range(min(3, len(df_sorted_consistent))):
        row = df_sorted_consistent.iloc[i]
        ax.scatter(
            row['avg_similarity'],
            row['avg_variability'],
            s=400,
            facecolors='none',
            edgecolors='green',
            linewidths=3,
            label='Most Consistent' if i == 0 else ''
        )

    # Highlight top 3 least consistent (high variability)
    df_sorted_variable = df.sort_values('avg_variability', ascending=False)
    for i in range(min(3, len(df_sorted_variable))):
        row = df_sorted_variable.iloc[i]
        ax.scatter(
            row['avg_similarity'],
            row['avg_variability'],
            s=400,
            facecolors='none',
            edgecolors='red',
            linewidths=3,
            label='Least Consistent' if i == 0 else ''
        )

    ax.set_xlabel('Average Similarity (Higher = More Consistent)', fontsize=12)
    ax.set_ylabel('Standard Deviation (Higher = More Variable)', fontsize=12)
    ax.set_title('Consistency vs Variability Trade-off', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sensitivity Gap', fontsize=10)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def create_all_variance_plots(
    control_similarities: Dict,
    persona_similarities: Dict,
    output_dir: Optional[str] = None
) -> None:
    """
    Create all variance comparison visualizations at once.

    Args:
        control_similarities: Dict mapping model names to control similarity scores
        persona_similarities: Dict mapping model names to persona similarity scores
        output_dir: Directory to save plots (optional, will create if doesn't exist)
    """
    # Prepare data
    df = prepare_comparison_data(control_similarities, persona_similarities)

    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create all plots
    print("Creating quadrant plot...")
    create_quadrant_plot(
        df,
        output_path=f"{output_dir}/1_quadrant_plot.png" if output_dir else None
    )

    print("Creating sensitivity ranking...")
    create_sensitivity_ranking(
        df,
        output_path=f"{output_dir}/2_sensitivity_ranking.png" if output_dir else None
    )

    print("Creating landscape zones...")
    create_landscape_zones(
        df,
        output_path=f"{output_dir}/3_landscape_zones.png" if output_dir else None
    )

    print("Creating distribution violin plot...")
    create_distribution_violin(
        df,
        output_path=f"{output_dir}/4_distribution_violin.png" if output_dir else None
    )

    print("Creating consistency-variability plot...")
    create_consistency_variability_plot(
        control_similarities,
        persona_similarities,
        output_path=f"{output_dir}/5_consistency_variability.png" if output_dir else None
    )

    print("All variance plots created successfully!")
