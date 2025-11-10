"""Plotting utilities and 3D embedding visualizer."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_similarity_matrix_with_values(
    similarity_matrix: np.ndarray,
    persona_ids: List[int],
    show_values: bool = True,
    value_format: str = ".2f",
    model_name: Optional[str] = None,
    topic: Optional[str] = None,
    save_path: Optional[str] = None,
):
    # Calculate appropriate figure size based on matrix dimensions
    matrix_size = len(persona_ids)
    if matrix_size <= 20:
        figsize = (12, 10)
    elif matrix_size <= 50:
        figsize = (20, 18)
    elif matrix_size <= 100:
        figsize = (30, 28)
    else:
        figsize = (40, 38)
    
    fig, ax = plt.subplots(figsize=figsize)
    # Use RdYlBu colormap (not reversed) for better differentiation in 0.78-1.0 range
    # This provides cool colors (blues) for high values and warm colors (reds/oranges) for low values
    cmap = plt.cm.RdYlBu

    # For large matrices, don't show individual values to avoid clutter
    if matrix_size > 50:
        show_values = False

    sns.heatmap(
        similarity_matrix,
        annot=show_values,
        fmt=value_format,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"label": "Similarity Score (0.78-1.0 = blue colors, 0/negative = red/orange colors)"},
        ax=ax,
    )

    # Optimize tick labels for large matrices
    if matrix_size <= 20:
        # Show all ticks for small matrices
        ax.set_xticks(np.arange(len(persona_ids)) + 0.5)
        ax.set_yticks(np.arange(len(persona_ids)) + 0.5)
        ax.set_xticklabels(persona_ids, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(persona_ids, rotation=0, fontsize=8)
    else:
        # For large matrices, show every 5th tick to avoid overcrowding
        step = max(1, matrix_size // 20)  # Show ~20 ticks max
        tick_positions = np.arange(0, len(persona_ids), step)
        ax.set_xticks(tick_positions + 0.5)
        ax.set_yticks(tick_positions + 0.5)
        ax.set_xticklabels([persona_ids[i] for i in tick_positions], rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels([persona_ids[i] for i in tick_positions], rotation=0, fontsize=10)

    # Add axis labels
    ax.set_xlabel("Persona", fontsize=14, fontweight='bold')
    ax.set_ylabel("Persona", fontsize=14, fontweight='bold')
    
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    title_parts: List[str] = ["Similarity matrix across personas"]
    if model_name:
        title_parts.append(f"Model: {model_name}")
    if topic:
        title_parts.append(f"Topic: {topic}")
    plt.title(" | ".join(title_parts), pad=20, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        import os
        dir_path = os.path.dirname(save_path)
        if dir_path:  # Only create directory if there's a path
            os.makedirs(dir_path, exist_ok=True)
        # Use higher DPI for large matrices
        dpi = 300 if matrix_size <= 50 else 400
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
    
    return fig, ax


def find_extreme_similarities_with_values(
    similarity_matrix: Union[List[List[float]], np.ndarray], persona_ids: List[int]
) -> Tuple[Tuple[int, int, float], Tuple[int, int, float]]:
    matrix = np.array(similarity_matrix)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    max_value = np.max(matrix[mask])
    max_indices = np.where(matrix == max_value)
    min_value = np.min(matrix[mask])
    min_indices = np.where(matrix == min_value)
    max_row, max_col = max_indices[0][0], max_indices[1][0]
    min_row, min_col = min_indices[0][0], min_indices[1][0]
    return (persona_ids[max_row], persona_ids[max_col], float(max_value)), (
        persona_ids[min_row],
        persona_ids[min_col],
        float(min_value),
    )


def print_extreme_similarities(similarity_matrix: Union[List[List[float]], np.ndarray], persona_ids: List[int]):
    max_pair_with_value, min_pair_with_value = find_extreme_similarities_with_values(
        similarity_matrix, persona_ids
    )
    print("\nThe personas who received the most similar responses from AI were:")
    print(f"{max_pair_with_value[0]} and {max_pair_with_value[1]} (similarity: {max_pair_with_value[2]:.6f})")
    print("\nThe personas who received the most different responses from AI were:")
    print(f"Least similar: {min_pair_with_value[0]} and {min_pair_with_value[1]} (similarity: {min_pair_with_value[2]:.6f})")


def plot_similarity_by_sphere(avg_scores: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    spheres = list(next(iter(avg_scores.values())).keys()) if avg_scores else []
    for sphere in spheres:
        models: List[str] = []
        scores: List[float] = []
        for model, sphere_scores in avg_scores.items():
            models.append(model)
            scores.append(sphere_scores[sphere])
        sorted_data = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
        if not sorted_data:
            continue
        sorted_models, sorted_scores = zip(*sorted_data)
        fig = plt.figure(figsize=(8, 4))
        plt.barh(sorted_models, sorted_scores, color="skyblue")
        plt.xlabel("Average Similarity")
        plt.title(f"Model Similarity in {sphere}")
        plt.gca().invert_yaxis()
        plt.xlim(0, 1)
        plt.tight_layout()
        
        # Save the figure if save_path is provided
        if save_path:
            # Create directory if it doesn't exist
            import os
            os.makedirs(save_path, exist_ok=True)
            # Create filename with sphere name
            safe_sphere = sphere.replace("/", "_").replace(":", "_").replace(" ", "_")
            filename = f"similarity_by_sphere_{safe_sphere}.png"
            full_path = os.path.join(save_path, filename)
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved similarity by sphere plot to: {full_path}")
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()


def plot_overall_leaderboard(
    avg_scores: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot a horizontal bar leaderboard of overall model average similarity.

    Args:
        avg_scores: Dict mapping model names to dicts of topic->score.
        save_path: Optional directory to save the plot as PNG.
        show: If True, always show the plot interactively (even if saving).
    """
    model_averages: Dict[str, float] = {}
    for model, topic_scores in avg_scores.items():
        model_averages[model] = float(np.mean(list(topic_scores.values()))) if topic_scores else 0.0
    sorted_data = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
    if not sorted_data:
        return
    sorted_models, sorted_scores = zip(*sorted_data)

    # Calculate height based on number of models (0.4 inches per model, min 6 inches)
    num_models = len(sorted_models)
    height = max(6, num_models * 0.4)
    fig = plt.figure(figsize=(10, height))
    plt.barh(sorted_models, sorted_scores, color="lightgreen")
    plt.xlabel("Overall Average Semantic Similarity")
    plt.title("Overall Model Consistency Across All Topics")
    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        filename = "overall_leaderboard.png"
        full_path = os.path.join(save_path, filename)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Saved overall leaderboard plot to: {full_path}")
        if show:
            plt.show()
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()


class Embedding3DVisualizer:
    """3D visualization tool for high-dimensional embeddings."""

    def __init__(self, embeddings: np.ndarray, ids: List[int], method: str = "tsne", random_state: int = 42):
        self.embeddings = np.array(embeddings)
        self.ids = ids
        self.method = method
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.reduced_3d: Optional[np.ndarray] = None
        if len(embeddings) != len(ids):
            raise ValueError("Number of embeddings must match number of IDs")
        print(f"Initialized with {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    def reduce_to_3d(self, method: Optional[str] = None, **kwargs) -> np.ndarray:
        method = method or self.method
        print(f"Reducing to 3D using {method.upper()}...")
        embeddings_scaled = self.scaler.fit_transform(self.embeddings)
        if method.lower() == "tsne":
            default_params = {
                "n_components": 3,
                "perplexity": min(30, len(self.embeddings) - 1) if len(self.embeddings) > 1 else 2,
                "random_state": self.random_state,
                "n_iter": 1000,
                "learning_rate": "auto",
            }
            default_params.update(kwargs)
            reducer = TSNE(**default_params)
            self.reduced_3d = reducer.fit_transform(embeddings_scaled)
        elif method.lower() == "pca":
            default_params = {"n_components": 3, "random_state": self.random_state}
            default_params.update(kwargs)
            reducer = PCA(**default_params)
            self.reduced_3d = reducer.fit_transform(embeddings_scaled)
            explained_variance = reducer.explained_variance_ratio_
            print(f"PCA explained variance ratios: {explained_variance}")
            print(f"Total explained variance: {sum(explained_variance):.3f}")
        elif method.lower() == "mds":
            default_params = {"n_components": 3, "random_state": self.random_state, "dissimilarity": "euclidean"}
            default_params.update(kwargs)
            reducer = MDS(**default_params)
            self.reduced_3d = reducer.fit_transform(embeddings_scaled)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'tsne', 'pca', or 'mds'")
        print(f"Reduction complete. 3D coordinates shape: {self.reduced_3d.shape}")
        return self.reduced_3d

    def create_interactive_3d_plot(
        self,
        title: str = "3D Embedding Visualization",
        color_by: Optional[List] = None,
        size_by: Optional[List] = None,
        hover_data: Optional[Dict[str, List]] = None,
        show_connections: bool = False,
        connection_threshold: float = 0.8,
    ) -> go.Figure:
        if self.reduced_3d is None:
            raise ValueError("Must call reduce_to_3d() first")
        df = pd.DataFrame({
            "x": self.reduced_3d[:, 0],
            "y": self.reduced_3d[:, 1],
            "z": self.reduced_3d[:, 2],
            "ID": self.ids,
        })
        if color_by is not None:
            df["color"] = color_by
        if size_by is not None:
            df["size"] = size_by
        else:
            df["size"] = 8
        fig = go.Figure()
        if color_by is not None and len(color_by) > 0 and isinstance(color_by[0], (int, float)):
            for i, row in df.iterrows():
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, row["x"]], y=[0, row["y"]], z=[0, row["z"]],
                        mode="lines+markers",
                        line=dict(color=row["color"], width=3, colorscale="Viridis"),
                        marker=dict(size=[0, row["size"]], color=[row["color"], row["color"]], colorscale="Viridis", showscale=(i == 0), colorbar=dict(title="Color Value") if i == 0 else None),
                        text=["Origin", row["ID"]],
                        hovertemplate='<b>%{text}</b><br><b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><b>Z:</b> %{z:.3f}<br><extra></extra>',
                        name=f'Vector {row["ID"]}' if i < 5 else None,
                        showlegend=i < 5,
                    )
                )
        elif color_by is not None and len(color_by) > 0:
            unique_colors = list(set(color_by))
            colors = px.colors.qualitative.Set3[: len(unique_colors)]
            for color_val in unique_colors:
                mask = df["color"] == color_val
                color_idx = unique_colors.index(color_val)
                for i, (idx, row) in enumerate(df[mask].iterrows()):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[0, row["x"]], y=[0, row["y"]], z=[0, row["z"]],
                            mode="lines+markers",
                            line=dict(color=colors[color_idx], width=3),
                            marker=dict(size=[0, row["size"]], color=[colors[color_idx], colors[color_idx]]),
                            text=["Origin", row["ID"]],
                            hovertemplate='<b>%{text}</b><br><b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><b>Z:</b> %{z:.3f}<br><extra></extra>',
                            name=f"Group: {color_val}" if i == 0 else None,
                            showlegend=i == 0,
                        )
                    )
        else:
            for i, row in df.iterrows():
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, row["x"]], y=[0, row["y"]], z=[0, row["z"]],
                        mode="lines+markers",
                        line=dict(color="blue", width=3),
                        opacity=0.7,
                        marker=dict(size=[0, row["size"]], color=["blue", "blue"], opacity=0.7),
                        text=["Origin", row["ID"]],
                        hovertemplate='<b>%{text}</b><br><b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><b>Z:</b> %{z:.3f}<br><extra></extra>',
                        name=f'Vector {row["ID"]}' if i < 5 else None,
                        showlegend=i < 5,
                    )
                )
        if show_connections and hasattr(self, "similarity_matrix"):
            connections_added = 0
            for i in range(len(self.ids)):
                for j in range(i + 1, len(self.ids)):
                    if self.similarity_matrix[i, j] >= connection_threshold:
                        fig.add_trace(
                            go.Scatter3d(
                                x=[self.reduced_3d[i, 0], self.reduced_3d[j, 0]],
                                y=[self.reduced_3d[i, 1], self.reduced_3d[j, 1]],
                                z=[self.reduced_3d[i, 2], self.reduced_3d[j, 2]],
                                mode="lines",
                                line=dict(color="gray", width=1),
                                opacity=0.3,
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                        connections_added += 1
            print(f"Added {connections_added} connections with threshold {connection_threshold}")
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            width=800,
            height=600,
            showlegend=True,
        )
        return fig

    def set_similarity_matrix(self, similarity_matrix: np.ndarray) -> None:
        self.similarity_matrix = similarity_matrix

    def save_plot(self, fig: go.Figure, filename: str) -> None:
        fig.write_html(filename)
        print(f"Plot saved to {filename}")

    def show_plot(self, fig: go.Figure) -> None:
        fig.show()


def create_3d_visualization_from_notebook_data(
    embeddings: np.ndarray,
    persona_ids: List[int],
    similarity_matrix: Optional[np.ndarray] = None,
    method: str = "tsne",
) -> Embedding3DVisualizer:
    visualizer = Embedding3DVisualizer(embeddings, persona_ids, method=method)
    if similarity_matrix is not None:
        visualizer.set_similarity_matrix(similarity_matrix)
    return visualizer


