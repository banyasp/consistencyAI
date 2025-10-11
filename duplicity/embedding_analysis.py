"""Embedding analysis and clustering utilities for similarity outputs."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path


def analyze_negative_similarities(
    all_similarity_matrices: Dict[str, Dict[str, np.ndarray]],
    all_sorted_personas: Dict[str, Dict[str, List[int]]]
) -> Dict[str, Dict[str, Dict]]:
    """
    Analyze cases with negative cosine similarity.
    
    Returns a nested dictionary with statistics about negative similarities
    for each model-topic pair.
    """
    negative_analysis = {}
    
    for model_name, topics in all_similarity_matrices.items():
        negative_analysis[model_name] = {}
        
        for topic_name, similarity_matrix in topics.items():
            persona_ids = all_sorted_personas[model_name][topic_name]
            
            # Find negative similarities (excluding diagonal)
            negative_mask = (similarity_matrix < 0) & ~np.eye(len(similarity_matrix), dtype=bool)
            negative_indices = np.where(negative_mask)
            
            if len(negative_indices[0]) == 0:
                # No negative similarities found
                negative_analysis[model_name][topic_name] = {
                    'has_negative': False,
                    'count': 0,
                    'min_value': None,
                    'max_value': None,
                    'mean_value': None,
                    'pairs': []
                }
                continue
            
            # Get the negative similarity values and their indices
            negative_values = similarity_matrix[negative_indices]
            row_indices = negative_indices[0]
            col_indices = negative_indices[1]
            
            # Create pairs of persona IDs with their negative similarities
            negative_pairs = []
            for i, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
                if row_idx < col_idx:  # Avoid duplicate pairs (matrix is symmetric)
                    negative_pairs.append({
                        'persona1': persona_ids[row_idx],
                        'persona2': persona_ids[col_idx],
                        'similarity': float(negative_values[i])
                    })
            
            negative_analysis[model_name][topic_name] = {
                'has_negative': True,
                'count': len(negative_pairs),
                'min_value': float(np.min(negative_values)),
                'max_value': float(np.max(negative_values)),
                'mean_value': float(np.mean(negative_values)),
                'pairs': negative_pairs
            }
    
    return negative_analysis


def find_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 10,
    random_state: int = 42
) -> Tuple[int, Dict[str, float]]:
    """
    Find the optimal number of clusters using multiple metrics.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features)
        max_clusters: Maximum number of clusters to test
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (optimal_n_clusters, metrics_dict)
    """
    if len(embeddings) < 2:
        return 1, {}
    
    # Limit max_clusters to number of samples
    max_clusters = min(max_clusters, len(embeddings) - 1)
    
    if max_clusters < 2:
        return 1, {}
    
    # Convert to float64 for better numerical stability
    embeddings = embeddings.astype(np.float64)
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    metrics = {}
    
    for n_clusters in range(2, max_clusters + 1):
        try:
            # Fit GMM with more robust parameters
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                n_init=10,
                reg_covar=1e-6,  # Increase regularization
                max_iter=1000,    # More iterations
                tol=1e-4          # More lenient tolerance
            )
            gmm.fit(embeddings_scaled)
            
            # Get cluster labels
            labels = gmm.predict(embeddings_scaled)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                try:
                    silhouette = silhouette_score(embeddings_scaled, labels)
                    calinski = calinski_harabasz_score(embeddings_scaled, labels)
                    davies = davies_bouldin_score(embeddings_scaled, labels)
                    
                    metrics[n_clusters] = {
                        'silhouette': silhouette,
                        'calinski_harabasz': calinski,
                        'davies_bouldin': davies,
                        'bic': gmm.bic(embeddings_scaled),
                        'aic': gmm.aic(embeddings_scaled)
                    }
                except Exception as metric_e:
                    # If metric calculation fails, still record the clustering
                    warnings.warn(f"Failed to calculate metrics for {n_clusters} clusters: {metric_e}")
                    metrics[n_clusters] = {
                        'silhouette': -1,
                        'calinski_harabasz': 0,
                        'davies_bouldin': float('inf'),
                        'bic': gmm.bic(embeddings_scaled) if hasattr(gmm, 'bic') else float('inf'),
                        'aic': gmm.aic(embeddings_scaled) if hasattr(gmm, 'aic') else float('inf')
                    }
            else:
                metrics[n_clusters] = {
                    'silhouette': -1,
                    'calinski_harabasz': 0,
                    'davies_bouldin': float('inf'),
                    'bic': float('inf'),
                    'aic': float('inf')
                }
                
        except Exception as e:
            warnings.warn(f"Failed to fit GMM with {n_clusters} clusters: {e}")
            metrics[n_clusters] = {
                'silhouette': -1,
                'calinski_harabasz': 0,
                'davies_bouldin': float('inf'),
                'bic': float('inf'),
                'aic': float('inf')
            }
    
    # Determine optimal number of clusters
    # Higher silhouette and calinski_harabasz scores are better
    # Lower davies_bouldin, BIC, and AIC scores are better
    if not metrics:
        return 1, {}
    
    # Normalize scores for comparison
    silhouette_norm = [(k, v['silhouette']) for k, v in metrics.items() if v['silhouette'] > -1]
    calinski_norm = [(k, v['calinski_harabasz']) for k, v in metrics.items() if v['calinski_harabasz'] > 0]
    davies_norm = [(k, 1/v['davies_bouldin']) for k, v in metrics.items() if v['davies_bouldin'] != float('inf')]
    bic_norm = [(k, 1/v['bic']) for k, v in metrics.items() if v['bic'] != float('inf')]
    aic_norm = [(k, 1/v['aic']) for k, v in metrics.items() if v['aic'] != float('inf')]
    
    # Calculate composite score
    cluster_scores = {}
    for n_clusters in metrics.keys():
        score = 0
        count = 0
        
        # Add normalized scores
        for metric_name, metric_values in [('silhouette', silhouette_norm), 
                                         ('calinski', calinski_norm),
                                         ('davies', davies_norm),
                                         ('bic', bic_norm),
                                         ('aic', aic_norm)]:
            for k, v in metric_values:
                if k == n_clusters:
                    score += v
                    count += 1
                    break
        
        if count > 0:
            cluster_scores[n_clusters] = score / count
        else:
            cluster_scores[n_clusters] = 0
    
    # Find optimal number of clusters
    if cluster_scores:
        optimal_clusters = max(cluster_scores, key=cluster_scores.get)
    else:
        optimal_clusters = 1
    
    return optimal_clusters, metrics


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Cluster embeddings using Gaussian Mixture Model.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features)
        n_clusters: Number of clusters to create
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (cluster_labels, fitted_gmm_model)
    """
    if len(embeddings) < n_clusters:
        # If we have fewer samples than clusters, assign each to its own cluster
        labels = np.arange(len(embeddings))
        return labels, None
    
    # Convert to float64 for better numerical stability
    embeddings = embeddings.astype(np.float64)
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    try:
        # Fit GMM with robust parameters
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            n_init=10,
            reg_covar=1e-6,  # Increase regularization
            max_iter=1000,    # More iterations
            tol=1e-4          # More lenient tolerance
        )
        gmm.fit(embeddings_scaled)
        
        # Get cluster labels
        labels = gmm.predict(embeddings_scaled)
        
        return labels, gmm
        
    except Exception as e:
        warnings.warn(f"GMM clustering failed with {n_clusters} clusters: {e}")
        warnings.warn("Falling back to simple clustering")
        
        # Fallback: use simple k-means or assign random clusters
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings_scaled)
            return labels, None
        except Exception as kmeans_e:
            warnings.warn(f"K-means fallback also failed: {kmeans_e}")
            # Last resort: assign random clusters
            np.random.seed(random_state)
            labels = np.random.randint(0, n_clusters, size=len(embeddings))
            return labels, None


def robust_cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Robust clustering that tries multiple approaches.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features)
        n_clusters: Number of clusters to create
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (cluster_labels, clustering_model_or_none)
    """
    if len(embeddings) < n_clusters:
        # If we have fewer samples than clusters, assign each to its own cluster
        labels = np.arange(len(embeddings))
        return labels, None
    
    # Convert to float64 for better numerical stability
    embeddings = embeddings.astype(np.float64)
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Try different clustering approaches in order of preference
    
    # 1. Try GMM with robust parameters
    try:
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            n_init=10,
            reg_covar=1e-4,  # Higher regularization
            max_iter=500,     # Reasonable iterations
            tol=1e-3          # More lenient tolerance
        )
        gmm.fit(embeddings_scaled)
        labels = gmm.predict(embeddings_scaled)
        return labels, gmm
    except Exception as e:
        warnings.warn(f"GMM failed: {e}")
    
    # 2. Try K-means
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings_scaled)
        return labels, kmeans
    except Exception as e:
        warnings.warn(f"K-means failed: {e}")
    
    # 3. Try Agglomerative clustering
    try:
        from sklearn.cluster import AgglomerativeClustering
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(embeddings_scaled)
        return labels, agg
    except Exception as e:
        warnings.warn(f"Agglomerative clustering failed: {e}")
    
    # 4. Try Spectral clustering
    try:
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(
            n_clusters=n_clusters, 
            random_state=random_state,
            affinity='nearest_neighbors',
            n_neighbors=min(10, len(embeddings_scaled) - 1)
        )
        labels = spectral.fit_predict(embeddings_scaled)
        return labels, spectral
    except Exception as e:
        warnings.warn(f"Spectral clustering failed: {e}")
    
    # 5. Last resort: random assignment
    warnings.warn("All clustering methods failed, using random assignment")
    np.random.seed(random_state)
    labels = np.random.randint(0, n_clusters, size=len(embeddings))
    return labels, None


def create_cluster_visualization(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    model_name: str,
    topic_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Create 2D and 3D visualizations of clustered embeddings.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features)
        cluster_labels: Array of cluster labels for each sample
        model_name: Name of the model for the title
        topic_name: Name of the topic for the title
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
    
    Returns:
        matplotlib Figure object
    """
    if len(embeddings) < 2:
        # Create a simple figure for insufficient data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient data for clustering', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Clustering Results: {model_name} - {topic_name}')
        return fig
    
    # Reduce dimensionality for visualization
    if embeddings.shape[1] > 3:
        # Use PCA for 3D projection
        pca_3d = PCA(n_components=3, random_state=42)
        embeddings_3d = pca_3d.fit_transform(embeddings)
        
        # Use PCA for 2D projection
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_2d.fit_transform(embeddings)
    else:
        embeddings_3d = embeddings
        embeddings_2d = embeddings[:, :2] if embeddings.shape[1] >= 2 else embeddings
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 2D subplot
    ax1 = fig.add_subplot(121)
    scatter_2d = ax1.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=cluster_labels, cmap='tab10', alpha=0.7, s=50
    )
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_title('2D Projection')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter_2d, ax=ax1)
    cbar.set_label('Cluster')
    
    # 3D subplot
    ax2 = fig.add_subplot(122, projection='3d')
    scatter_3d = ax2.scatter(
        embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
        c=cluster_labels, cmap='tab10', alpha=0.7, s=50
    )
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')
    ax2.set_title('3D Projection')
    
    # Add colorbar
    cbar = plt.colorbar(scatter_3d, ax=ax2)
    cbar.set_label('Cluster')
    
    # Main title
    fig.suptitle(f'Embedding Clustering: {model_name} - {topic_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_and_cluster_embeddings(
    all_embeddings: Dict[str, Dict[str, np.ndarray]],
    all_similarity_matrices: Dict[str, Dict[str, np.ndarray]],
    all_sorted_personas: Dict[str, Dict[str, List[int]]],
    max_clusters: int = 10,
    random_state: int = 42,
    save_plots: bool = True,
    plots_dir: Optional[str] = None
) -> Dict[str, Dict[str, Dict]]:
    """
    Comprehensive analysis of embeddings including negative similarity analysis and clustering.
    
    Args:
        all_embeddings: Dictionary of embeddings from supercompute_similarities
        all_similarity_matrices: Dictionary of similarity matrices
        all_sorted_personas: Dictionary of persona IDs
        max_clusters: Maximum number of clusters to test
        random_state: Random state for reproducibility
        save_plots: Whether to save clustering plots
        plots_dir: Directory to save plots (if None, uses logs directory)
    
    Returns:
        Dictionary containing analysis results for each model-topic pair
    """
    # Analyze negative similarities
    negative_analysis = analyze_negative_similarities(
        all_similarity_matrices, all_sorted_personas
    )
    
    # Set up plots directory
    if plots_dir is None:
        from .similarity import _logs_dir
        plots_dir = os.path.join(_logs_dir(), "clustering_plots")
    
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
    
    # Comprehensive analysis results
    analysis_results = {}
    
    for model_name, topics in all_embeddings.items():
        analysis_results[model_name] = {}
        
        for topic_name, embeddings in topics.items():
            print(f"Analyzing {model_name} - {topic_name}...")
            
            # Skip if no embeddings
            if len(embeddings) == 0:
                analysis_results[model_name][topic_name] = {
                    'negative_analysis': negative_analysis[model_name][topic_name],
                    'clustering': {
                        'optimal_clusters': 1,
                        'metrics': {},
                        'cluster_labels': [],
                        'gmm_model': None
                    },
                    'error': 'No embeddings available'
                }
                continue
            
            # Find optimal number of clusters
            optimal_clusters, metrics = find_optimal_clusters(
                embeddings, max_clusters, random_state
            )
            
            # Perform clustering using robust method
            cluster_labels, gmm_model = robust_cluster_embeddings(
                embeddings, optimal_clusters, random_state
            )
            
            # Create visualization
            if save_plots and len(embeddings) >= 2:
                plot_filename = f"clustering_{model_name}_{topic_name}.png"
                plot_path = os.path.join(plots_dir, plot_filename)
                
                try:
                    fig = create_cluster_visualization(
                        embeddings, cluster_labels, model_name, topic_name, plot_path
                    )
                    plt.close(fig)  # Close to free memory
                except Exception as e:
                    print(f"Warning: Failed to create plot for {model_name} - {topic_name}: {e}")
                    plot_path = None
            else:
                plot_path = None
            
            # Store results
            analysis_results[model_name][topic_name] = {
                'negative_analysis': negative_analysis[model_name][topic_name],
                'clustering': {
                    'optimal_clusters': optimal_clusters,
                    'metrics': metrics,
                    'cluster_labels': cluster_labels.tolist() if cluster_labels is not None else [],
                    'gmm_model': gmm_model,
                    'plot_path': plot_path
                }
            }
    
    return analysis_results


def print_analysis_summary(analysis_results: Dict[str, Dict[str, Dict]]) -> None:
    """
    Print a summary of the analysis results.
    
    Args:
        analysis_results: Results from analyze_and_cluster_embeddings
    """
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS SUMMARY")
    print("="*80)
    
    total_negative_cases = 0
    total_clusters = 0
    
    for model_name, topics in analysis_results.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)
        
        for topic_name, results in topics.items():
            print(f"  Topic: {topic_name}")
            
            # Negative similarity analysis
            neg_analysis = results['negative_analysis']
            if neg_analysis['has_negative']:
                print(f"    Negative similarities: {neg_analysis['count']} pairs")
                print(f"    Range: [{neg_analysis['min_value']:.3f}, {neg_analysis['max_value']:.3f}]")
                print(f"    Mean: {neg_analysis['mean_value']:.3f}")
                total_negative_cases += neg_analysis['count']
            else:
                print("    No negative similarities found")
            
            # Clustering analysis
            clustering = results['clustering']
            if 'error' not in results:
                optimal_clusters = clustering['optimal_clusters']
                print(f"    Optimal clusters: {optimal_clusters}")
                total_clusters += optimal_clusters
                
                if clustering['plot_path']:
                    print(f"    Plot saved: {clustering['plot_path']}")
            else:
                print(f"    Error: {results['error']}")
    
    print("\n" + "="*80)
    print(f"TOTAL NEGATIVE SIMILARITY PAIRS: {total_negative_cases}")
    print(f"TOTAL OPTIMAL CLUSTERS: {total_clusters}")
    print("="*80)
