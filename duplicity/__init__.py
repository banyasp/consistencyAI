"""
ConsistencyAI - A benchmark for evaluating LLM consistency across demographics.

This package provides tools to:
- Query multiple LLMs with personalized prompts
- Generate personas from the NVIDIA Nemotron dataset
- Compute semantic similarity of LLM responses using embeddings
- Visualize and analyze response consistency patterns
- Perform clustering and advanced embedding analysis

Core modules:
- llm_tool: Async LLM client with retry logic and rate limiting
- personas: Persona fetching and query generation
- queries: High-level query orchestration (sync and async)
- fast_robust_queries: Advanced fast query processor with 100% success guarantee
- similarity: Embedding and similarity computations
- visualization: Plotting and 3D visualization tools
- embedding_analysis: Statistical analysis and clustering of embeddings
"""

__version__ = "2.0.0"
__author__ = "Peter Banyas, Shristi Sharma, Alistair Simmons, Atharva Vispute"

# Import config module for API key management
from . import config

from .llm_tool import (
    LLMResponse,
    LLMComparisonTool,
    get_provider_from_model,
    get_model_name_from_spec,
)
from .personas import (
    get_and_clean_personas,
    generate_queries_for_personas,
    save_personas_log,
    list_cached_personas,
    load_personas_from_log,
    load_latest_personas,
)
from .queries import (
    query_llm,
    query_llm_sync,
    query_llm_multithreaded,
    query_llm_multithreaded_sync,
    save_results_log,
    list_cached_results,
    load_results_from_log,
    load_latest_results,
)
from .fast_robust_queries import (
    query_llm_fast,
    query_llm_fast_resume,
    load_incremental_results,
    load_latest_fast_results,
    find_missing_queries,
    find_missing_model_persona_combinations,
)
from .similarity import (
    compute_persona_similarity,
    compute_all_similarity_matrices,
    supercompute_similarities,
    compute_avg_similarity,
    collect_avg_scores_by_model,
    save_similarity_results,
    list_cached_similarity_results,
    load_similarity_results,
    load_latest_similarity_results,
)
from .visualization import (
    plot_similarity_matrix_with_values,
    find_extreme_similarities_with_values,
    print_extreme_similarities,
    plot_similarity_by_sphere,
    plot_overall_leaderboard,
    Embedding3DVisualizer,
    create_3d_visualization_from_notebook_data,
)
from .embedding_analysis import (
    analyze_negative_similarities,
    find_optimal_clusters,
    cluster_embeddings,
    robust_cluster_embeddings,
    create_cluster_visualization,
    analyze_and_cluster_embeddings,
    print_analysis_summary,
)
from .central_analysis import (
    compute_central_analysis,
    print_central_analysis_summary,
)
from .control_experiment import (
    load_mary_alberti_persona,
    generate_control_queries,
    run_control_experiment,
)
from .variance_analysis import (
    compute_within_model_variance,
    compute_across_persona_variance,
    create_variance_comparison_visualizations,
    generate_variance_report,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Configuration
    "config",

    # LLM Tool
    "LLMResponse",
    "LLMComparisonTool",
    "get_provider_from_model",
    "get_model_name_from_spec",
    
    # Personas
    "get_and_clean_personas",
    "generate_queries_for_personas",
    "save_personas_log",
    "list_cached_personas",
    "load_personas_from_log",
    "load_latest_personas",
    
    # Queries
    "query_llm",
    "query_llm_sync",
    "query_llm_multithreaded",
    "query_llm_multithreaded_sync",
    "save_results_log",
    "list_cached_results",
    "load_results_from_log",
    "load_latest_results",
    
    # Fast Robust Queries
    "query_llm_fast",
    "query_llm_fast_resume",
    "load_incremental_results",
    "load_latest_fast_results",
    "find_missing_queries",
    "find_missing_model_persona_combinations",
    
    # Similarity
    "compute_persona_similarity",
    "compute_all_similarity_matrices",
    "supercompute_similarities",
    "compute_avg_similarity",
    "collect_avg_scores_by_model",
    "save_similarity_results",
    "list_cached_similarity_results",
    "load_similarity_results",
    "load_latest_similarity_results",
    
    # Visualization
    "plot_similarity_matrix_with_values",
    "find_extreme_similarities_with_values",
    "print_extreme_similarities",
    "plot_similarity_by_sphere",
    "plot_overall_leaderboard",
    "Embedding3DVisualizer",
    "create_3d_visualization_from_notebook_data",
    
    # Embedding Analysis
    "analyze_negative_similarities",
    "find_optimal_clusters",
    "cluster_embeddings",
    "robust_cluster_embeddings",
    "create_cluster_visualization",
    "analyze_and_cluster_embeddings",
    "print_analysis_summary",
    
    # Central Analysis
    "compute_central_analysis",
    "print_central_analysis_summary",

    # Control Experiment
    "load_mary_alberti_persona",
    "generate_control_queries",
    "run_control_experiment",

    # Variance Analysis
    "compute_within_model_variance",
    "compute_across_persona_variance",
    "create_variance_comparison_visualizations",
    "generate_variance_report",
]

