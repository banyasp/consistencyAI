# ConsistencyAI Directory Structure

Complete file structure of the ConsistencyAI package.

```
consistencyAI/
│
├── README.md                      # Main project documentation
├── QUICKSTART.md                  # Quick start guide (5 minutes)
├── API.md                         # Complete API reference
├── INSTALL.md                     # Complete installation guide
├── STRUCTURE.md                   # This file - directory structure
├── LICENSE                        # MIT License
├── MANIFEST.in                    # Package installation manifest
├── setup.py                       # Python package setup script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
├── verify_installation.py         # Installation verification script
├── main.ipynb                     # Main example notebook
│
└── duplicity/                     # Main package directory
    ├── __init__.py               # Package initialization & exports
    ├── config.py                 # Configuration (env-based API keys)
    ├── llm_tool.py               # Async LLM client with retry logic
    ├── personas.py               # Persona fetching and query generation
    ├── queries.py                # Standard query orchestration
    ├── fast_robust_queries.py   # Advanced fast query processor
    ├── similarity.py             # Embedding and similarity computation
    ├── visualization.py          # Plotting and 3D visualizations
    ├── embedding_analysis.py    # Clustering and statistical analysis
    └── logs/                     # Directory for cached results (gitignored)
```

## File Descriptions

### Root Documentation Files

- **README.md** (8.4 KB)
  - Comprehensive project overview
  - Features, installation, and usage
  - Quick start examples
  - Citation information

- **QUICKSTART.md**
  - Get started in 5 minutes
  - Minimal working example
  - Common issues and solutions

- **API.md**
  - Complete API reference
  - All functions with parameters and returns
  - Class documentation
  - Usage examples for each function

- **INSTALL.md**
  - Complete installation instructions
  - Prerequisites and setup
  - Configuration options
  - Troubleshooting guide

- **STRUCTURE.md** (this file)
  - Directory structure overview
  - File descriptions
  - Import patterns

### Configuration Files

- **setup.py**
  - Python package configuration
  - Dependencies specification
  - Package metadata
  - Package classifiers

- **requirements.txt**
  - Python dependencies list
  - Version specifications
  - All required packages

- **MANIFEST.in**
  - Files to include in package
  - Patterns for documentation
  - Example files inclusion

- **.gitignore** (701 B)
  - Python artifacts
  - Virtual environments
  - API keys and configs
  - Output directories
  - IDE files

### Core Package Files

#### duplicity/__init__.py (4.2 KB)
- Package initialization
- Public API exports
- Version information
- Clean import interface

#### duplicity/config.py (1.6 KB)
- Environment-based configuration
- API key management
- Programmatic key setting
- Configuration validation

#### duplicity/llm_tool.py (14.7 KB)
- `LLMResponse` dataclass
- `LLMComparisonTool` async client
- Retry logic and rate limiting
- Multi-provider support (OpenRouter, OpenAI, Google)
- Response validation
- Session health management

#### duplicity/personas.py (5.3 KB)
- `get_and_clean_personas()` - Fetch from NVIDIA dataset
- `generate_queries_for_personas()` - Create personalized queries
- `save_personas_log()` - Cache personas
- `load_personas_from_log()` - Load cached personas
- `list_cached_personas()` - List available caches

#### duplicity/queries.py (6.6 KB)
- `query_llm()` - Async query function
- `query_llm_sync()` - Synchronous wrapper
- `query_llm_multithreaded()` - Parallel processing
- `save_results_log()` - Save query results
- `load_results_from_log()` - Load cached results

#### duplicity/fast_robust_queries.py (47.6 KB)
- `FastRobustQueryProcessor` class
- `query_llm_fast()` - Fast processor with optimization
- `query_llm_fast_resume()` - Resume from checkpoints
- Adaptive concurrency and batch sizing
- 100% success guarantee with retries
- Incremental progress saving
- Real-time performance tracking
- Model-specific rate limiting

#### duplicity/similarity.py (8.2 KB)
- `compute_persona_similarity()` - Single model/topic
- `supercompute_similarities()` - All combinations
- `compute_avg_similarity()` - Average score
- `collect_avg_scores_by_model()` - Aggregate scores
- `save_similarity_results()` - Cache similarities
- `load_similarity_results()` - Load cached similarities
- SentenceBERT integration
- Progress bars with tqdm

#### duplicity/visualization.py
- `plot_similarity_matrix_with_values()` - Heatmaps
- `plot_overall_leaderboard()` - Model comparison
- `plot_similarity_by_sphere()` - Topic-specific plots
- `Embedding3DVisualizer` class - 3D visualizations
- `create_interactive_3d_plot()` - Plotly interactive plots
- Support for PCA, t-SNE, MDS dimensionality reduction

#### duplicity/embedding_analysis.py
- `analyze_negative_similarities()` - Detect opposing facts
- `find_optimal_clusters()` - Optimal cluster number
- `cluster_embeddings()` - GMM clustering
- `robust_cluster_embeddings()` - Fallback methods
- `create_cluster_visualization()` - 2D/3D plots
- `analyze_and_cluster_embeddings()` - Comprehensive analysis
- `print_analysis_summary()` - Formatted results
- Silhouette, Calinski-Harabasz, Davies-Bouldin metrics

### Notebook

#### main.ipynb
- Complete end-to-end workflow example
- Demonstrates all major features
- Interactive experimentation
- Visualization generation

### Utility Files

#### verify_installation.py
- Test imports
- Test core functions
- Test dependencies
- Test configuration
- Version information
- Diagnostic output


## Key Features by File

### Performance Optimization
- `fast_robust_queries.py` - Adaptive concurrency, batching
- `similarity.py` - Batch processing, caching
- `llm_tool.py` - Rate limiting, session management

### Error Handling
- `fast_robust_queries.py` - 100% success guarantee
- `llm_tool.py` - Retry logic, response validation
- `embedding_analysis.py` - Robust clustering fallbacks

### User Experience
- `main.ipynb` - Interactive example notebook
- `verify_installation.py` - Installation validation
- Progressive documentation (Quick → API → Full)

### Research Features
- `embedding_analysis.py` - Advanced clustering
- `visualization.py` - Multiple plot types
- `similarity.py` - SentenceBERT embeddings

## Import Structure

```python
# Top-level imports (recommended)
from duplicity import (
    get_and_clean_personas,        # personas.py
    query_llm_fast,                 # fast_robust_queries.py
    supercompute_similarities,      # similarity.py
    plot_overall_leaderboard,       # visualization.py
    analyze_and_cluster_embeddings, # embedding_analysis.py
)

# Module-level imports (if needed)
from duplicity.llm_tool import LLMComparisonTool
from duplicity.config import set_openrouter_key
from duplicity.visualization import Embedding3DVisualizer
```

## Cache Structure (Generated at Runtime)

```
consistencyAI/
└── duplicity/
    └── logs/
        ├── personas_YYYYMMDD_HHMMSS_tag.json
        ├── results_YYYYMMDD_HHMMSS_tag.json
        ├── similarities_YYYYMMDD_HHMMSS_tag.pkl
        ├── incremental/
        │   └── incremental_results_*.json
        └── clustering_plots/
            └── clustering_*.png
```

