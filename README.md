# ConsistencyAI

A novel benchmark for evaluating Large Language Models' tendency to provide inconsistent facts to different demographics.

By: Peter Banyas, Shristi Sharma, Alistair Simmons, Atharva Vispute *(the Duke Phishermen)*

Get started fast with main.ipynb!

Or see our more user-friendly app at https://v0-llm-comparison-webapp.vercel.app/

## Overview

ConsistencyAI is a research tool that evaluates whether LLMs provide consistent factual information across different demographic groups. We define "duplicity" as saying different things to different people when asked the same question.

### Motivation

As LLMs play an increasing role in information delivery, users often rely on them without conducting independent research. This makes access to truth subject to the spin, focus, and authenticity of the LLM. In an increasingly polarized society with echo chambers, we need to ensure that AI systems don't turbocharge our discordance by providing inconsistent facts to different groups.

### Methodology

1. Generate diverse personas from the NVIDIA Nemotron Personas database
2. Create personalized queries that include persona descriptions followed by identical factual questions
3. Query multiple LLMs with these personalized prompts
4. Use SentenceBERT to generate semantic embeddings of the responses
5. Compute cosine similarity between responses to measure consistency
6. Analyze patterns of inconsistency across demographic groups

Perfect consistency would yield a similarity of 1.0, while completely opposing facts would yield -1.0.

## Features

- Async LLM querying with retry logic and rate limiting
- Support for multiple LLM providers (OpenRouter, OpenAI, Google Gemini)
- Fast robust query processor with 100% success guarantee
- Incremental progress saving for long-running experiments
- Semantic similarity analysis using SentenceBERT
- Advanced clustering and embedding analysis
- Rich visualizations (heatmaps, 3D plots, leaderboards)
- Comprehensive caching and resumption capabilities

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/banyasp/consistencyAI.git
cd consistencyAI
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Configuration

Before using ConsistencyAI, you need to configure your API keys.

### Option 1: Environment Variables (Recommended)

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_API_KEY="your-openai-key"  # Optional
export GOOGLE_API_KEY="your-google-key"  # Optional
```

### Option 2: Configuration File

Edit `duplicity/config.py` and add your API keys:

```python
OPENROUTER_API_KEY = "your-openrouter-key"
OPENAI_API_KEY = "your-openai-key"  # Optional
GOOGLE_API_KEY = "your-google-key"  # Optional
```

### Option 3: Programmatic Configuration

```python
from duplicity import config

config.set_openrouter_key("your-openrouter-key")
config.set_openai_key("your-openai-key")  # Optional
config.set_google_key("your-google-key")  # Optional
```

## Quick Start

Run main.ipynb !!

### Basic Usage

```python
from duplicity import (
    get_and_clean_personas,
    generate_queries_for_personas,
    query_llm_fast,
    supercompute_similarities,
    plot_overall_leaderboard,
    collect_avg_scores_by_model
)

# 1. Get personas
personas = get_and_clean_personas(offset=0, length=10, cache=True)

# 2. Define topics to query about
topics = ["Climate Change", "Vaccines", "Economic Policy"]

# 3. Generate queries
queries = generate_queries_for_personas(personas, topics)

# 4. Define models to test
# WARNING: GPT-5 via OpenRouter often fails. 
models = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro-1.5"
]

# 5. Query the models
results = query_llm_fast(
    nested_queries=queries,
    list_of_models=models,
    initial_batch_size=50,
    initial_concurrency=20,
    adaptive_mode=True,
    ensure_100_percent_success=True
)

# 6. Compute similarity
matrices, dfs, personas, embeddings = supercompute_similarities(results)

# 7. Analyze and visualize
avg_scores = collect_avg_scores_by_model(matrices)
plot_overall_leaderboard(avg_scores)
```

### Advanced Usage with Clustering

```python
from duplicity import (
    analyze_and_cluster_embeddings,
    print_analysis_summary
)

# Perform comprehensive embedding analysis
analysis = analyze_and_cluster_embeddings(
    all_embeddings=embeddings,
    all_similarity_matrices=matrices,
    all_sorted_personas=personas,
    max_clusters=10,
    save_plots=True
)

# Print summary
print_analysis_summary(analysis)
```

### Fast Robust Query Processing

For large-scale experiments with thousands of queries:

```python
from duplicity import query_llm_fast

results = query_llm_fast(
    nested_queries=queries,
    list_of_models=models,
    initial_batch_size=50,      # Start with 50 queries per batch
    initial_concurrency=20,     # 20 concurrent requests
    max_concurrency=100,        # Allow scaling up to 100
    adaptive_mode=True,         # Auto-optimize based on performance
    ensure_100_percent_success=True,  # Retry until all succeed
    save_incremental=True,      # Save progress after each batch
    incremental_interval=1,     # Save every batch
    max_retries=5              # Retry failed queries up to 5 times
)
```

### Resume from Interrupted Run

```python
from duplicity import query_llm_fast_resume

# Resume from the last incremental checkpoint
results = query_llm_fast_resume(
    incremental_file_path="logs/incremental/incremental_results_20251009_000435_batch015_retry_round_2_010.json",
    original_queries=queries,
    list_of_models=models
)
```

## Core Modules

### llm_tool.py
Async LLM client with retry logic, rate limiting, and response validation.

### personas.py
Fetch and clean personas from the NVIDIA Nemotron dataset, generate personalized queries.

### queries.py
High-level query orchestration with sync and async interfaces.

### fast_robust_queries.py
Advanced query processor with adaptive concurrency, 100% success guarantee, and incremental saving.

### similarity.py
Compute semantic embeddings and cosine similarity matrices using SentenceBERT.

### visualization.py
Create heatmaps, 3D plots, and leaderboards to visualize consistency patterns.

### embedding_analysis.py
Advanced clustering, negative similarity analysis, and statistical evaluation.

## Caching and Logs

ConsistencyAI automatically caches intermediate results in the `logs/` directory:

- `personas_*.json` - Cached persona data
- `results_*.json` - Query results
- `similarities_*.pkl` - Computed similarity matrices and embeddings
- `incremental/` - Incremental progress checkpoints
- `clustering_plots/` - Generated visualizations

## API Keys and Providers

### OpenRouter (Recommended)
OpenRouter provides unified access to multiple LLM providers. Get your key at [openrouter.ai](https://openrouter.ai).

### Direct Provider APIs
For better performance with specific models, you can use direct APIs:
- OpenAI: [platform.openai.com](https://platform.openai.com)
- Google Gemini: [ai.google.dev](https://ai.google.dev)

## Performance

The fast robust query processor can achieve:
- 20-100 queries per second (depending on API limits)
- 99%+ success rate with automatic retries
- Adaptive concurrency scaling
- Real-time progress tracking and ETA

## Citation

If you use ConsistencyAI in your research, please cite:

```
@software{ConsistencyAI,
  title={ConsistencyAI:  Benchmark to Assess LLMs’ Factual Consistency When
Responding to Different Demographic Groups},
  author={Banyas, Peter and Sharma, Shristi and Simmons, Alistair and Vispute, Atharva},
  year={2025},
  url={https://github.com/banyasp/consistencyAI}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the authors (peter dot banyas at duke dot edu)

## Acknowledgments

- Dr. Brinnae Bent & Dr. Chris Bail, for their generous feedback, guidance, and API credit funding.
- Duke University Society-Centered AI (SCAI) conference, for hosting the hackathon where this project was born.
- NVIDIA, for the Nemotron Personas dataset.
- All LLM providers for their APIs.

