# Quick Start Guide

Get started with ConsistencyAI in 5 minutes. Just run main.ipynb!

## Installation

### 1. Set up a virtual environment (optional but recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Setup API Key

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Or in Python:

```python
from duplicity import config
config.set_openrouter_key("your-key-here")
```

## Your First Experiment
Run main.ipynb.

Alternatively:

```python
from duplicity import (
    get_and_clean_personas,
    generate_queries_for_personas,
    query_llm_fast,
    supercompute_similarities,
    collect_avg_scores_by_model
)

# 1. Get 10 test personas
personas = get_and_clean_personas(offset=0, length=10)

# 2. Define topics
topics = ["Climate Change", "Vaccines"]

# 3. Generate queries
queries = generate_queries_for_personas(personas, topics)

# 4. Define models
models = ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]

# 5. Query models
results = query_llm_fast(
    nested_queries=queries,
    list_of_models=models,
    initial_batch_size=10,
    initial_concurrency=5
)

# 6. Compute similarity
matrices, _, _, _ = supercompute_similarities(results)

# 7. Get scores
scores = collect_avg_scores_by_model(matrices)
print(scores)
```

## Understanding Results

- **Score of 1.0**: Perfect consistency (same facts to everyone)
- **Score of 0.5**: Moderate consistency (substantial variation)
- **Score of 0.0**: No correlation (unrelated responses)
- **Negative scores**: Opposing facts (concerning!)

## Next Steps

1. Review the main.ipynb notebook for a complete example
2. Read the API.md documentation for all available functions
3. Build your own experiments with different personas, topics, and models

## Common Issues

### Missing API Key
```
ValueError: OPENROUTER_API_KEY is not set
```
**Solution**: Set your API key as shown above

### Import Error
```
ModuleNotFoundError: No module named 'duplicity'
```
**Solution**: Install from the package directory:
```bash
pip install -e .
```

### Out of Memory
Large experiments may require more RAM. Reduce batch size:
```python
results = query_llm_fast(
    ...,
    initial_batch_size=10,  # Smaller batches
    max_concurrency=5       # Less concurrency
)
```

## Getting Help

- Check the full README.md
- Review the API.md documentation
- Open an issue on GitHub

