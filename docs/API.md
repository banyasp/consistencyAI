# API Reference

Complete API documentation for ConsistencyAI.

## Core Functions

### Personas

#### `get_and_clean_personas(offset=0, length=100, cache=True, tag=None)`

Fetch personas from the NVIDIA Nemotron dataset.

**Parameters:**
- `offset` (int): Starting index in the dataset
- `length` (int): Number of personas to fetch
- `cache` (bool): Whether to cache results to logs/
- `tag` (str, optional): Tag for the cache file

**Returns:**
- `dict`: Cleaned persona data with structure `{"rows": [...]}`

**Example:**
```python
personas = get_and_clean_personas(offset=0, length=50, cache=True, tag="experiment1")
```

#### `generate_queries_for_personas(personas_dict, topics_list)`

Generate personalized queries for each persona and topic.

**Parameters:**
- `personas_dict` (dict): Persona data from `get_and_clean_personas()`
- `topics_list` (list): List of topic strings

**Returns:**
- `dict`: Nested dict `{topic: {persona_id: query}}`

**Example:**
```python
topics = ["Climate Change", "Vaccines"]
queries = generate_queries_for_personas(personas, topics)
```

### Querying

#### `query_llm_fast(nested_queries, list_of_models, **kwargs)`

Fast robust query processor with adaptive optimization.

**Parameters:**
- `nested_queries` (dict): Queries from `generate_queries_for_personas()`
- `list_of_models` (list): Model specifications (e.g., "openai/gpt-4o")
- `initial_batch_size` (int, default=50): Starting batch size
- `initial_concurrency` (int, default=20): Starting concurrency
- `max_concurrency` (int, default=100): Maximum concurrency
- `adaptive_mode` (bool, default=True): Enable adaptive optimization
- `all_open_router` (bool, default=False): Route all through OpenRouter
- `ensure_100_percent_success` (bool, default=True): Retry until all succeed
- `save_incremental` (bool, default=True): Save progress incrementally
- `incremental_interval` (int, default=1): Save every N batches
- `max_retries` (int, default=5): Maximum retry attempts

**Returns:**
- `dict`: Results structure `{model: {topic: {persona_id: response}}}`

**Example:**
```python
results = query_llm_fast(
    nested_queries=queries,
    list_of_models=["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
    initial_batch_size=50,
    max_concurrency=100,
    ensure_100_percent_success=True
)
```

#### `query_llm_fast_resume(incremental_file_path, original_queries, list_of_models, **kwargs)`

Resume from an interrupted run.

**Parameters:**
- `incremental_file_path` (str): Path to incremental checkpoint file
- `original_queries` (dict): The original queries used in the run
- `list_of_models` (list): Models to process
- Additional kwargs same as `query_llm_fast()`

**Returns:**
- `dict`: Merged results with completed queries

**Example:**
```python
results = query_llm_fast_resume(
    incremental_file_path="logs/incremental/incremental_results_20241201_143022.json",
    original_queries=queries,
    list_of_models=models
)
```

### Similarity Analysis

#### `supercompute_similarities(results)`

Compute embeddings and similarity matrices for all model/topic combinations.

**Parameters:**
- `results` (dict): Results from query functions

**Returns:**
- Tuple of:
  - `all_similarity_matrices` (dict): Similarity matrices
  - `all_similarity_dfs` (dict): DataFrames of similarities
  - `all_sorted_personas` (dict): Persona ID mappings
  - `all_embeddings` (dict): Computed embeddings

**Example:**
```python
matrices, dfs, personas, embeddings = supercompute_similarities(results)
```

#### `compute_avg_similarity(similarity_matrix)`

Compute average non-diagonal similarity.

**Parameters:**
- `similarity_matrix` (ndarray): Square similarity matrix

**Returns:**
- `float`: Average similarity score (0-1)

#### `collect_avg_scores_by_model(all_similarity_matrices)`

Collect average scores for all models and topics.

**Parameters:**
- `all_similarity_matrices` (dict): From `supercompute_similarities()`

**Returns:**
- `dict`: Structure `{model: {topic: avg_score}}`

### Visualization

#### `plot_similarity_matrix_with_values(similarity_matrix, persona_ids, **kwargs)`

Create a heatmap of the similarity matrix.

**Parameters:**
- `similarity_matrix` (ndarray): Matrix to plot
- `persona_ids` (list): IDs for axis labels
- `show_values` (bool, default=True): Display values in cells
- `model_name` (str, optional): Model name for title
- `topic` (str, optional): Topic name for title
- `save_path` (str, optional): Path to save figure

**Returns:**
- Tuple of (figure, axes)

#### `plot_overall_leaderboard(avg_scores, save_path=None)`

Plot overall model consistency leaderboard.

**Parameters:**
- `avg_scores` (dict): From `collect_avg_scores_by_model()`
- `save_path` (str, optional): Directory to save plot

#### `plot_similarity_by_sphere(avg_scores, save_path=None)`

Plot similarity scores grouped by topic.

**Parameters:**
- `avg_scores` (dict): From `collect_avg_scores_by_model()`
- `save_path` (str, optional): Directory to save plots

### Embedding Analysis

#### `analyze_and_cluster_embeddings(all_embeddings, all_similarity_matrices, all_sorted_personas, **kwargs)`

Comprehensive embedding analysis with clustering.

**Parameters:**
- `all_embeddings` (dict): From `supercompute_similarities()`
- `all_similarity_matrices` (dict): From `supercompute_similarities()`
- `all_sorted_personas` (dict): From `supercompute_similarities()`
- `max_clusters` (int, default=10): Maximum clusters to test
- `random_state` (int, default=42): Random seed
- `save_plots` (bool, default=True): Save visualization plots
- `plots_dir` (str, optional): Directory for plots

**Returns:**
- `dict`: Analysis results including clustering and negative similarities

**Example:**
```python
analysis = analyze_and_cluster_embeddings(
    all_embeddings=embeddings,
    all_similarity_matrices=matrices,
    all_sorted_personas=personas,
    max_clusters=10,
    save_plots=True
)
```

#### `print_analysis_summary(analysis_results)`

Print a formatted summary of analysis results.

**Parameters:**
- `analysis_results` (dict): From `analyze_and_cluster_embeddings()`

## Classes

### `LLMComparisonTool`

Async context manager for LLM queries.

**Usage:**
```python
async with LLMComparisonTool() as tool:
    responses = await tool.run_comparison(prompt, models)
```

### `Embedding3DVisualizer`

3D visualization of embedding spaces.

**Methods:**
- `reduce_to_3d(method='tsne')`: Reduce embeddings to 3D
- `create_interactive_3d_plot()`: Create interactive Plotly visualization
- `save_plot(fig, filename)`: Save plot to file

**Example:**
```python
from duplicity import Embedding3DVisualizer

visualizer = Embedding3DVisualizer(embeddings, persona_ids)
visualizer.reduce_to_3d(method='tsne')
fig = visualizer.create_interactive_3d_plot(
    title="Embedding Space",
    show_connections=True
)
visualizer.save_plot(fig, "embedding_3d.html")
```

## Configuration

### `config.set_openrouter_key(api_key)`
Set OpenRouter API key programmatically.

### `config.set_openai_key(api_key)`
Set OpenAI API key programmatically.

### `config.validate_config()`
Validate that required API keys are set.

## Data Structures

### Query Structure
```python
{
    "Topic Name": {
        "persona_id": "full query text with persona description",
        ...
    },
    ...
}
```

### Results Structure
```python
{
    "model_spec": {
        "Topic Name": {
            "persona_id": "model response text",
            ...
        },
        ...
    },
    ...
}
```

### Similarity Structure
```python
{
    "model_spec": {
        "Topic Name": numpy_array([[sim_matrix]]),
        ...
    },
    ...
}
```

## Model Specifications

Models are specified as `"provider/model-name"`:

- OpenAI: `"openai/gpt-4o"`, `"openai/gpt-4o-mini"`
- Anthropic: `"anthropic/claude-3.5-sonnet"`, `"anthropic/claude-3.5-haiku"`
- Google: `"google/gemini-pro-1.5"`, `"google/gemini-flash-1.5"`
- Meta: `"meta-llama/llama-3.1-70b-instruct"`
- And many more via OpenRouter

See [OpenRouter models](https://openrouter.ai/models) for complete list.

## Error Handling

All query functions include robust error handling and return error messages in the response text if queries fail. Use the `ensure_100_percent_success` parameter to automatically retry failed queries.

## Performance Tips

1. **Batch Size**: Start with 50, adjust based on API performance
2. **Concurrency**: Start with 20, allow scaling to 100
3. **Adaptive Mode**: Enable for automatic optimization
4. **Incremental Saving**: Always enable for long runs
5. **Direct APIs**: Use `all_open_router=False` for better performance

## Caching

Results are automatically cached in `logs/`:
- Personas: `personas_*.json`
- Results: `results_*.json`
- Similarities: `similarities_*.pkl`
- Incremental: `incremental/*.json`

Use the `tag` parameter to organize different experiments.

