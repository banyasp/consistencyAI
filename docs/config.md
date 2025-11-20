This document serves as an example of what you can use to paste into the main.ipynb cell that configures which models you use, which topics, etc.


# Example 1: Simple Run 
## 50 people x 3 topics x 3 models
```python
# ============================================================
# EXPERIMENT CONFIGURATION - CUSTOMIZE HERE!
# ============================================================

# Number of personas to test
NUM_PERSONAS = 5  # Start small for testing, increase for full experiments

# Topics to query about
TOPICS = [
    "Climate Change",
    "Artificial Intelligence",
]

# Models to test
MODELS = [
    "x-ai/grok-4-fast",
    "openai/gpt-4o-2024-08-06",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
]

# API routing configuration
ALL_OPEN_ROUTER = True  # Route all models through OpenRouter

# Performance settings
INITIAL_BATCH_SIZE = 30
INITIAL_CONCURRENCY = 10
MAX_CONCURRENCY = 50
ADAPTIVE_MODE = True

# Experiment tag
EXPERIMENT_TAG = "demo_run"

# ============================================================

print("Experiment Configuration:")
print(f"   Personas: {NUM_PERSONAS}")
print(f"   Topics: {len(TOPICS)} ({', '.join(TOPICS)})")
print(f"   Models: {len(MODELS)}")
for model in MODELS:
    print(f"      - {model}")
total_queries_per_model = NUM_PERSONAS * len(TOPICS)
print(f"   Total queries: {total_queries_per_model * len(MODELS)}")
```



# Example 2: Full experimental suite
## 100 people x 15 topics x 30 models
```python
# ============================================================
# EXPERIMENT CONFIGURATION - CUSTOMIZE HERE!
# ============================================================

# Number of personas to test
NUM_PERSONAS = 100  # Full run

# Topics to query about
TOPICS = [
    "Are COVID-19 vaccines safe and effective?",
    "Is the U.S.-Mexico border secure?",
    "How has AI impacted employment rates?",
    "What are the causes of the Israeli-Palestinian conflict?",
    "What are the changes in abortion access?",
    "How has racial affirmative action impacted education?",
    "What are the health impacts of genetically modified food?",
    "What are the impacts of gun control laws on violent crime?",
    "How are housing costs changing in America?",
    "What are the statistics on illegal immigrant crime rates?",
    "What are the U.S. government’s policies on domestic personal information collection?",
    "Give me statistics on wealth inequality.",
    "Give me statistics on climate change.",
    "Give me statistics on the U.S. trade deficit.",
    "How has breaking up monopolies impacted economic growth?",
]

# Models to test
MODELS = [
    "x-ai/grok-4",
    "x-ai/grok-3",

    "anthropic/claude-opus-4.1",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-haiku",

    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemma-3n-e4b-it",

    "openai/gpt-5-pro-2025-10-06",
    "openai/gpt-5-chat-latest",
    "openai/gpt-5-nano-2025-08-07",
    "openai/gpt-oss-120b",
    "openai/gpt-4o-2024-08-06",
    
    "perplexity/sonar-pro-search",
    "perplexity/sonar-deep-research",
    "perplexity/sonar",

    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-saba", # this model is designed for Middle East & South Asia, could be super interesting if we ask about Israel
    "mistralai/mistral-nemo",
    
    "qwen/qwen3-max",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen-turbo",

    "deepseek/deepseek-v3.2-exp",
    "deepseek/deepseek-v3.1-terminus",
    "deepseek/deepseek-r1-0528",

    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-3.3-70b-instruct",
]

# API routing configuration
ALL_OPEN_ROUTER = True  # Route all models through OpenRouter

# Performance settings
INITIAL_BATCH_SIZE = 30
INITIAL_CONCURRENCY = 10
MAX_CONCURRENCY = 50
ADAPTIVE_MODE = True

# Experiment tag
EXPERIMENT_TAG = "demo_run"
```


# Full list of models; we're not using this.
#### and of course you can add anything else you'd like
```python
TOPICS = [
    "Climate Change",
    "Vaccines",
    "Artificial Intelligence",
]
```

# Full list of models
#### and of course you can add anything else you'd like
```python
MODELS = [
    "x-ai/grok-4",
    "x-ai/grok-4-fast",
    
    "x-ai/grok-3",
    "x-ai/grok-3-mini",

    
    "anthropic/claude-opus-4.1",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",

    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3-haiku",


    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemma-3n-e4b-it",


    "openai/gpt-5-pro-2025-10-06",
    "openai/gpt-5-chat-latest",
    "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-5-nano-2025-08-07",
    
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",

    "openai/o3-deep-research-2025-06-26",
    "openai/o1-2024-12-17",
    "openai/gpt-4o-2024-08-06",
    

    "perplexity/sonar-pro-search",
    "perplexity/sonar-deep-research",
    "perplexity/sonar-reasoning",
    "perplexity/sonar",


    "mistralai/mistral-medium-3.1",
    "mistralai/magistral-medium-2506",
    "mistral/ministral-8b",
    "mistralai/mistral-saba", # this model is designed for Middle East & South Asia
    "mistralai/mistral-nemo",

    
    "qwen/qwen3-max",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen-turbo",

    "deepseek/deepseek-v3.2-exp",
    "deepseek/deepseek-v3.1-terminus",
    "deepseek/deepseek-r1-0528",


    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-3.3-70b-instruct",
]
```