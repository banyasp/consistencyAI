#!/usr/bin/env python3
"""
Debug script to test why resume isn't detecting completed work
"""

import json
from duplicity import load_incremental_results, generate_queries_for_personas, get_and_clean_personas
from pathlib import Path

# Configuration (from your notebook)
NUM_PERSONAS = 100
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
    "What are the U.S. government's policies on domestic personal information collection?",
    "Give me statistics on wealth inequality.",
    "Give me statistics on climate change.",
    "Give me statistics on the U.S. trade deficit.",
    "How has breaking up monopolies impacted economic growth?",
]

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
    "openai/gpt-5-chat-latest",
    "openai/gpt-4o-2024-08-06",
    "perplexity/sonar",
    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-saba",
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

print("="*80)
print("DEBUG: Testing Resume Detection")
print("="*80)

# Load the cleaned file
cleaned_file = "logs/main/incremental/incremental_results_20251110_000222_cleaned.json"
print(f"\n1. Loading cleaned file: {cleaned_file}")
existing_data = load_incremental_results(cleaned_file)
existing_results = existing_data.get("results", {})

print(f"   Models in existing results: {len(existing_results)}")
print(f"   Total responses: {sum(len(personas) for model_data in existing_results.values() for personas in model_data.values())}")

# Load personas and generate queries (simulating what notebook does)
print(f"\n2. Loading personas...")
main_personas = get_and_clean_personas(offset=0, length=NUM_PERSONAS, cache=True, tag="main", subdir="main")
print(f"   Loaded {len(main_personas.get('rows', []))} personas")

print(f"\n3. Generating queries...")
main_queries = generate_queries_for_personas(main_personas, TOPICS)
total_queries_per_model = sum(len(topic_queries) for topic_queries in main_queries.values())
print(f"   Generated {total_queries_per_model} queries per model")

# Test the find_missing_model_persona_combinations function
print(f"\n4. Testing find_missing_model_persona_combinations...")
print(f"   Target models: {len(MODELS)}")

# Manually check what should be detected as complete
print(f"\n5. Checking what's actually in existing results:")
sample_model = "x-ai/grok-4"
sample_topic = "Are COVID-19 vaccines safe and effective?"
if sample_model in existing_results and sample_topic in existing_results[sample_model]:
    completed_personas = list(existing_results[sample_model][sample_topic].keys())
    print(f"   {sample_model} / {sample_topic}:")
    print(f"   Completed personas: {sorted(completed_personas, key=int)[:10]}... (showing first 10)")
    print(f"   Total completed: {len(completed_personas)}")
else:
    print(f"   {sample_model} / {sample_topic}: NOT FOUND")

# Check what the function would return
from duplicity.fast_robust_queries import find_missing_model_persona_combinations

missing_queries = find_missing_model_persona_combinations(main_queries, existing_data, MODELS)

total_missing = sum(len(topic_queries) for topic_queries in missing_queries.values())
print(f"\n6. Function results:")
print(f"   Total missing persona IDs: {total_missing}")

# Check specific topic
if sample_topic in missing_queries:
    missing_persona_ids = list(missing_queries[sample_topic].keys())
    print(f"   Missing personas for '{sample_topic}': {sorted(missing_persona_ids, key=int)[:10]}... (showing first 10)")
    print(f"   Total missing: {len(missing_persona_ids)}")

# Count completed combinations
print(f"\n7. Counting completed combinations:")
total_combinations = total_queries_per_model * len(MODELS)
completed_combinations = 0

for topic, topic_queries in main_queries.items():
    for persona_id in topic_queries.keys():
        for model in MODELS:
            if (model in existing_results and
                topic in existing_results[model] and
                persona_id in existing_results[model][topic]):
                completed_combinations += 1

print(f"   Total combinations expected: {total_combinations}")
print(f"   Completed combinations: {completed_combinations}")
print(f"   Missing combinations: {total_combinations - completed_combinations}")
print(f"   Completion rate: {completed_combinations/total_combinations*100:.1f}%")

print("\n" + "="*80)
