#!/usr/bin/env python3
"""Test the patched resume function"""

from patched_resume import find_missing_model_persona_combinations_fixed
from duplicity import load_incremental_results, generate_queries_for_personas, get_and_clean_personas

# Configuration
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
print("TEST: Patched Resume Function")
print("="*80)

# Load the cleaned file
cleaned_file = "logs/main/incremental/incremental_results_20251110_000222_cleaned.json"
print(f"\n1. Loading cleaned file...")
existing_data = load_incremental_results(cleaned_file)
existing_results = existing_data.get("results", {})
print(f"   Total responses: {sum(len(personas) for model_data in existing_results.values() for personas in model_data.values())}")

# Load personas and generate queries
print(f"\n2. Generating queries...")
main_personas = get_and_clean_personas(offset=0, length=NUM_PERSONAS, cache=True, tag="main", subdir="main")
main_queries = generate_queries_for_personas(main_personas, TOPICS)
total_queries_per_model = sum(len(topic_queries) for topic_queries in main_queries.values())

# Test the PATCHED function
print(f"\n3. Testing PATCHED find_missing_model_persona_combinations...")
missing_queries = find_missing_model_persona_combinations_fixed(main_queries, existing_data, MODELS)

total_missing = sum(len(topic_queries) for topic_queries in missing_queries.values())
print(f"   Total missing persona IDs: {total_missing}")

# Check specific topic
sample_topic = "Are COVID-19 vaccines safe and effective?"
if sample_topic in missing_queries:
    missing_persona_ids = sorted(list(missing_queries[sample_topic].keys()))
    print(f"   Missing personas for '{sample_topic}': {missing_persona_ids[:10]}... (showing first 10)")
    print(f"   Total missing: {len(missing_persona_ids)}")

# Count completed combinations with FIXED logic
print(f"\n4. Counting completed combinations (with fix):")
total_combinations = total_queries_per_model * len(MODELS)
completed_combinations = 0

for topic, topic_queries in main_queries.items():
    for persona_id in topic_queries.keys():
        persona_id_str = str(persona_id)  # FIX: Convert to string
        for model in MODELS:
            if (model in existing_results and
                topic in existing_results[model] and
                persona_id_str in existing_results[model][topic]):
                completed_combinations += 1

print(f"   Total combinations expected: {total_combinations}")
print(f"   Completed combinations: {completed_combinations}")
print(f"   Missing combinations: {total_combinations - completed_combinations}")
print(f"   Completion rate: {completed_combinations/total_combinations*100:.1f}%")

print("\n" + "="*80)
print("✅ PATCHED FUNCTION WORKS!" if completed_combinations > 0 else "❌ STILL BROKEN")
print("="*80)
