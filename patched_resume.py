#!/usr/bin/env python3
"""
Patched resume function that fixes the persona ID type mismatch bug.
Use this instead of query_llm_fast_resume() until the bug is fixed in duplicity.
"""

from duplicity import (
    load_incremental_results,
    query_llm_fast,
)
from typing import Dict, List


def find_missing_model_persona_combinations_fixed(
    original_queries: Dict[str, Dict[str, str]],
    existing_data: Dict,
    target_models: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    FIXED VERSION: Find queries that are missing for specific model-persona combinations.
    Handles type mismatch between int and str persona IDs.
    """
    missing_queries = {}
    existing_results = existing_data.get("results", {})

    for topic, topic_queries in original_queries.items():
        missing_queries[topic] = {}

        for persona_id, prompt in topic_queries.items():
            # FIX: Convert persona_id to string for comparison
            persona_id_str = str(persona_id)

            # Check if this persona is missing for ANY of the target models
            is_missing_for_any_model = False

            for model in target_models:
                if (model not in existing_results or
                    topic not in existing_results[model] or
                    persona_id_str not in existing_results[model][topic]):  # FIX: Use str version
                    # This model-persona combination is missing
                    is_missing_for_any_model = True
                    break

            if is_missing_for_any_model:
                missing_queries[topic][persona_id] = prompt

    return missing_queries


def query_llm_fast_resume_fixed(
    incremental_file_path: str,
    original_queries: Dict[str, Dict[str, str]],
    list_of_models: List[str],
    initial_batch_size: int = 50,
    initial_concurrency: int = 20,
    max_concurrency: int = 100,
    adaptive_mode: bool = True,
    all_open_router: bool = False,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    ensure_100_percent_success: bool = True,
    save_incremental: bool = True,
    incremental_interval: int = 1,
    subdir: str = ""
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    FIXED VERSION: Resume query processing from an incremental results file.
    Handles persona ID type mismatch (int vs str).
    """
    print(f"🔄 Resuming from incremental file: {incremental_file_path}")
    print(f"   Using PATCHED version that fixes persona ID type mismatch")

    # Load existing results
    existing_data = load_incremental_results(incremental_file_path)
    existing_results = existing_data.get("results", {})

    # Find missing queries for the target models (using FIXED function)
    missing_queries = find_missing_model_persona_combinations_fixed(
        original_queries, existing_data, list_of_models
    )

    # Count missing queries
    total_missing = sum(len(topic_queries) for topic_queries in missing_queries.values())
    total_original = sum(len(topic_queries) for topic_queries in original_queries.values())

    # Count completed model-persona combinations (FIX: convert to str for comparison)
    total_combinations = total_original * len(list_of_models)
    completed_combinations = 0

    for topic, topic_queries in original_queries.items():
        for persona_id in topic_queries.keys():
            persona_id_str = str(persona_id)  # FIX: Convert to string
            for model in list_of_models:
                if (model in existing_results and
                    topic in existing_results[model] and
                    persona_id_str in existing_results[model][topic]):  # FIX: Use str version
                    completed_combinations += 1

    print(f"📊 Resume Status:")
    print(f"   Total model-persona combinations: {total_combinations}")
    print(f"   Completed combinations: {completed_combinations}")
    print(f"   Missing combinations: {total_combinations - completed_combinations}")
    print(f"   Completion rate: {completed_combinations/total_combinations*100:.1f}%")
    print(f"   Missing personas to retry: {total_missing}")

    if total_missing == 0:
        print("✅ All queries already completed!")
        return existing_results

    print(f"🚀 Processing {total_missing} missing queries...")

    # Get the last batch number and incremental counter from the existing data (FIX: was looking in wrong place)
    metadata = existing_data.get("metadata", {})  # FIX: Get from existing_data not existing_results
    last_batch_number = metadata.get("batch_number", 0)
    last_incremental_counter = metadata.get("incremental_counter", 0)

    start_batch_number = last_batch_number + 1

    print(f"📊 Continuing from batch {start_batch_number}")
    print(f"📊 Continuing incremental counter from {last_incremental_counter}")

    # Process missing queries
    missing_results = query_llm_fast(
        nested_queries=missing_queries,
        list_of_models=list_of_models,
        initial_batch_size=initial_batch_size,
        initial_concurrency=initial_concurrency,
        max_concurrency=max_concurrency,
        adaptive_mode=adaptive_mode,
        all_open_router=all_open_router,
        max_retries=max_retries,
        retry_delay=retry_delay,
        ensure_100_percent_success=ensure_100_percent_success,
        save_incremental=save_incremental,
        incremental_interval=incremental_interval,
        start_batch_number=start_batch_number,
        initial_incremental_counter=last_incremental_counter,
        subdir=subdir
    )

    # Merge results - preserve ALL existing results and add new ones
    print("🔗 Merging results...")
    merged_results = existing_results.copy()

    for model, model_results in missing_results.items():
        if model not in merged_results:
            merged_results[model] = {}

        for topic, topic_results in model_results.items():
            if topic not in merged_results[model]:
                merged_results[model][topic] = {}

            # Only update with new results, don't overwrite existing ones
            for persona_id, response in topic_results.items():
                persona_id_str = str(persona_id)  # FIX: Ensure consistent string keys
                if persona_id_str not in merged_results[model][topic]:
                    merged_results[model][topic][persona_id_str] = response

    print("✅ Resume completed successfully!")
    print(f"📊 Final results contain {len(merged_results)} models")

    # Count final combinations
    final_combinations = 0
    for model in list_of_models:
        if model in merged_results:
            for topic in merged_results[model]:
                final_combinations += len(merged_results[model][topic])

    print(f"📊 Final model-persona combinations: {final_combinations}")
    return merged_results


if __name__ == "__main__":
    print("This module provides patched resume functions.")
    print("Import and use query_llm_fast_resume_fixed() in your notebook.")
