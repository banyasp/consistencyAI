"""Control experiment to measure within-model variance."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional
from pathlib import Path

from .personas import generate_queries_for_personas
from .fast_robust_queries import query_llm_fast


def load_mary_alberti_persona() -> Dict:
    """Load pre-cached Mary Alberti persona from logs/control/.

    Returns:
        Dictionary with persona data in standard format: {"rows": [{"row": {...}}]}
    """
    # Get path to Mary Alberti persona file
    project_root = Path(__file__).resolve().parent.parent
    persona_path = project_root / "logs" / "control" / "mary_alberti_persona.json"

    if not persona_path.exists():
        raise FileNotFoundError(
            f"Mary Alberti persona file not found at {persona_path}. "
            "Please ensure the control experiment setup has been completed."
        )

    with open(persona_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_control_queries(
    persona_dict: Dict,
    topic: str,
    repetitions: int = 10
) -> Dict[str, Dict[str, str]]:
    """Generate N identical queries with unique rep IDs for control experiment.

    Args:
        persona_dict: Persona dictionary (from load_mary_alberti_persona)
        topic: Single topic string to query about
        repetitions: Number of repetitions (default 10)

    Returns:
        Dictionary in format: {topic_key: {persona_uuid_rep1: query, ...}}

    Example:
        {
            "GMO_Health": {
                "e7c0574639a244c8972c92aab9501035_rep1": "query text",
                "e7c0574639a244c8972c92aab9501035_rep2": "query text",
                ...
            }
        }
    """
    # Generate the base query using standard persona query generation
    base_queries = generate_queries_for_personas(persona_dict, [topic])

    # Should have one topic with one persona
    if not base_queries or len(base_queries) == 0:
        raise ValueError("Failed to generate base query from persona")

    topic_key = list(base_queries.keys())[0]
    persona_queries = base_queries[topic_key]

    if len(persona_queries) != 1:
        raise ValueError(f"Expected 1 persona query, got {len(persona_queries)}")

    # Get the original persona UUID and query text
    original_uuid = list(persona_queries.keys())[0]
    query_text = persona_queries[original_uuid]

    # Create repetitions with unique IDs
    control_queries = {topic_key: {}}
    for rep_num in range(1, repetitions + 1):
        rep_id = f"{original_uuid}_rep{rep_num}"
        control_queries[topic_key][rep_id] = query_text

    return control_queries


def run_control_experiment(
    models: List[str],
    repetitions: int = 10,
    all_open_router: bool = True,
    initial_batch_size: int = 30,
    initial_concurrency: int = 10,
    max_concurrency: int = 50,
    adaptive_mode: bool = True,
    max_retries: int = 5,
    tag: str = "control"
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Run control experiment with Mary Alberti and GMO health topic.

    This experiment measures within-model variance by running the same prompt
    multiple times for each model.

    Args:
        models: List of model specifications to test
        repetitions: Number of times to repeat the query per model (default 10)
        all_open_router: Route non-OpenAI models through OpenRouter
                        (OpenAI models always go direct via routing logic)
        initial_batch_size: Starting batch size
        initial_concurrency: Starting concurrency
        max_concurrency: Maximum concurrency
        adaptive_mode: Auto-adjust concurrency
        max_retries: Max retry attempts
        tag: Tag for output files (default "control")

    Returns:
        Dictionary of {model: {topic: {persona_id_repN: response}}}
    """
    print("="*60)
    print("CONTROL EXPERIMENT: Within-Model Variance")
    print("="*60)
    print(f"   Persona: Mary Alberti (e7c0574639a244c8972c92aab9501035)")
    print(f"   Topic: What are the health impacts of genetically modified food?")
    print(f"   Repetitions per model: {repetitions}")
    print(f"   Models: {len(models)}")
    print(f"   Total queries: {repetitions * len(models)}")
    print()

    # Load Mary Alberti persona
    print("📋 Loading Mary Alberti persona...")
    persona = load_mary_alberti_persona()

    # Generate control queries
    print(f"🔄 Generating {repetitions} repetitions...")
    control_queries = generate_control_queries(
        persona,
        topic="What are the health impacts of genetically modified food?",
        repetitions=repetitions
    )

    # Run queries
    print("🚀 Querying models...")
    results = query_llm_fast(
        nested_queries=control_queries,
        list_of_models=models,
        initial_batch_size=initial_batch_size,
        initial_concurrency=initial_concurrency,
        max_concurrency=max_concurrency,
        adaptive_mode=adaptive_mode,
        all_open_router=all_open_router,
        max_retries=max_retries,
        ensure_100_percent_success=True,
        save_incremental=True,
        subdir="control"
    )

    print()
    print("✅ Control experiment complete!")
    print(f"   Results saved to: logs/control/")
    print()

    return results
