#!/usr/bin/env python3
"""Check the types of persona IDs in main_queries"""

from duplicity import generate_queries_for_personas, get_and_clean_personas

TOPICS = [
    "Are COVID-19 vaccines safe and effective?",
    "Is the U.S.-Mexico border secure?",
]

NUM_PERSONAS = 10

# Load personas
main_personas = get_and_clean_personas(offset=0, length=NUM_PERSONAS, cache=True, tag="main", subdir="main")

# Generate queries
main_queries = generate_queries_for_personas(main_personas, TOPICS)

# Check types
sample_topic = list(main_queries.keys())[0]
sample_persona_id = list(main_queries[sample_topic].keys())[0]

print(f"Sample topic: {sample_topic}")
print(f"Sample persona ID: {sample_persona_id}")
print(f"Type of persona ID: {type(sample_persona_id)}")
print(f"First 5 persona IDs: {list(main_queries[sample_topic].keys())[:5]}")
print(f"Types: {[type(pid) for pid in list(main_queries[sample_topic].keys())[:5]]}")
