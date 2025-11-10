#!/usr/bin/env python3
"""
Comprehensive validation script to verify:
1. Every model has all 15 topics
2. Every topic has exactly 100 personas (1-100)
3. Personas are in correct numerical order
4. No missing or duplicate persona IDs
"""

import json
from collections import defaultdict

# File path
reordered_file = "/Users/peterbanyas/Desktop/Cyber/openai/whitehouse/results_full2/incremental_results_20251110_074409_MERGED_REORDERED.json"

# Expected configuration
EXPECTED_MODELS = 25
EXPECTED_TOPICS = 15
EXPECTED_PERSONAS = 100
EXPECTED_TOTAL = EXPECTED_MODELS * EXPECTED_TOPICS * EXPECTED_PERSONAS

# Expected topics (from the configuration)
EXPECTED_TOPIC_LIST = [
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

print("="*80)
print("COMPREHENSIVE VALIDATION: Checking All Model/Topic Combinations")
print("="*80)

# Load the file
print(f"\n📂 Loading file...")
with open(reordered_file, 'r') as f:
    data = json.load(f)

results = data.get('results', {})

print(f"   File loaded successfully")
print(f"   Models found: {len(results)}")

# Validation tracking
issues = []
warnings = []
stats = {
    'total_models': len(results),
    'total_topics': 0,
    'total_personas': 0,
    'complete_combinations': 0,
    'incomplete_combinations': 0,
}

print(f"\n🔍 Starting validation...\n")

# Check each model
for model_idx, (model, topics) in enumerate(sorted(results.items()), 1):
    print(f"[{model_idx}/{len(results)}] Checking {model}...")

    # Check number of topics
    if len(topics) != EXPECTED_TOPICS:
        issues.append(f"Model '{model}' has {len(topics)} topics, expected {EXPECTED_TOPICS}")

    stats['total_topics'] += len(topics)

    # Check each topic
    for topic, personas in topics.items():
        num_personas = len(personas)
        stats['total_personas'] += num_personas

        # Check number of personas
        if num_personas != EXPECTED_PERSONAS:
            issues.append(f"Model '{model}' / Topic '{topic}' has {num_personas} personas, expected {EXPECTED_PERSONAS}")
            stats['incomplete_combinations'] += 1
            continue

        # Get persona IDs as integers for checking
        persona_ids = list(personas.keys())
        persona_ids_int = []

        # Convert to integers and check format
        for pid in persona_ids:
            try:
                persona_ids_int.append(int(pid))
            except ValueError:
                issues.append(f"Model '{model}' / Topic '{topic}' has non-numeric persona ID: '{pid}'")
                continue

        # Check if IDs are exactly 1-100
        expected_ids = list(range(1, 101))
        if persona_ids_int != expected_ids:
            # Find what's wrong
            missing_ids = set(expected_ids) - set(persona_ids_int)
            extra_ids = set(persona_ids_int) - set(expected_ids)
            duplicate_ids = [pid for pid in persona_ids_int if persona_ids_int.count(pid) > 1]

            if missing_ids:
                issues.append(f"Model '{model}' / Topic '{topic}' missing persona IDs: {sorted(missing_ids)}")
            if extra_ids:
                issues.append(f"Model '{model}' / Topic '{topic}' has extra persona IDs: {sorted(extra_ids)}")
            if duplicate_ids:
                issues.append(f"Model '{model}' / Topic '{topic}' has duplicate persona IDs: {sorted(set(duplicate_ids))}")

            stats['incomplete_combinations'] += 1
            continue

        # Check if ordered correctly (string comparison)
        expected_order = [str(i) for i in range(1, 101)]
        if persona_ids != expected_order:
            issues.append(f"Model '{model}' / Topic '{topic}' personas not in correct order")
            stats['incomplete_combinations'] += 1
            continue

        # Check that each persona has a response
        for pid in persona_ids:
            if not personas[pid] or personas[pid] == "":
                issues.append(f"Model '{model}' / Topic '{topic}' / Persona {pid} has empty response")

        # All checks passed
        stats['complete_combinations'] += 1

    print(f"    Topics: {len(topics)}, Personas: {sum(len(p) for p in topics.values())}")

# Print summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print(f"\n📊 Statistics:")
print(f"   Total models: {stats['total_models']} (expected: {EXPECTED_MODELS})")
print(f"   Total topics: {stats['total_topics']} (expected: {EXPECTED_MODELS * EXPECTED_TOPICS})")
print(f"   Total personas: {stats['total_personas']} (expected: {EXPECTED_TOTAL})")
print(f"   Complete model/topic combinations: {stats['complete_combinations']}")
print(f"   Incomplete model/topic combinations: {stats['incomplete_combinations']}")

# Check overall stats
if stats['total_models'] != EXPECTED_MODELS:
    issues.append(f"Total models mismatch: got {stats['total_models']}, expected {EXPECTED_MODELS}")

if stats['total_topics'] != EXPECTED_MODELS * EXPECTED_TOPICS:
    issues.append(f"Total topics mismatch: got {stats['total_topics']}, expected {EXPECTED_MODELS * EXPECTED_TOPICS}")

if stats['total_personas'] != EXPECTED_TOTAL:
    issues.append(f"Total personas mismatch: got {stats['total_personas']}, expected {EXPECTED_TOTAL}")

# Print issues
print(f"\n🔍 Issues Found: {len(issues)}")
if issues:
    print("\n❌ VALIDATION FAILED - Issues detected:\n")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print("\n✅ VALIDATION PASSED!")
    print(f"\n   All checks passed:")
    print(f"   ✓ {EXPECTED_MODELS} models")
    print(f"   ✓ Each model has {EXPECTED_TOPICS} topics")
    print(f"   ✓ Each topic has exactly 100 personas (1-100)")
    print(f"   ✓ All personas in correct numerical order")
    print(f"   ✓ No missing, duplicate, or extra persona IDs")
    print(f"   ✓ Total: {EXPECTED_TOTAL:,} responses")

# Print warnings
if warnings:
    print(f"\n⚠️  Warnings: {len(warnings)}")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

print("\n" + "="*80)

# Exit code
exit(0 if len(issues) == 0 else 1)
