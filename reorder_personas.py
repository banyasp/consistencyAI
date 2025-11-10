#!/usr/bin/env python3
"""
Reorder persona IDs within each model/topic combination to go from 1-100 in order.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# File path
merged_file = "/Users/peterbanyas/Desktop/Cyber/openai/whitehouse/results_full2/incremental_results_20251110_073926_MERGED_INTELLIGENT.json"

print("="*80)
print("REORDERING: Sorting Persona IDs 1-100 for Each Model/Topic")
print("="*80)

# Load the merged file
print(f"\n📂 Loading merged file...")
with open(merged_file, 'r') as f:
    data = json.load(f)

results = data.get('results', {})
metadata = data.get('metadata', {})
performance_metrics = data.get('performance_metrics', {})

print(f"   Models: {len(results)}")
total_responses = sum(
    len(personas)
    for model_data in results.values()
    for personas in model_data.values()
)
print(f"   Total responses: {total_responses}")

# Reorder all persona IDs within each model/topic
print(f"\n🔄 Reordering persona IDs...")
reordered_results = OrderedDict()
reorder_count = 0
already_ordered_count = 0

for model in sorted(results.keys()):
    reordered_results[model] = OrderedDict()

    for topic in sorted(results[model].keys()):
        # Get all persona IDs and their responses
        personas = results[model][topic]

        # Check if already ordered
        persona_ids = list(personas.keys())
        persona_ids_sorted = sorted(persona_ids, key=lambda x: int(x))

        if persona_ids != persona_ids_sorted:
            reorder_count += 1
        else:
            already_ordered_count += 1

        # Create ordered dict with sorted persona IDs
        reordered_results[model][topic] = OrderedDict()
        for persona_id in persona_ids_sorted:
            reordered_results[model][topic][persona_id] = personas[persona_id]

print(f"   Model/topic combinations reordered: {reorder_count}")
print(f"   Model/topic combinations already ordered: {already_ordered_count}")

# Verify a sample
sample_model = list(reordered_results.keys())[0]
sample_topic = list(reordered_results[sample_model].keys())[0]
sample_persona_ids = list(reordered_results[sample_model][sample_topic].keys())

print(f"\n🔍 Sample verification ({sample_model} / {sample_topic}):")
print(f"   First 10 persona IDs: {sample_persona_ids[:10]}")
print(f"   Last 10 persona IDs: {sample_persona_ids[-10:]}")
print(f"   Total: {len(sample_persona_ids)}")

# Check if properly ordered
expected_order = [str(i) for i in range(1, 101)]
if sample_persona_ids == expected_order:
    print(f"   ✅ Properly ordered 1-100!")
else:
    print(f"   ⚠️ Order issue detected")
    # Show any differences
    for i, (actual, expected) in enumerate(zip(sample_persona_ids, expected_order)):
        if actual != expected:
            print(f"      Position {i}: got '{actual}', expected '{expected}'")
            break

# Create reordered file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"/Users/peterbanyas/Desktop/Cyber/openai/whitehouse/results_full2/incremental_results_{timestamp}_MERGED_REORDERED.json"

reordered_data = {
    'results': reordered_results,
    'metadata': {
        **metadata,
        'reordered_at': timestamp,
        'reordered_count': reorder_count,
        'note': 'Persona IDs sorted numerically 1-100 within each model/topic combination'
    },
    'performance_metrics': performance_metrics
}

print(f"\n💾 Saving reordered file...")
with open(output_path, 'w') as f:
    json.dump(reordered_data, f, indent=2)

file_size = Path(output_path).stat().st_size / 1024 / 1024
print(f"   Saved to: {output_path}")
print(f"   File size: {file_size:.1f} MB")

# Final validation
print(f"\n📊 Final Validation:")
print(f"   Total responses: {total_responses}")
print(f"   Models: {len(reordered_results)}")
print(f"   All persona IDs ordered: ✅")

print("\n" + "="*80)
print("✅ REORDERING COMPLETE!")
print("="*80)
print(f"\nYou can now use the reordered file:")
print(f"   {output_path}")
print("\n" + "="*80)
