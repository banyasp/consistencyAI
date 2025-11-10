#!/usr/bin/env python3
"""
Emergency script to merge progress from cleaned file into latest batch file.
This recovers lost progress when resume didn't load the cleaned file properly.
"""

import json
from pathlib import Path
from datetime import datetime

# File paths
cleaned_file = "logs/main/incremental/incremental_results_20251110_000222_cleaned.json"
latest_batch = "logs/main/incremental/incremental_results_20251110_000622_batch004_batch_003.json"

print("="*80)
print("EMERGENCY: Merging Progress from Cleaned File")
print("="*80)

# Load both files
print(f"\n📂 Loading cleaned file: {cleaned_file}")
with open(cleaned_file, 'r') as f:
    cleaned_data = json.load(f)

print(f"📂 Loading latest batch: {latest_batch}")
with open(latest_batch, 'r') as f:
    batch_data = json.load(f)

# Get results
cleaned_results = cleaned_data.get('results', {})
batch_results = batch_data.get('results', {})

print(f"\n📊 Cleaned file stats:")
print(f"   Models: {len(cleaned_results)}")
cleaned_total = sum(
    len(personas)
    for model_data in cleaned_results.values()
    for personas in model_data.values()
)
print(f"   Total responses: {cleaned_total}")

print(f"\n📊 Batch file stats:")
print(f"   Models: {len(batch_results)}")
batch_total = sum(
    len(personas)
    for model_data in batch_results.values()
    for personas in model_data.values()
)
print(f"   Total responses: {batch_total}")

# Merge: Start with cleaned data, then add any new responses from batch
print(f"\n🔄 Merging responses...")
merged_results = {}

# First, copy all cleaned data
for model, topics in cleaned_results.items():
    if model not in merged_results:
        merged_results[model] = {}
    for topic, personas in topics.items():
        if topic not in merged_results[model]:
            merged_results[model][topic] = {}
        merged_results[model][topic].update(personas)

# Then add any new responses from batch (this captures any work done after cleaning)
for model, topics in batch_results.items():
    if model not in merged_results:
        merged_results[model] = {}
    for topic, personas in topics.items():
        if topic not in merged_results[model]:
            merged_results[model][topic] = {}
        # Only add personas that aren't already in merged results
        for persona_id, response in personas.items():
            if persona_id not in merged_results[model][topic]:
                merged_results[model][topic][persona_id] = response

merged_total = sum(
    len(personas)
    for model_data in merged_results.values()
    for personas in model_data.values()
)

print(f"\n✅ Merged results:")
print(f"   Models: {len(merged_results)}")
print(f"   Total responses: {merged_total}")
print(f"   Recovered: {merged_total - batch_total} responses")

# Save merged file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_path = f"logs/main/incremental/incremental_results_{timestamp}_MERGED.json"

merged_data = {
    'results': merged_results,
    'metadata': {
        'merged_from': [cleaned_file, latest_batch],
        'merged_at': timestamp,
        'note': 'Emergency merge to recover lost progress'
    }
}

with open(merged_path, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"\n💾 Merged file saved to:")
print(f"   {merged_path}")

print(f"\n📋 Next steps:")
print(f"   1. STOP any currently running cells in the notebook")
print(f"   2. In the resume cell, explicitly specify the merged file:")
print(f"      incremental_file_path='{merged_path}'")
print(f"   3. Re-run the resume cell to continue from the merged progress")

print("\n" + "="*80)
