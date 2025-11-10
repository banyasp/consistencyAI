#!/usr/bin/env python3
"""
Intelligent merge of two incremental files.
Carefully combines responses, avoiding duplicates and checking every model/topic combination.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# File paths
cleaned_file = "logs/main/to_merge/incremental_results_20251110_000222_cleaned.json"
final_file = "logs/main/to_merge/incremental_results_20251110_002232_batch1216_final_390.json"

print("="*80)
print("INTELLIGENT MERGE: Combining Two Incremental Files")
print("="*80)

# Load both files
print(f"\n📂 Loading file 1 (cleaned): {cleaned_file}")
with open(cleaned_file, 'r') as f:
    cleaned_data = json.load(f)

print(f"📂 Loading file 2 (final): {final_file}")
with open(final_file, 'r') as f:
    final_data = json.load(f)

# Get results
cleaned_results = cleaned_data.get('results', {})
final_results = final_data.get('results', {})

# Stats
print(f"\n📊 File 1 (cleaned) stats:")
print(f"   Models: {len(cleaned_results)}")
cleaned_total = sum(
    len(personas)
    for model_data in cleaned_results.values()
    for personas in model_data.values()
)
print(f"   Total responses: {cleaned_total}")

print(f"\n📊 File 2 (final) stats:")
print(f"   Models: {len(final_results)}")
final_total = sum(
    len(personas)
    for model_data in final_results.values()
    for personas in model_data.values()
)
print(f"   Total responses: {final_total}")

# Merge strategy:
# 1. Start with final_results as the base (most complete)
# 2. Add any missing responses from cleaned_results
# 3. Track statistics

print(f"\n🔄 Starting intelligent merge...")
merged_results = {}
merge_stats = {
    'from_final': 0,
    'from_cleaned': 0,
    'duplicates_found': 0,
    'models_processed': set(),
    'topics_per_model': defaultdict(set)
}

# Get all models from both files
all_models = set(cleaned_results.keys()) | set(final_results.keys())
print(f"   Total unique models: {len(all_models)}")

for model in sorted(all_models):
    merged_results[model] = {}
    merge_stats['models_processed'].add(model)

    # Get topics from both files for this model
    cleaned_topics = cleaned_results.get(model, {})
    final_topics = final_results.get(model, {})
    all_topics = set(cleaned_topics.keys()) | set(final_topics.keys())

    for topic in all_topics:
        merged_results[model][topic] = {}
        merge_stats['topics_per_model'][model].add(topic)

        # Get personas from both files for this model/topic
        cleaned_personas = cleaned_topics.get(topic, {})
        final_personas = final_topics.get(topic, {})

        # Start with final file personas (most recent/complete)
        for persona_id, response in final_personas.items():
            merged_results[model][topic][persona_id] = response
            merge_stats['from_final'] += 1

        # Add any personas from cleaned file that aren't in final file
        for persona_id, response in cleaned_personas.items():
            if persona_id not in merged_results[model][topic]:
                merged_results[model][topic][persona_id] = response
                merge_stats['from_cleaned'] += 1
            else:
                merge_stats['duplicates_found'] += 1

# Calculate final stats
merged_total = sum(
    len(personas)
    for model_data in merged_results.values()
    for personas in model_data.values()
)

print(f"\n✅ Merge complete!")
print(f"\n📊 Merge Statistics:")
print(f"   Responses from final file: {merge_stats['from_final']}")
print(f"   Responses from cleaned file: {merge_stats['from_cleaned']}")
print(f"   Duplicate persona IDs skipped: {merge_stats['duplicates_found']}")
print(f"   Total merged responses: {merged_total}")
print(f"   Models in merged file: {len(merged_results)}")

# Analyze coverage per model
print(f"\n📋 Coverage Analysis:")
for model in sorted(merged_results.keys()):
    model_responses = sum(len(personas) for personas in merged_results[model].values())
    topics_covered = len(merged_results[model])
    print(f"   {model}: {model_responses} responses across {topics_covered} topics")

# Check for expected total (25 models × 1500 queries = 37,500)
expected_total = len(merged_results) * 1500
print(f"\n📊 Validation:")
print(f"   Expected total (if complete): {expected_total}")
print(f"   Actual total: {merged_total}")
print(f"   Difference: {expected_total - merged_total}")
print(f"   Completion rate: {merged_total/expected_total*100:.1f}%")

# Sample check: show persona ID ranges for first model/topic
sample_model = sorted(merged_results.keys())[0]
sample_topic = sorted(merged_results[sample_model].keys())[0]
sample_persona_ids = sorted([int(pid) for pid in merged_results[sample_model][sample_topic].keys()])
print(f"\n🔍 Sample check ({sample_model} / {sample_topic}):")
print(f"   Persona IDs: {sample_persona_ids[:5]}...{sample_persona_ids[-5:]}")
print(f"   Total personas: {len(sample_persona_ids)}")
print(f"   Expected: 100")
print(f"   Missing: {100 - len(sample_persona_ids)}")

# Check for gaps in persona IDs
gaps = []
for i in range(1, 101):
    if str(i) not in merged_results[sample_model][sample_topic]:
        gaps.append(i)
if gaps:
    print(f"   Gap(s) in persona IDs: {gaps[:10]}..." if len(gaps) > 10 else f"   Gap(s) in persona IDs: {gaps}")
else:
    print(f"   ✅ No gaps - all personas 1-100 present!")

# Create merged file with metadata
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_path = f"logs/main/incremental/incremental_results_{timestamp}_MERGED_INTELLIGENT.json"

merged_data = {
    'results': merged_results,
    'metadata': {
        'merged_from': [cleaned_file, final_file],
        'merged_at': timestamp,
        'merge_method': 'intelligent',
        'statistics': {
            'from_final': merge_stats['from_final'],
            'from_cleaned': merge_stats['from_cleaned'],
            'duplicates_skipped': merge_stats['duplicates_found'],
            'total_responses': merged_total,
            'models': len(merged_results)
        },
        'note': 'Intelligently merged two incremental files, prioritizing final file for duplicates'
    },
    'performance_metrics': final_data.get('performance_metrics', {})
}

print(f"\n💾 Saving merged file...")
with open(merged_path, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"   Saved to: {merged_path}")
print(f"   File size: {Path(merged_path).stat().st_size / 1024 / 1024:.1f} MB")

print("\n" + "="*80)
print("✅ MERGE COMPLETE!")
print("="*80)
print(f"\nNext steps:")
print(f"1. Review the merge statistics above")
print(f"2. If completion rate is close to 100%, you can proceed with analysis")
print(f"3. If there are significant gaps, you may need to resume one more time")
print(f"4. Use this merged file as your main_results data source")
print("\n" + "="*80)
