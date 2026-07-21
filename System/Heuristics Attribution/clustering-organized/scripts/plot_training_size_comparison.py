#!/usr/bin/env python3
"""
Plot Training Size Comparison with Baselines and LLMs

Creates a comprehensive visualization comparing:
- Random Forest with different training sizes (1% to 100%)
- Baseline strategies
- LLM models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Paths
RESULTS_DIR = Path('/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results')
CLASSIFIER_SIZES_DIR = RESULTS_DIR / 'classifier_varied_sizes'
BASELINES_DIR = RESULTS_DIR / 'baselines'
OUTPUT_DIR = CLASSIFIER_SIZES_DIR

# Load data
print("Loading data...")

# Training size results
training_sizes = pd.read_csv(CLASSIFIER_SIZES_DIR / 'training_size_comparison.csv')
training_sizes['model_name'] = training_sizes['train_fraction'].apply(lambda x: f'RF-{int(x*100)}%')
training_sizes['type'] = 'Random Forest'
training_sizes['f1_weighted'] = training_sizes['f1_weighted']
training_sizes['accuracy'] = training_sizes['accuracy']

# Baseline results
baseline_file = BASELINES_DIR / 'baseline_comparison.csv'
if baseline_file.exists():
    baselines_raw = pd.read_csv(baseline_file)
    baselines = []
    for _, row in baselines_raw.iterrows():
        baselines.append({
            'model_name': row['Strategy'],
            'type': 'Baseline',
            'accuracy': row['Accuracy'],
            'f1_weighted': row['Macro_F1']  # Using Macro F1 as weighted F1
        })
    baselines_df = pd.DataFrame(baselines)
else:
    baselines_df = pd.DataFrame(columns=['model_name', 'type', 'accuracy', 'f1_weighted'])

# LLM results - check for classifier comparison file which has both
classifier_comparison_file = RESULTS_DIR / 'classifier' / 'classifier_vs_baselines_llms.csv'
if classifier_comparison_file.exists():
    all_models_df = pd.read_csv(classifier_comparison_file)
    llm_df = all_models_df[all_models_df['Type'] == 'LLM'][['Model/Strategy', 'Accuracy', 'F1_Weighted']].copy()
    llm_df.columns = ['model_name', 'accuracy', 'f1_weighted']
    llm_df['type'] = 'LLM'
else:
    llm_df = pd.DataFrame(columns=['model_name', 'accuracy', 'f1_weighted', 'type'])

# Combine all data
all_results = pd.concat([
    training_sizes[['model_name', 'type', 'accuracy', 'f1_weighted']],
    baselines_df,
    llm_df
], ignore_index=True)

# Sort by F1 score
all_results = all_results.sort_values('f1_weighted', ascending=False)

print(f"\nTotal models: {len(all_results)}")
print(f"  Random Forest: {len(training_sizes)}")
print(f"  Baselines: {len(baselines_df)}")
print(f"  LLMs: {len(llm_df)}")

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Color mapping
color_map = {
    'Random Forest': '#2ecc71',  # Green
    'Baseline': '#3498db',       # Blue
    'LLM': '#e74c3c'            # Red
}

# Plot 1: F1 Score comparison
ax1 = axes[0]
for model_type in ['Random Forest', 'Baseline', 'LLM']:
    data = all_results[all_results['type'] == model_type]
    if len(data) > 0:
        ax1.barh(data['model_name'], data['f1_weighted'], 
                label=model_type, color=color_map[model_type], alpha=0.7)

ax1.set_xlabel('F1 Weighted Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison: F1 Weighted Score\n(Random Forest with Varied Training Sizes vs Baselines vs LLMs)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_xlim(0, 1.0)

# Add value labels
for idx, row in all_results.iterrows():
    ax1.text(row['f1_weighted'] + 0.01, row['model_name'], 
            f"{row['f1_weighted']:.4f}", 
            va='center', fontsize=8)

# Plot 2: Accuracy comparison
ax2 = axes[1]
for model_type in ['Random Forest', 'Baseline', 'LLM']:
    data = all_results[all_results['type'] == model_type]
    if len(data) > 0:
        ax2.barh(data['model_name'], data['accuracy'], 
                label=model_type, color=color_map[model_type], alpha=0.7)

ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison: Accuracy\n(Random Forest with Varied Training Sizes vs Baselines vs LLMs)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, 1.0)

# Add value labels
for idx, row in all_results.iterrows():
    ax2.text(row['accuracy'] + 0.01, row['model_name'], 
            f"{row['accuracy']:.4f}", 
            va='center', fontsize=8)

plt.tight_layout()

# Save
output_file = OUTPUT_DIR / 'training_size_vs_baselines_llms.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot: {output_file}")

# Create learning curve plot
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot RF training size progression
training_sizes_sorted = training_sizes.sort_values('train_fraction')
ax.plot(training_sizes_sorted['train_fraction'] * 100, 
        training_sizes_sorted['f1_weighted'], 
        marker='o', linewidth=2, markersize=8, label='Random Forest', color='#2ecc71')

ax.plot(training_sizes_sorted['train_fraction'] * 100, 
        training_sizes_sorted['accuracy'], 
        marker='s', linewidth=2, markersize=8, label='Random Forest (Accuracy)', 
        color='#27ae60', linestyle='--', alpha=0.7)

# Add horizontal lines for best baseline and best LLM
if len(baselines_df) > 0:
    best_baseline_f1 = baselines_df['f1_weighted'].max()
    best_baseline_name = baselines_df.loc[baselines_df['f1_weighted'].idxmax(), 'model_name']
    ax.axhline(y=best_baseline_f1, color='#3498db', linestyle='--', linewidth=2, 
              label=f'Best Baseline: {best_baseline_name} ({best_baseline_f1:.4f})')

if len(llm_df) > 0:
    best_llm_f1 = llm_df['f1_weighted'].max()
    best_llm_name = llm_df.loc[llm_df['f1_weighted'].idxmax(), 'model_name']
    ax.axhline(y=best_llm_f1, color='#e74c3c', linestyle='--', linewidth=2,
              label=f'Best LLM: {best_llm_name} ({best_llm_f1:.4f})')

ax.set_xlabel('Training Data Size (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Random Forest Learning Curve\nvs Best Baseline and Best LLM', 
            fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

# Annotate points
for idx, row in training_sizes_sorted.iterrows():
    ax.annotate(f"{row['f1_weighted']:.3f}", 
               (row['train_fraction']*100, row['f1_weighted']),
               textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()

output_file2 = OUTPUT_DIR / 'learning_curve_comparison.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Saved learning curve: {output_file2}")

# Print summary
print("\n" + "="*70)
print("SUMMARY: Top 10 Models by F1 Weighted Score")
print("="*70)
print(all_results.head(10).to_string(index=False))
print("\n" + "="*70)

# Find where RF models rank
rf_ranks = []
for idx, row in all_results.iterrows():
    if row['type'] == 'Random Forest':
        rank = all_results.index.get_loc(idx) + 1
        rf_ranks.append((row['model_name'], rank, len(all_results)))

print("\nRandom Forest Rankings:")
for name, rank, total in rf_ranks:
    print(f"  {name}: Rank {rank}/{total}")

print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")
