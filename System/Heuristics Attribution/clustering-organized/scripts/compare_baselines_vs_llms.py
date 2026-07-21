#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline vs LLM Performance Comparison - Adult Income Dataset
--------------------------------------------------------------
Compares baseline guessing strategies with LLM intent attribution performance.
Creates visualizations and detailed comparison tables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
BASELINE_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/baselines/baseline_comparison.csv"
LLM_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/analysis/analysis_comparison/summary_csvs/overall_comparison.csv"
OUTPUT_DIR = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/baselines"

def load_and_prepare_data():
    """Load baseline and LLM results and prepare for comparison."""
    
    # Load baseline results
    baseline_df = pd.read_csv(BASELINE_CSV)
    baseline_df['Type'] = 'Baseline'
    baseline_df = baseline_df.rename(columns={'Strategy': 'Model'})
    baseline_df['Trial'] = 'Baseline'
    
    # Load LLM results
    llm_df = pd.read_csv(LLM_CSV)
    llm_df['Type'] = 'LLM'
    llm_df = llm_df.rename(columns={'Macro F1': 'Macro_F1', 'Macro Precision': 'Macro_Precision', 'Macro Recall': 'Macro_Recall'})
    
    # Create combined dataset
    combined = pd.concat([
        baseline_df[['Model', 'Macro_F1', 'Type', 'Trial']],
        llm_df[['Model', 'Macro_F1', 'Type', 'Trial']]
    ], ignore_index=True)
    
    return combined, baseline_df, llm_df


def create_comparison_plot(combined_df, output_path):
    """Create a bar plot comparing baseline and LLM performance."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort by Macro F1
    combined_sorted = combined_df.sort_values('Macro_F1', ascending=True)
    
    # Create color map
    colors = ['#FF6B6B' if t == 'Baseline' else '#4ECDC4' for t in combined_sorted['Type']]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(combined_sorted)), 
                   combined_sorted['Macro_F1'],
                   color=colors,
                   edgecolor='black',
                   linewidth=0.5)
    
    # Set labels
    labels = [f"{row['Model']} ({row['Trial']})" if row['Type'] == 'LLM' else row['Model'] 
              for _, row in combined_sorted.iterrows()]
    ax.set_yticks(range(len(combined_sorted)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Intent Attribution Performance: Baselines vs LLMs (Adult Income Dataset)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(combined_sorted.iterrows()):
        f1 = row['Macro_F1']
        ax.text(f1 + 0.01, i, f'{f1:.3f}', 
                va='center', fontsize=7)
    
    # Add reference lines
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random (0.5)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Baseline Strategy'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='LLM Model'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Random (0.5)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(combined_sorted['Macro_F1']) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    plt.close()


def create_statistics_summary(combined_df, baseline_df, llm_df, output_path):
    """Create a detailed statistics summary."""
    
    with open(output_path, 'w') as f:
        f.write("# Baseline vs LLM Performance Analysis - Adult Income Dataset\n\n")
        f.write("## Summary Statistics\n\n")
        
        # Overall stats
        f.write("### Overall Performance\n\n")
        f.write("| Category | Count | Mean Macro F1 | Std Dev | Min | Max |\n")
        f.write("|----------|-------|---------------|---------|-----|-----|\n")
        
        for name, df in [('Baseline', baseline_df), ('LLM', llm_df)]:
            f1 = df['Macro_F1']
            f.write(f"| {name} | {len(f1)} | {f1.mean():.4f} | ")
            f.write(f"{f1.std():.4f} | {f1.min():.4f} | {f1.max():.4f} |\n")
        
        # Top performers
        f.write("\n### Top 10 Performers (All Categories)\n\n")
        f.write("| Rank | Model | Trial | Type | Macro F1 |\n")
        f.write("|------|-------|-------|------|----------|\n")
        
        top10 = combined_df.nlargest(10, 'Macro_F1')
        for idx, (_, row) in enumerate(top10.iterrows(), 1):
            trial = row.get('Trial', 'N/A')
            f.write(f"| {idx} | {row['Model']} | {trial} | {row['Type']} | {row['Macro_F1']:.4f} |\n")
        
        # Top LLM performers by trial
        f.write("\n### Top 5 LLM Models by Trial\n\n")
        
        for trial in ['Bare Minimum', 'With Info', 'Info + Few-Shot']:
            trial_llms = llm_df[llm_df['Trial'] == trial].nlargest(5, 'Macro_F1')
            if len(trial_llms) > 0:
                f.write(f"\n#### {trial}\n\n")
                f.write("| Rank | Model | Macro F1 | vs Random (0.5) |\n")
                f.write("|------|-------|----------|----------------|\n")
                
                for idx, (_, row) in enumerate(trial_llms.iterrows(), 1):
                    f1 = row['Macro_F1']
                    vs_random = f1 - 0.5
                    f.write(f"| {idx} | {row['Model']} | {f1:.4f} | {vs_random:+.4f} |\n")
        
        # Key insights
        f.write("\n## Key Insights\n\n")
        
        best_llm = llm_df.loc[llm_df['Macro_F1'].idxmax()]
        worst_llm = llm_df.loc[llm_df['Macro_F1'].idxmin()]
        
        f.write(f"1. **Best LLM Performance**: `{best_llm['Model']} ({best_llm['Trial']})` achieved {best_llm['Macro_F1']:.4f} Macro F1\n")
        f.write(f"2. **Worst LLM Performance**: `{worst_llm['Model']} ({worst_llm['Trial']})` achieved {worst_llm['Macro_F1']:.4f} Macro F1\n")
        
        llms_above_random = len(llm_df[llm_df['Macro_F1'] > 0.5])
        llms_total = len(llm_df)
        f.write(f"3. **LLMs Above Random Baseline (0.5)**: {llms_above_random}/{llms_total} ({llms_above_random/llms_total*100:.1f}%)\n")
        
        mean_llm = llm_df['Macro_F1'].mean()
        f.write(f"4. **Mean LLM Macro F1**: {mean_llm:.4f}\n")
        
        best_baseline = baseline_df['Macro_F1'].max()
        llms_above_best_baseline = len(llm_df[llm_df['Macro_F1'] > best_baseline])
        f.write(f"5. **LLMs Above Best Baseline ({best_baseline:.4f})**: {llms_above_best_baseline}/{llms_total} ({llms_above_best_baseline/llms_total*100:.1f}%)\n")
        
        f.write("\n## Ground Truth Distribution\n\n")
        f.write("- **Intentional**: 47.04% (13,291 changes)\n")
        f.write("- **Unintentional**: 52.96% (14,965 changes)\n")
        f.write("- **Total Changes**: 28,256\n\n")
        
        f.write("## Baseline Strategy Analysis\n\n")
        f.write("The baseline strategies show interesting patterns:\n\n")
        f.write("- **Random Guessing (50/50)**: ~0.50 Macro F1 (as expected)\n")
        f.write("- **Constant strategies** perform poorly due to class imbalance\n")
        f.write("- **Probability-based strategies** are sensitive to ground truth distribution\n\n")
        
        f.write("## Recommendations\n\n")
        best_trial = llm_df.groupby('Trial')['Macro_F1'].mean().idxmax()
        f.write(f"1. **Best prompting strategy**: {best_trial}\n")
        f.write(f"2. Models should exceed 0.5 Macro F1 to be useful\n")
        f.write(f"3. The {best_llm['Model']} model with {best_llm['Trial']} prompting shows best performance\n")
    
    print(f"✓ Statistics summary saved to: {output_path}")


def main():
    """Main execution function."""
    print("="*70)
    print("BASELINE vs LLM PERFORMANCE COMPARISON")
    print("Adult Income Dataset - Intent Attribution")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    combined_df, baseline_df, llm_df = load_and_prepare_data()
    print(f"  Loaded {len(baseline_df)} baseline strategies")
    print(f"  Loaded {len(llm_df)} LLM models")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    plot1_path = f"{OUTPUT_DIR}/baseline_vs_llm_comparison.png"
    create_comparison_plot(combined_df, plot1_path)
    
    # Create statistics summary
    print("\nGenerating statistics summary...")
    summary_path = f"{OUTPUT_DIR}/BASELINE_VS_LLM_ANALYSIS.md"
    create_statistics_summary(combined_df, baseline_df, llm_df, summary_path)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nKey Files:")
    print(f"  - {plot1_path}")
    print(f"  - {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()
