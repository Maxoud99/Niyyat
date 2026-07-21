#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline vs LLM Performance Comparison
---------------------------------------
Compares baseline guessing strategies with LLM intent attribution performance.
Creates visualizations and detailed comparison tables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
BASELINE_CSV = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs/baselines/baseline_comparison.csv"
LLM_CSV = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs/evaluation_summary.csv"
OUTPUT_DIR = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs/baselines"

def load_and_prepare_data():
    """Load baseline and LLM results and prepare for comparison."""
    
    # Load baseline results
    baseline_df = pd.read_csv(BASELINE_CSV)
    baseline_df['Type'] = 'Baseline'
    baseline_df = baseline_df.rename(columns={'Strategy': 'Model'})
    
    # Load LLM results
    llm_df = pd.read_csv(LLM_CSV)
    llm_df['Type'] = 'LLM'
    llm_df = llm_df.rename(columns={'Experiment': 'Model'})
    
    # Combine datasets
    combined = pd.concat([
        baseline_df[['Model', 'Accuracy', 'Type']],
        llm_df[['Model', 'Accuracy', 'Type']]
    ], ignore_index=True)
    
    return combined, baseline_df, llm_df


def create_comparison_plot(combined_df, output_path):
    """Create a bar plot comparing baseline and LLM performance."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by accuracy
    combined_sorted = combined_df.sort_values('Accuracy', ascending=True)
    
    # Create color map
    colors = ['#FF6B6B' if t == 'Baseline' else '#4ECDC4' for t in combined_sorted['Type']]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(combined_sorted)), 
                   combined_sorted['Accuracy'] * 100,
                   color=colors,
                   edgecolor='black',
                   linewidth=0.5)
    
    # Set labels
    ax.set_yticks(range(len(combined_sorted)))
    ax.set_yticklabels(combined_sorted['Model'], fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Intent Attribution Performance: Baselines vs LLMs', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(combined_sorted.iterrows()):
        acc_pct = row['Accuracy'] * 100
        ax.text(acc_pct + 1, i, f'{acc_pct:.1f}%', 
                va='center', fontsize=9)
    
    # Add reference lines
    ax.axvline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random Guessing (50%)')
    ax.axvline(100, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Perfect (100%)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Baseline Strategy'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='LLM Model'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Random (50%)'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='Perfect (100%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    plt.close()


def create_category_comparison(combined_df, output_path):
    """Create a boxplot comparing baseline vs LLM categories."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    baseline_data = combined_df[combined_df['Type'] == 'Baseline']['Accuracy'] * 100
    llm_data = combined_df[combined_df['Type'] == 'LLM']['Accuracy'] * 100
    
    bp = ax.boxplot([baseline_data, llm_data],
                     labels=['Baseline Strategies', 'LLM Models'],
                     patch_artist=True,
                     widths=0.6)
    
    # Customize boxplot colors
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, data in enumerate([baseline_data, llm_data], 1):
        x = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=50, color='black')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Accuracy: Baselines vs LLMs', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add reference line at 50%
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.5, 51, 'Random Guessing', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Category comparison plot saved to: {output_path}")
    plt.close()


def create_statistics_summary(combined_df, baseline_df, llm_df, output_path):
    """Create a detailed statistics summary."""
    
    with open(output_path, 'w') as f:
        f.write("# Baseline vs LLM Performance Analysis\n\n")
        f.write("## Summary Statistics\n\n")
        
        # Overall stats
        f.write("### Overall Performance\n\n")
        f.write("| Category | Count | Mean Accuracy | Std Dev | Min | Max |\n")
        f.write("|----------|-------|---------------|---------|-----|-----|\n")
        
        for name, df in [('Baseline', baseline_df), ('LLM', llm_df)]:
            acc = df['Accuracy']
            f.write(f"| {name} | {len(acc)} | {acc.mean():.4f} ({acc.mean()*100:.2f}%) | ")
            f.write(f"{acc.std():.4f} | {acc.min():.4f} | {acc.max():.4f} |\n")
        
        # Top performers
        f.write("\n### Top 5 Performers (All Categories)\n\n")
        f.write("| Rank | Model | Type | Accuracy |\n")
        f.write("|------|-------|------|----------|\n")
        
        top5 = combined_df.nlargest(5, 'Accuracy')
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            f.write(f"| {idx} | {row['Model']} | {row['Type']} | {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%) |\n")
        
        # Top LLM performers
        f.write("\n### Top 5 LLM Models\n\n")
        f.write("| Rank | Model | Accuracy | vs Random | vs Best Baseline |\n")
        f.write("|------|-------|----------|-----------|------------------|\n")
        
        top_llms = llm_df.nlargest(5, 'Accuracy')
        best_baseline = baseline_df['Accuracy'].max()
        for idx, (_, row) in enumerate(top_llms.iterrows(), 1):
            acc = row['Accuracy']
            vs_random = (acc - 0.5) * 100
            vs_best = (acc - best_baseline) * 100
            f.write(f"| {idx} | {row['Model']} | {acc:.4f} ({acc*100:.2f}%) | ")
            f.write(f"+{vs_random:.2f}% | {vs_best:+.2f}% |\n")
        
        # Worst performers
        f.write("\n### Bottom 5 Performers (All Categories)\n\n")
        f.write("| Rank | Model | Type | Accuracy |\n")
        f.write("|------|-------|------|----------|\n")
        
        bottom5 = combined_df.nsmallest(5, 'Accuracy')
        for idx, (_, row) in enumerate(bottom5.iterrows(), 1):
            f.write(f"| {idx} | {row['Model']} | {row['Type']} | {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%) |\n")
        
        # Key insights
        f.write("\n## Key Insights\n\n")
        
        best_llm = llm_df.loc[llm_df['Accuracy'].idxmax()]
        worst_llm = llm_df.loc[llm_df['Accuracy'].idxmin()]
        
        f.write(f"1. **Best LLM Performance**: `{best_llm['Model']}` achieved {best_llm['Accuracy']*100:.2f}% accuracy\n")
        f.write(f"2. **Worst LLM Performance**: `{worst_llm['Model']}` achieved {worst_llm['Accuracy']*100:.2f}% accuracy\n")
        
        llms_above_random = len(llm_df[llm_df['Accuracy'] > 0.5])
        llms_total = len(llm_df)
        f.write(f"3. **LLMs Above Random Baseline**: {llms_above_random}/{llms_total} ({llms_above_random/llms_total*100:.1f}%)\n")
        
        llms_above_70 = len(llm_df[llm_df['Accuracy'] > 0.7])
        f.write(f"4. **LLMs Above 70% Accuracy**: {llms_above_70}/{llms_total} ({llms_above_70/llms_total*100:.1f}%)\n")
        
        mean_llm = llm_df['Accuracy'].mean()
        f.write(f"5. **Mean LLM Accuracy**: {mean_llm*100:.2f}%\n")
        
        f.write("\n## Baseline Strategy Analysis\n\n")
        f.write("The baseline strategies validate expected behaviors:\n\n")
        f.write("- **Constant Always Intentional** achieves 100% (since all ground truth is intentional)\n")
        f.write("- **Random Guessing** achieves ~50% (as expected for 50/50 chance)\n")
        f.write("- **Probability-based** strategies achieve accuracy proportional to their bias\n\n")
        
        f.write("## Recommendations\n\n")
        f.write(f"1. Models performing below 50% are worse than random guessing\n")
        f.write(f"2. Models between 50-70% show limited value over probabilistic baselines\n")
        f.write(f"3. Models above 70% demonstrate meaningful intent attribution capability\n")
        f.write(f"4. Models above 80% show strong performance but still have room for improvement\n")
    
    print(f"✓ Statistics summary saved to: {output_path}")


def main():
    """Main execution function."""
    print("="*70)
    print("BASELINE vs LLM PERFORMANCE COMPARISON")
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
    
    plot2_path = f"{OUTPUT_DIR}/baseline_vs_llm_distribution.png"
    create_category_comparison(combined_df, plot2_path)
    
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
    print(f"  - {plot2_path}")
    print(f"  - {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()
