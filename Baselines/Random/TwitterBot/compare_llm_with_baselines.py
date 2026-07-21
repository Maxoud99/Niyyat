#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare LLM Intent Attribution Results with Full Range Baselines
----------------------------------------------------------------
This script compares Gemini (and other LLM) intent attribution results
against the full range of random guessing baselines to show where the
model's performance falls in the probability spectrum.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Paths
BASE_DIR = Path("/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs")
BASELINE_SUMMARY = BASE_DIR / "baselines/full_range/full_range_summary.csv"
OUTPUT_DIR = BASE_DIR / "baselines/full_range"

# LLM Models to compare
LLM_MODELS = {
    'Gemini Bare-min': BASE_DIR / 'bare-min-gemini/evaluation_results.json',
    'Gemini Info': BASE_DIR / 'info-gemini/evaluation_results.json',
    'Gemini Few-shots': BASE_DIR / 'few-shots-gemini/evaluation_results.json',
    'Llama Bare-min': BASE_DIR / 'bare-min-llama/evaluation_results.json',
    'Llama Info': BASE_DIR / 'info-llama/evaluation_results.json',
    'Llama Few-shots': BASE_DIR / 'few-shots-llama/evaluation_results.json',
    'Mixtral Bare-min': BASE_DIR / 'bare-min-mixtral/evaluation_results.json',
    'Mixtral Info': BASE_DIR / 'info-mixtral/evaluation_results.json',
    'Mixtral Few-shots': BASE_DIR / 'few-shots-mixtral/evaluation_results.json',
    'Qwen Bare-min': BASE_DIR / 'bare-min-qwen/evaluation_results.json',
    'Qwen Info': BASE_DIR / 'info-qwen/evaluation_results.json',
    'Qwen Few-shots': BASE_DIR / 'few-shots-qwen/evaluation_results.json',
    'R1 Bare-min': BASE_DIR / 'bare-min-R1/evaluation_results.json',
    'R1 Info': BASE_DIR / 'info-R1/evaluation_results.json',
    'R1 Few-shots': BASE_DIR / 'few-shots-R1/evaluation_results.json',
}


def load_llm_results():
    """Load LLM evaluation results."""
    llm_results = {}
    
    for model_name, result_path in LLM_MODELS.items():
        if result_path.exists():
            try:
                with open(result_path, 'r') as f:
                    data = json.load(f)
                    # Extract accuracy - try different possible keys
                    if 'overall_metrics' in data:
                        accuracy = data['overall_metrics'].get('accuracy', None)
                    elif 'accuracy' in data:
                        accuracy = data['accuracy']
                    else:
                        print(f"Warning: Could not find accuracy in {model_name}")
                        continue
                    
                    if accuracy is not None:
                        llm_results[model_name] = accuracy
                        print(f"✓ Loaded {model_name}: {accuracy:.4f}")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
    
    return llm_results


def create_comparison_visualization(baselines_df, llm_results):
    """Create comprehensive comparison visualization."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Color schemes
    baseline_color = '#2E86AB'
    llm_colors = {
        'Gemini': '#9370DB',
        'Llama': '#FF6B6B',
        'Mixtral': '#4ECDC4',
        'Qwen': '#FFD93D',
        'R1': '#95E1D3'
    }
    
    # Plot 1: Full comparison with all LLMs
    ax1 = axes[0]
    
    # Plot baseline curve
    ax1.plot(baselines_df['probability_percent'], baselines_df['mean_accuracy'], 
             'o-', markersize=8, linewidth=3, color=baseline_color, 
             label='Random Guessing Baseline', zorder=1)
    
    # Add error band
    ax1.fill_between(baselines_df['probability_percent'],
                     baselines_df['mean_accuracy'] - baselines_df['std_accuracy'],
                     baselines_df['mean_accuracy'] + baselines_df['std_accuracy'],
                     alpha=0.2, color=baseline_color)
    
    # Plot theoretical line
    ax1.plot([0, 100], [0, 1], 'k--', label='Theoretical', linewidth=2, alpha=0.5)
    
    # Plot LLM results
    for model_name, accuracy in llm_results.items():
        # Determine color based on model family
        color = '#555555'  # Default gray
        for family, family_color in llm_colors.items():
            if family in model_name:
                color = family_color
                break
        
        # Determine marker based on strategy
        if 'Bare-min' in model_name:
            marker = 'o'
        elif 'Info' in model_name:
            marker = 's'
        elif 'Few-shots' in model_name:
            marker = '^'
        else:
            marker = 'D'
        
        # Plot with slight x-offset for visibility
        x_pos = 105 + list(llm_results.keys()).index(model_name) * 2
        ax1.scatter(x_pos, accuracy, marker=marker, s=200, 
                   color=color, edgecolor='black', linewidth=2,
                   label=model_name, zorder=3)
    
    ax1.set_xlabel('Probability of Predicting Intentional (%) / LLM Models →', 
                   fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('LLM Performance vs Random Guessing Baselines', 
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 105 + len(llm_results) * 2 + 5)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add reference lines
    ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax1.text(2, 0.51, 'Random Guessing (50%)', fontsize=10, color='red', alpha=0.7)
    
    # Plot 2: Focused comparison (just baselines and LLM averages)
    ax2 = axes[1]
    
    # Plot baseline curve
    ax2.plot(baselines_df['probability_percent'], baselines_df['mean_accuracy'], 
             'o-', markersize=10, linewidth=3, color=baseline_color, 
             label='Random Guessing Baseline', zorder=1)
    
    # Plot theoretical line
    ax2.plot([0, 100], [0, 1], 'k--', label='Theoretical', linewidth=2, alpha=0.5)
    
    # Group LLMs by family and calculate average
    family_averages = {}
    for model_name, accuracy in llm_results.items():
        for family in llm_colors.keys():
            if family in model_name:
                if family not in family_averages:
                    family_averages[family] = []
                family_averages[family].append(accuracy)
                break
    
    # Plot family averages
    x_offset = 105
    for family, accuracies in family_averages.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        ax2.scatter(x_offset, avg_accuracy, marker='D', s=300,
                   color=llm_colors[family], edgecolor='black', linewidth=2,
                   label=f'{family} Average ({avg_accuracy:.3f})', zorder=3)
        x_offset += 8
    
    ax2.set_xlabel('Probability of Predicting Intentional (%) / Model Family →', 
                   fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Model Family Averages vs Random Guessing Baselines', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, x_offset + 5)
    ax2.set_ylim(-0.05, 1.05)
    
    # Add reference lines
    ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax2.text(2, 0.51, 'Random (50%)', fontsize=10, color='red', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = OUTPUT_DIR / "llm_vs_baselines_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved to: {fig_path}")
    
    plt.close()


def create_performance_table(baselines_df, llm_results):
    """Create a detailed comparison table."""
    
    # Find equivalent baseline probability for each LLM
    results_table = []
    
    for model_name, accuracy in llm_results.items():
        # Find closest baseline probability
        baselines_df['diff'] = abs(baselines_df['mean_accuracy'] - accuracy)
        closest = baselines_df.loc[baselines_df['diff'].idxmin()]
        
        results_table.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Closest_Baseline_Prob': f"{closest['probability_percent']:.0f}%",
            'Baseline_Accuracy': closest['mean_accuracy'],
            'Difference': accuracy - closest['mean_accuracy'],
            'Beats_Random_50': 'Yes ✓' if accuracy > 0.5 else 'No ✗',
            'Performance': 'Excellent' if accuracy > 0.9 else 
                          'Very Good' if accuracy > 0.8 else
                          'Good' if accuracy > 0.7 else
                          'Fair' if accuracy > 0.6 else
                          'Poor'
        })
    
    df_results = pd.DataFrame(results_table)
    df_results = df_results.sort_values('Accuracy', ascending=False)
    
    # Save as CSV
    csv_path = OUTPUT_DIR / "llm_baseline_comparison_table.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Comparison table saved to: {csv_path}")
    
    # Create markdown report
    md_path = OUTPUT_DIR / "LLM_BASELINE_COMPARISON.md"
    with open(md_path, 'w') as f:
        f.write("# LLM Intent Attribution vs Random Guessing Baselines\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares LLM intent attribution performance against random guessing ")
        f.write("baselines across the full probability range (0% to 100%).\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Model | Accuracy | Closest Baseline | Baseline Acc | Difference | Beats Random | Rating |\n")
        f.write("|-------|----------|------------------|--------------|------------|--------------|--------|\n")
        
        for _, row in df_results.iterrows():
            f.write(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Closest_Baseline_Prob']} | ")
            f.write(f"{row['Baseline_Accuracy']:.4f} | {row['Difference']:+.4f} | ")
            f.write(f"{row['Beats_Random_50']} | {row['Performance']} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Calculate statistics
        best_model = df_results.iloc[0]
        worst_model = df_results.iloc[-1]
        avg_accuracy = df_results['Accuracy'].mean()
        
        f.write(f"1. **Best Performing Model**: {best_model['Model']} ({best_model['Accuracy']:.4f})\n")
        f.write(f"2. **Weakest Model**: {worst_model['Model']} ({worst_model['Accuracy']:.4f})\n")
        f.write(f"3. **Average Accuracy**: {avg_accuracy:.4f}\n")
        f.write(f"4. **Models Beating Random (50%)**: {sum(df_results['Accuracy'] > 0.5)}/{len(df_results)}\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("- **< 50%**: Worse than random guessing (likely implementation issue)\n")
        f.write("- **50-60%**: Marginally better than random\n")
        f.write("- **60-70%**: Demonstrates learning\n")
        f.write("- **70-80%**: Good performance\n")
        f.write("- **80-90%**: Very good performance\n")
        f.write("- **> 90%**: Excellent performance\n\n")
        
        f.write("## Baseline Reference\n\n")
        f.write("Random guessing with different probabilities:\n")
        f.write("| Probability | Mean Accuracy | Std Dev |\n")
        f.write("|-------------|---------------|---------|\n")
        for _, row in baselines_df.iterrows():
            f.write(f"| {row['probability_percent']:.0f}% | {row['mean_accuracy']:.4f} | ")
            f.write(f"{row['std_accuracy']:.4f} |\n")
    
    print(f"✓ Markdown report saved to: {md_path}")
    
    return df_results


def main():
    """Main execution function."""
    print("="*80)
    print("LLM vs BASELINES COMPARISON")
    print("="*80)
    
    # Load baselines
    print("\nLoading baseline results...")
    baselines_df = pd.read_csv(BASELINE_SUMMARY)
    print(f"✓ Loaded {len(baselines_df)} baseline probability points")
    
    # Load LLM results
    print("\nLoading LLM results...")
    llm_results = load_llm_results()
    
    if not llm_results:
        print("\n✗ No LLM results found! Check the paths in LLM_MODELS dictionary.")
        return
    
    print(f"\n✓ Loaded {len(llm_results)} LLM models")
    
    # Create visualizations
    print("\nCreating comparison visualizations...")
    create_comparison_visualization(baselines_df, llm_results)
    
    # Create comparison table
    print("\nGenerating comparison table...")
    df_results = create_performance_table(baselines_df, llm_results)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTop 5 Models:")
    for idx, row in df_results.head(5).iterrows():
        print(f"  {row['Model']:30s} - {row['Accuracy']:.4f} ({row['Performance']})")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
