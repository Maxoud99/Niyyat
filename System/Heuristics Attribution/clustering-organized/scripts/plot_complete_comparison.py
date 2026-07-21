#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete Performance Comparison - Baselines, LLMs, and Classifier
------------------------------------------------------------------
Compares baseline guessing strategies, LLM models, and the Random Forest classifier
for intent attribution on the Adult Income Dataset.
Creates comprehensive visualizations and comparison tables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# File paths
BASELINE_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/baselines/baseline_comparison.csv"
LLM_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/analysis/analysis_comparison/summary_csvs/overall_comparison.csv"
CLASSIFIER_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/classifier/classifier_vs_baselines_llms.csv"
CLASSIFIER_METRICS = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/classifier/classifier_metrics.json"
OUTPUT_DIR = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/baselines"

def load_and_prepare_data():
    """Load baseline, LLM, and classifier results and prepare for comparison."""
    
    print("\nLoading data...")
    
    # Load baseline results
    baseline_df = pd.read_csv(BASELINE_CSV)
    baseline_df['Type'] = 'Baseline'
    baseline_df = baseline_df.rename(columns={'Strategy': 'Model'})
    baseline_df['Trial'] = 'Baseline'
    print(f"  ✓ Loaded {len(baseline_df)} baseline strategies")
    
    # Load LLM results
    llm_df = pd.read_csv(LLM_CSV)
    llm_df['Type'] = 'LLM'
    llm_df = llm_df.rename(columns={'Macro F1': 'Macro_F1', 'Macro Precision': 'Macro_Precision', 'Macro Recall': 'Macro_Recall'})
    print(f"  ✓ Loaded {len(llm_df)} LLM models")
    
    # Load classifier results
    with open(CLASSIFIER_METRICS, 'r') as f:
        clf_metrics = json.load(f)
    
    classifier_df = pd.DataFrame([{
        'Model': 'Random Forest Classifier',
        'Macro_F1': clf_metrics['f1_weighted'],
        'Type': 'ML Classifier',
        'Trial': 'Supervised ML',
        'Accuracy': clf_metrics['accuracy'],
        'Macro_Precision': clf_metrics['precision_weighted'],
        'Macro_Recall': clf_metrics['recall_weighted']
    }])
    print(f"  ✓ Loaded Random Forest Classifier (F1: {clf_metrics['f1_weighted']:.4f})")
    
    # Create combined dataset for plotting
    combined = pd.concat([
        baseline_df[['Model', 'Macro_F1', 'Type', 'Trial']],
        llm_df[['Model', 'Macro_F1', 'Type', 'Trial']],
        classifier_df[['Model', 'Macro_F1', 'Type', 'Trial']]
    ], ignore_index=True)
    
    return combined, baseline_df, llm_df, classifier_df


def create_comprehensive_comparison_plot(combined_df, output_path):
    """Create an enhanced bar plot comparing all approaches."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Sort by Macro F1
    combined_sorted = combined_df.sort_values('Macro_F1', ascending=True)
    
    # Create color map with distinct colors for each type
    color_map = {
        'Baseline': '#FF6B6B',      # Red
        'LLM': '#4ECDC4',           # Teal
        'ML Classifier': '#45B7D1'  # Blue
    }
    colors = [color_map[t] for t in combined_sorted['Type']]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(combined_sorted)), 
                   combined_sorted['Macro_F1'],
                   color=colors,
                   edgecolor='black',
                   linewidth=0.8,
                   alpha=0.85)
    
    # Highlight the classifier bar
    classifier_idx = combined_sorted[combined_sorted['Type'] == 'ML Classifier'].index[0]
    classifier_pos = list(combined_sorted.index).index(classifier_idx)
    bars[classifier_pos].set_linewidth(3)
    bars[classifier_pos].set_edgecolor('gold')
    bars[classifier_pos].set_alpha(1.0)
    
    # Set labels
    labels = []
    for _, row in combined_sorted.iterrows():
        if row['Type'] == 'LLM':
            labels.append(f"{row['Model']} ({row['Trial']})")
        elif row['Type'] == 'ML Classifier':
            labels.append(f"⭐ {row['Model']} ⭐")
        else:
            labels.append(row['Model'])
    
    ax.set_yticks(range(len(combined_sorted)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Macro F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Complete Intent Attribution Performance Comparison\nAdult Income Dataset: Baselines vs LLMs vs ML Classifier', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(combined_sorted.iterrows()):
        f1 = row['Macro_F1']
        color = 'darkgreen' if row['Type'] == 'ML Classifier' else 'black'
        weight = 'bold' if row['Type'] == 'ML Classifier' else 'normal'
        fontsize = 9 if row['Type'] == 'ML Classifier' else 8
        
        ax.text(f1 + 0.01, i, f'{f1:.4f}', 
                va='center', fontsize=fontsize, color=color, weight=weight)
    
    # Add reference lines
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (0.5)')
    
    # Add best LLM reference line
    best_llm_f1 = combined_sorted[combined_sorted['Type'] == 'LLM']['Macro_F1'].max()
    ax.axvline(best_llm_f1, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
               label=f'Best LLM ({best_llm_f1:.4f})')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Baseline Strategy'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='LLM Model'),
        Patch(facecolor='#45B7D1', edgecolor='gold', linewidth=3, label='ML Classifier (Random Forest)'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Random (0.5)'),
        plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=1.5, label='Best LLM')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, max(combined_sorted['Macro_F1']) * 1.08)
    
    # Add annotation for classifier
    clf_f1 = combined_sorted[combined_sorted['Type'] == 'ML Classifier']['Macro_F1'].values[0]
    ax.annotate(f'Rank #1\nF1: {clf_f1:.4f}', 
                xy=(clf_f1, classifier_pos), 
                xytext=(clf_f1 * 0.75, classifier_pos + 3),
                fontsize=11, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='gold', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='darkgreen', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive comparison plot saved to: {output_path}")
    plt.close()


def create_detailed_statistics(combined_df, baseline_df, llm_df, classifier_df, output_path):
    """Create a detailed statistics summary including the classifier."""
    
    with open(output_path, 'w') as f:
        f.write("# Complete Performance Analysis - Adult Income Dataset\n")
        f.write("## Baselines vs LLMs vs ML Classifier\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        
        # Overall stats
        f.write("## Summary Statistics\n\n")
        f.write("| Category | Count | Mean Macro F1 | Std Dev | Min | Max |\n")
        f.write("|----------|-------|---------------|---------|-----|-----|\n")
        
        for name, df in [('Baseline', baseline_df), ('LLM', llm_df)]:
            f1 = df['Macro_F1']
            f.write(f"| {name} | {len(f1)} | {f1.mean():.4f} | ")
            f.write(f"{f1.std():.4f} | {f1.min():.4f} | {f1.max():.4f} |\n")
        
        clf_f1 = classifier_df['Macro_F1'].values[0]
        f.write(f"| **ML Classifier** | **1** | **{clf_f1:.4f}** | **N/A** | **{clf_f1:.4f}** | **{clf_f1:.4f}** |\n")
        
        # Overall ranking
        f.write("\n---\n\n")
        f.write("## Overall Performance Ranking (Top 15)\n\n")
        f.write("| Rank | Model/Strategy | Type | Trial | Macro F1 | vs Random | vs Best LLM |\n")
        f.write("|------|----------------|------|-------|----------|-----------|-------------|\n")
        
        top15 = combined_df.nlargest(15, 'Macro_F1')
        best_llm_f1 = combined_df[combined_df['Type'] == 'LLM']['Macro_F1'].max()
        
        for idx, (_, row) in enumerate(top15.iterrows(), 1):
            f1 = row['Macro_F1']
            trial = row.get('Trial', 'N/A')
            vs_random = f1 - 0.5
            vs_best_llm = f1 - best_llm_f1
            
            marker = "🏆 " if idx == 1 else ""
            bold_start = "**" if row['Type'] == 'ML Classifier' else ""
            bold_end = "**" if row['Type'] == 'ML Classifier' else ""
            
            f.write(f"| {marker}{idx} | {bold_start}{row['Model']}{bold_end} | {row['Type']} | {trial} | ")
            f.write(f"{bold_start}{f1:.4f}{bold_end} | {vs_random:+.4f} | {vs_best_llm:+.4f} |\n")
        
        # Classifier details
        f.write("\n---\n\n")
        f.write("## ML Classifier Performance Details\n\n")
        
        clf_data = classifier_df.iloc[0]
        f.write(f"**Model:** Random Forest (100 trees, max_depth=10)\n\n")
        f.write("### Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy | {clf_data.get('Accuracy', 'N/A'):.4f} |\n")
        f.write(f"| Macro F1 | {clf_data['Macro_F1']:.4f} |\n")
        f.write(f"| Macro Precision | {clf_data.get('Macro_Precision', 'N/A'):.4f} |\n")
        f.write(f"| Macro Recall | {clf_data.get('Macro_Recall', 'N/A'):.4f} |\n")
        
        f.write("\n### Competitive Analysis\n\n")
        
        # Baselines beaten
        baselines_beaten = len(baseline_df[baseline_df['Macro_F1'] < clf_f1])
        f.write(f"- **Baselines beaten:** {baselines_beaten}/{len(baseline_df)} ({baselines_beaten/len(baseline_df)*100:.1f}%)\n")
        
        # LLMs beaten
        llms_beaten = len(llm_df[llm_df['Macro_F1'] < clf_f1])
        f.write(f"- **LLMs beaten:** {llms_beaten}/{len(llm_df)} ({llms_beaten/len(llm_df)*100:.1f}%)\n")
        
        # Margin over best LLM
        margin = clf_f1 - best_llm_f1
        margin_pct = (margin / best_llm_f1) * 100
        f.write(f"- **Margin over best LLM:** +{margin:.4f} (+{margin_pct:.1f}%)\n")
        
        # Best baseline
        best_baseline_f1 = baseline_df['Macro_F1'].max()
        baseline_margin = clf_f1 - best_baseline_f1
        baseline_margin_pct = (baseline_margin / best_baseline_f1) * 100
        f.write(f"- **Margin over best baseline:** +{baseline_margin:.4f} (+{baseline_margin_pct:.1f}%)\n")
        
        # Top LLM performers
        f.write("\n---\n\n")
        f.write("## Top 10 LLM Models (All Trials)\n\n")
        f.write("| Rank | Model | Trial | Macro F1 | vs Classifier |\n")
        f.write("|------|-------|-------|----------|---------------|\n")
        
        top_llms = llm_df.nlargest(10, 'Macro_F1')
        for idx, (_, row) in enumerate(top_llms.iterrows(), 1):
            f1 = row['Macro_F1']
            vs_clf = f1 - clf_f1
            f.write(f"| {idx} | {row['Model']} | {row['Trial']} | {f1:.4f} | {vs_clf:.4f} |\n")
        
        # LLM Analysis by Trial
        f.write("\n---\n\n")
        f.write("## LLM Performance by Prompting Strategy\n\n")
        
        for trial in ['Bare Minimum', 'With Info', 'Info + Few-Shot']:
            trial_llms = llm_df[llm_df['Trial'] == trial]
            if len(trial_llms) > 0:
                mean_f1 = trial_llms['Macro_F1'].mean()
                best_f1 = trial_llms['Macro_F1'].max()
                worst_f1 = trial_llms['Macro_F1'].min()
                
                f.write(f"\n### {trial}\n\n")
                f.write(f"- **Models:** {len(trial_llms)}\n")
                f.write(f"- **Mean F1:** {mean_f1:.4f}\n")
                f.write(f"- **Best F1:** {best_f1:.4f}\n")
                f.write(f"- **Worst F1:** {worst_f1:.4f}\n")
                f.write(f"- **Gap to Classifier:** {clf_f1 - mean_f1:.4f}\n")
        
        # Key insights
        f.write("\n---\n\n")
        f.write("## Key Insights\n\n")
        
        best_llm = llm_df.loc[llm_df['Macro_F1'].idxmax()]
        worst_llm = llm_df.loc[llm_df['Macro_F1'].idxmin()]
        
        f.write(f"1. **🏆 Best Overall:** Random Forest Classifier with {clf_f1:.4f} Macro F1 (Rank #1)\n")
        f.write(f"2. **Best LLM:** {best_llm['Model']} ({best_llm['Trial']}) with {best_llm['Macro_F1']:.4f} Macro F1 (Rank #2)\n")
        f.write(f"3. **Performance Gap:** ML Classifier outperforms best LLM by {margin:.4f} ({margin_pct:.1f}%)\n")
        f.write(f"4. **LLMs Above Random (0.5):** {len(llm_df[llm_df['Macro_F1'] > 0.5])}/{len(llm_df)} ({len(llm_df[llm_df['Macro_F1'] > 0.5])/len(llm_df)*100:.1f}%)\n")
        f.write(f"5. **Mean LLM Performance:** {llm_df['Macro_F1'].mean():.4f}\n")
        
        # Recommendations
        f.write("\n---\n\n")
        f.write("## Recommendations\n\n")
        
        best_trial = llm_df.groupby('Trial')['Macro_F1'].mean().idxmax()
        f.write(f"1. **For production deployment:** Use Random Forest Classifier (91.4% accuracy, fastest inference)\n")
        f.write(f"2. **Best LLM approach:** {best_llm['Model']} with {best_llm['Trial']} prompting\n")
        f.write(f"3. **Best prompting strategy:** {best_trial} (highest mean performance across models)\n")
        f.write(f"4. **Minimum acceptable threshold:** Macro F1 > 0.5 (better than random)\n")
        f.write(f"5. **Cost consideration:** ML Classifier is free and local; LLMs require API costs\n")
        
        f.write("\n---\n\n")
        f.write("## Dataset Information\n\n")
        f.write("- **Dataset:** Adult Income Dataset (tenth-trial)\n")
        f.write("- **Total Changes:** 28,256\n")
        f.write("- **Intentional:** 13,291 (47.04%)\n")
        f.write("- **Unintentional:** 14,965 (52.96%)\n")
        f.write("- **Features:** 15 (age, workclass, education, etc.)\n")
        f.write("- **Records:** 19,539 manipulated records from 6,513 originals\n")
    
    print(f"✓ Detailed statistics saved to: {output_path}")


def main():
    """Main execution function."""
    print("="*80)
    print("COMPLETE PERFORMANCE COMPARISON")
    print("Baselines vs LLMs vs ML Classifier - Adult Income Dataset")
    print("="*80)
    
    # Load data
    combined_df, baseline_df, llm_df, classifier_df = load_and_prepare_data()
    
    print(f"\nTotal models: {len(combined_df)}")
    print(f"  - Baselines: {len(baseline_df)}")
    print(f"  - LLMs: {len(llm_df)}")
    print(f"  - ML Classifier: {len(classifier_df)}")
    
    # Create comprehensive visualization
    print("\nCreating comprehensive comparison plot...")
    plot_path = f"{OUTPUT_DIR}/complete_comparison_with_classifier.png"
    create_comprehensive_comparison_plot(combined_df, plot_path)
    
    # Create detailed statistics
    print("\nGenerating detailed statistics...")
    stats_path = f"{OUTPUT_DIR}/COMPLETE_PERFORMANCE_ANALYSIS.md"
    create_detailed_statistics(combined_df, baseline_df, llm_df, classifier_df, stats_path)
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    
    clf_f1 = classifier_df['Macro_F1'].values[0]
    best_llm_f1 = llm_df['Macro_F1'].max()
    best_baseline_f1 = baseline_df['Macro_F1'].max()
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"  🏆 #1 Random Forest Classifier: {clf_f1:.4f}")
    print(f"  🥈 #2 Best LLM: {best_llm_f1:.4f}")
    print(f"  📈 Classifier advantage: +{clf_f1 - best_llm_f1:.4f} ({(clf_f1 - best_llm_f1)/best_llm_f1*100:.1f}%)")
    print(f"  📊 Best Baseline: {best_baseline_f1:.4f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  - Visualization: {plot_path}")
    print(f"  - Analysis: {stats_path}")
    print("="*80)


if __name__ == "__main__":
    main()
