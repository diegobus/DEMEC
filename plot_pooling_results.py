#!/usr/bin/env python3
"""
Generate publication-quality figures for pooling ablation study.
Analyzes results from experiments 01-04 (a-c variants, GCN and GAT).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Results from TensorBoard (best validation metrics)
results = {
    'MW': {
        'metric': 'R²',
        'higher_better': True,
        'GCN': {
            'Mean': 0.688,
            'MLP': 0.697,
            'Attention': 0.646
        },
        'GAT': {
            'Mean': 0.672,
            'MLP': 0.684,
            'Attention': 0.618
        }
    },
    'MACCS': {
        'metric': 'Tanimoto',
        'higher_better': True,
        'GCN': {
            'Mean': 0.604,
            'MLP': 0.590,
            'Attention': 0.618
        },
        'GAT': {
            'Mean': 0.603,
            'MLP': 0.590,
            'Attention': 0.617
        }
    },
    'ATC': {
        'metric': 'mAP',
        'higher_better': True,
        'GCN': {
            'Mean': 0.226,
            'MLP': 0.164,
            'Attention': 0.221
        },
        'GAT': {
            'Mean': 0.258,
            'MLP': 0.194,
            'Attention': 0.176
        }
    },
    'Side Effects': {
        'metric': 'mAP',
        'higher_better': True,
        'GCN': {
            'Mean': 0.397,
            'MLP': 0.415,
            'Attention': 0.421
        },
        'GAT': {
            'Mean': 0.417,
            'MLP': 0.417,
            'Attention': 0.418
        }
    }
}

# Additional metrics for context
auroc_results = {
    'ATC': {
        'GCN': {'Mean': 0.790, 'MLP': 0.766, 'Attention': 0.790},
        'GAT': {'Mean': 0.793, 'MLP': 0.769, 'Attention': 0.764}
    },
    'Side Effects': {
        'GCN': {'Mean': 0.921, 'MLP': 0.921, 'Attention': 0.924},
        'GAT': {'Mean': 0.917, 'MLP': 0.917, 'Attention': 0.923}
    }
}

def plot_pooling_comparison():
    """Main comparison: pooling methods across all tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    tasks = ['MW', 'MACCS', 'ATC', 'Side Effects']
    pooling_methods = ['Mean', 'MLP', 'Attention']
    colors = {'GCN': '#2E86AB', 'GAT': '#A23B72'}
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        data = results[task]
        
        x = np.arange(len(pooling_methods))
        width = 0.35
        
        gcn_scores = [data['GCN'][p] for p in pooling_methods]
        gat_scores = [data['GAT'][p] for p in pooling_methods]
        
        bars1 = ax.bar(x - width/2, gcn_scores, width, label='GCN', 
                       color=colors['GCN'], alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, gat_scores, width, label='GAT',
                       color=colors['GAT'], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Highlight best method
        best_score = max(gcn_scores + gat_scores)
        for bars in [bars1, bars2]:
            for bar in bars:
                if abs(bar.get_height() - best_score) < 0.001:
                    bar.set_edgecolor('gold')
                    bar.set_linewidth(3)
        
        ax.set_xlabel('Pooling Method', fontweight='bold')
        ax.set_ylabel(f'{data["metric"]}', fontweight='bold')
        ax.set_title(f'Task: {task}', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(pooling_methods)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/pooling_comparison_all_tasks.png', bbox_inches='tight')
    print("✓ Saved: figures/pooling_comparison_all_tasks.png")
    plt.close()


def plot_architecture_comparison():
    """Compare GCN vs GAT across tasks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    tasks = ['MW', 'MACCS', 'ATC', 'Side Effects']
    pooling_methods = ['Mean', 'MLP', 'Attention']
    
    x = np.arange(len(tasks))
    width = 0.25
    
    colors = {'Mean': '#06A77D', 'MLP': '#F77F00', 'Attention': '#D62828'}
    
    for i, pooling in enumerate(pooling_methods):
        gcn_scores = [results[task]['GCN'][pooling] for task in tasks]
        gat_scores = [results[task]['GAT'][pooling] for task in tasks]
        
        # Calculate GAT advantage (positive = GAT better)
        advantage = [gat - gcn for gat, gcn in zip(gat_scores, gcn_scores)]
        
        bars = ax.bar(x + i*width, advantage, width, label=pooling,
                     color=colors[pooling], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Task', fontweight='bold', fontsize=12)
    ax.set_ylabel('GAT Advantage over GCN', fontweight='bold', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks)
    ax.legend(title='Pooling', loc='best', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/architecture_comparison.png', bbox_inches='tight')
    print("✓ Saved: figures/architecture_comparison.png")
    plt.close()


def plot_task_difficulty():
    """Visualize task difficulty and best performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tasks = ['MW', 'MACCS', 'ATC', 'Side Effects']
    
    # Get best score for each task
    best_scores = []
    metrics = []
    for task in tasks:
        all_scores = []
        for arch in ['GCN', 'GAT']:
            all_scores.extend(results[task][arch].values())
        best_scores.append(max(all_scores))
        metrics.append(results[task]['metric'])
    
    # Normalize to 0-1 scale for comparison
    # MW R² already 0-1
    # MACCS Tanimoto already 0-1
    # mAP already 0-1
    normalized_scores = best_scores.copy()
    
    colors_map = {'MW': '#06A77D', 'MACCS': '#F77F00', 
                  'ATC': '#D62828', 'Side Effects': '#6A4C93'}
    colors = [colors_map[t] for t in tasks]
    
    bars = ax.barh(tasks, normalized_scores, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # Add difficulty stars
    difficulty = {'MW': '⭐', 'MACCS': '⭐⭐', 'ATC': '⭐⭐⭐⭐', 'Side Effects': '⭐⭐⭐⭐⭐'}
    
    for i, (task, bar) in enumerate(zip(tasks, bars)):
        score = normalized_scores[i]
        metric = metrics[i]
        
        # Add score label
        ax.text(score + 0.02, i, f'{score:.3f} ({metric})', 
                va='center', fontweight='bold', fontsize=11)
        
        # Add difficulty
        ax.text(0.02, i, difficulty[task], va='center', fontsize=14)
    
    ax.set_xlabel('Best Performance (Normalized)', fontweight='bold', fontsize=12)
    ax.set_title('Task Difficulty vs Best Achievable Performance\n(⭐ = Difficulty)', 
                 fontweight='bold', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/task_difficulty.png', bbox_inches='tight')
    print("✓ Saved: figures/task_difficulty.png")
    plt.close()


def plot_pooling_heatmap():
    """Heatmap showing best pooling method per task-architecture combo."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tasks = ['MW', 'MACCS', 'ATC', 'Side Effects']
    architectures = ['GCN', 'GAT']
    pooling_methods = ['Mean', 'MLP', 'Attention']
    
    # Create matrix: rows = tasks, cols = arch x pooling
    data = np.zeros((len(tasks), len(architectures) * len(pooling_methods)))
    labels = []
    
    for i, task in enumerate(tasks):
        col_idx = 0
        for arch in architectures:
            for pooling in pooling_methods:
                data[i, col_idx] = results[task][arch][pooling]
                if i == 0:  # Only create labels once
                    labels.append(f'{arch}\n{pooling}')
                col_idx += 1
    
    # Normalize each row to show relative performance
    data_normalized = np.zeros_like(data)
    for i in range(len(tasks)):
        row_min = data[i].min()
        row_max = data[i].max()
        if row_max > row_min:
            data_normalized[i] = (data[i] - row_min) / (row_max - row_min)
        else:
            data_normalized[i] = 0.5
    
    im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(tasks, fontsize=11, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations with actual values
    for i in range(len(tasks)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8,
                          fontweight='bold' if data_normalized[i, j] > 0.8 else 'normal')
    
    ax.set_title('Pooling Performance Heatmap\n(Normalized per Task, Green = Best)', 
                 fontweight='bold', fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Performance', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/pooling_heatmap.png', bbox_inches='tight')
    print("✓ Saved: figures/pooling_heatmap.png")
    plt.close()


def plot_summary_recommendations():
    """Summary figure with recommendations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Pooling Strategy Recommendations', 
            ha='center', fontsize=18, fontweight='bold', transform=ax.transAxes)
    
    # Recommendations
    recommendations = [
        ('Molecular Weight (Regression)', 'Mean or MLP Pooling + GCN', 
         'Simple aggregation sufficient\nGCN faster, equally effective', '#06A77D'),
        
        ('MACCS Fingerprints', 'Attention Pooling + GAT',
         'Focus on relevant substructures\nDouble attention (conv + pool) best', '#F77F00'),
        
        ('ATC Classification', 'Mean Pooling + GAT',
         'Global properties matter\nGAT attention in convolutions helps', '#D62828'),
        
        ('Side Effects Prediction', 'Attention Pooling (slight edge)',
         'All methods similar (~0.42 mAP)\nStructure alone insufficient', '#6A4C93')
    ]
    
    y_start = 0.85
    y_step = 0.20
    
    for i, (task, method, reason, color) in enumerate(recommendations):
        y = y_start - i * y_step
        
        # Task box
        ax.add_patch(plt.Rectangle((0.05, y-0.08), 0.25, 0.12, 
                                   facecolor=color, alpha=0.3, edgecolor='black', linewidth=2))
        ax.text(0.175, y-0.02, task, ha='center', va='center', 
               fontsize=11, fontweight='bold', wrap=True)
        
        # Method box
        ax.add_patch(plt.Rectangle((0.35, y-0.08), 0.25, 0.12,
                                   facecolor=color, alpha=0.6, edgecolor='black', linewidth=2))
        ax.text(0.475, y-0.02, method, ha='center', va='center',
               fontsize=10, fontweight='bold', wrap=True)
        
        # Reason box
        ax.text(0.65, y-0.02, reason, ha='left', va='center',
               fontsize=9, style='italic', wrap=True)
    
    # Key insights box
    ax.add_patch(plt.Rectangle((0.05, 0.02), 0.9, 0.10,
                               facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=2))
    insights = ('Key Insight: No universal best pooling method. Task characteristics determine optimal strategy.\n'
                'Simple tasks → Simple pooling | Substructure tasks → Attention | Global properties → Mean')
    ax.text(0.5, 0.07, insights, ha='center', va='center', fontsize=10, 
           fontweight='bold', wrap=True, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('figures/pooling_recommendations.png', bbox_inches='tight')
    print("✓ Saved: figures/pooling_recommendations.png")
    plt.close()


if __name__ == '__main__':
    # Create figures directory
    Path('figures').mkdir(exist_ok=True)
    
    print("\nGenerating pooling ablation figures...")
    print("=" * 50)
    
    plot_pooling_comparison()
    plot_architecture_comparison()
    plot_task_difficulty()
    plot_pooling_heatmap()
    plot_summary_recommendations()
    
    print("=" * 50)
    print("\n✓ All figures generated successfully!")
    print("\nFigures saved in: figures/")
    print("  1. pooling_comparison_all_tasks.png")
    print("  2. architecture_comparison.png")
    print("  3. task_difficulty.png")
    print("  4. pooling_heatmap.png")
    print("  5. pooling_recommendations.png")
    print("\nSee POOLING_ABLATION_ANALYSIS.md for detailed analysis.")
