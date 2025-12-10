#!/usr/bin/env python3
"""
Analyze ablation study results and generate summary report.

Usage:
    python scripts/analyze_ablations.py results/ablations_20241209_140000
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """Extract final epoch metrics from log file."""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find the last epoch line
    last_epoch = None
    for line in reversed(lines):
        if line.startswith('Epoch') and 'Loss:' in line:
            last_epoch = line
            break
    
    if not last_epoch:
        return None
    
    # Parse metrics using regex
    # Example: Epoch  50 | Loss: 0.4427/0.4546 | Train: mAP:0.418 P@50:0.497 P@100:0.393 AUROC:0.919 | Val: mAP:0.433 P@50:0.511 P@100:0.403 AUROC:0.911
    
    metrics = {}
    
    # Extract epoch number
    epoch_match = re.search(r'Epoch\s+(\d+)', last_epoch)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
    
    # Extract losses
    loss_match = re.search(r'Loss:\s+([\d.]+)/([\d.]+)', last_epoch)
    if loss_match:
        metrics['train_loss'] = float(loss_match.group(1))
        metrics['val_loss'] = float(loss_match.group(2))
    
    # Extract validation metrics (primary focus)
    val_match = re.search(r'Val:\s+mAP:([\d.]+)\s+P@50:([\d.]+)\s+P@100:([\d.]+)\s+AUROC:([\d.]+)', last_epoch)
    if val_match:
        metrics['val_mAP'] = float(val_match.group(1))
        metrics['val_P@50'] = float(val_match.group(2))
        metrics['val_P@100'] = float(val_match.group(3))
        metrics['val_AUROC'] = float(val_match.group(4))
    
    # Extract training metrics
    train_match = re.search(r'Train:\s+mAP:([\d.]+)\s+P@50:([\d.]+)\s+P@100:([\d.]+)\s+AUROC:([\d.]+)', last_epoch)
    if train_match:
        metrics['train_mAP'] = float(train_match.group(1))
        metrics['train_P@50'] = float(train_match.group(2))
        metrics['train_P@100'] = float(train_match.group(3))
        metrics['train_AUROC'] = float(train_match.group(4))
    
    # Extract configuration from earlier lines
    for line in lines[:50]:  # Check first 50 lines for config
        if 'Active tasks:' in line:
            tasks_match = re.search(r'Active tasks:\s+(.+)', line)
            if tasks_match:
                metrics['tasks'] = tasks_match.group(1).strip()
        
        if 'Task weights:' in line:
            weights_match = re.search(r'Task weights:\s+(.+)', line)
            if weights_match:
                metrics['task_weights'] = weights_match.group(1).strip()
    
    return metrics


def analyze_ablations(results_dir):
    """Analyze all ablation experiments in directory."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    # Find all log files
    log_files = sorted(results_path.glob('*.log'))
    
    if not log_files:
        print(f"Error: No log files found in {results_dir}")
        sys.exit(1)
    
    print(f"Found {len(log_files)} experiment logs")
    print("=" * 100)
    print()
    
    # Parse all experiments
    experiments = []
    for log_file in log_files:
        exp_name = log_file.stem
        metrics = parse_log_file(log_file)
        
        if metrics:
            metrics['name'] = exp_name
            experiments.append(metrics)
        else:
            print(f"Warning: Could not parse {log_file.name}")
    
    if not experiments:
        print("Error: No valid experiments found")
        sys.exit(1)
    
    # Sort by validation mAP (descending)
    experiments.sort(key=lambda x: x.get('val_mAP', 0), reverse=True)
    
    # Print summary table
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    print()
    print(f"{'Rank':<6} {'Experiment':<30} {'Val mAP':<10} {'P@50':<10} {'P@100':<10} {'AUROC':<10}")
    print("-" * 100)
    
    for i, exp in enumerate(experiments, 1):
        print(
            f"{i:<6} "
            f"{exp['name']:<30} "
            f"{exp.get('val_mAP', 0):<10.4f} "
            f"{exp.get('val_P@50', 0):<10.4f} "
            f"{exp.get('val_P@100', 0):<10.4f} "
            f"{exp.get('val_AUROC', 0):<10.4f}"
        )
    
    print()
    print("=" * 100)
    print()
    
    # Detailed analysis by phase
    print("PHASE ANALYSIS")
    print("=" * 100)
    print()
    
    # Phase 1: Task contribution
    print("Phase 1: Task Contribution")
    print("-" * 100)
    phase1 = [e for e in experiments if e['name'].startswith('0') and int(e['name'][1]) <= 4]
    for exp in sorted(phase1, key=lambda x: x['name']):
        tasks = exp.get('tasks', 'N/A')
        print(f"  {exp['name']}: {tasks:<40} → mAP {exp.get('val_mAP', 0):.4f}")
    print()
    
    # Phase 2: Task weighting
    print("Phase 2: Task Weighting")
    print("-" * 100)
    phase2 = [e for e in experiments if e['name'].startswith('0') and 5 <= int(e['name'][1]) <= 8]
    for exp in sorted(phase2, key=lambda x: x['name']):
        weights = exp.get('task_weights', 'N/A')
        print(f"  {exp['name']}: {weights:<40} → mAP {exp.get('val_mAP', 0):.4f}")
    print()
    
    # Phase 3: Focal alpha
    print("Phase 3: Focal Loss Alpha")
    print("-" * 100)
    phase3 = [e for e in experiments if e['name'].startswith('0') and 9 <= int(e['name'][:2]) <= 11]
    for exp in sorted(phase3, key=lambda x: x['name']):
        print(f"  {exp['name']}: mAP {exp.get('val_mAP', 0):.4f}")
    print()
    
    # Phase 4: Gradient clipping
    print("Phase 4: Gradient Clipping")
    print("-" * 100)
    phase4 = [e for e in experiments if e['name'].startswith('1') and 2 <= int(e['name'][1]) <= 3]
    for exp in sorted(phase4, key=lambda x: x['name']):
        print(f"  {exp['name']}: mAP {exp.get('val_mAP', 0):.4f}")
    print()
    
    # Phase 5: Model capacity
    print("Phase 5: Model Capacity")
    print("-" * 100)
    phase5 = [e for e in experiments if e['name'].startswith('1') and 4 <= int(e['name'][1]) <= 6]
    for exp in sorted(phase5, key=lambda x: x['name']):
        print(f"  {exp['name']}: mAP {exp.get('val_mAP', 0):.4f}")
    print()
    
    # Key findings
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print()
    
    best = experiments[0]
    print(f"🏆 Best Configuration: {best['name']}")
    print(f"   Val mAP: {best.get('val_mAP', 0):.4f}")
    print(f"   Val P@50: {best.get('val_P@50', 0):.4f}")
    print(f"   Val AUROC: {best.get('val_AUROC', 0):.4f}")
    print(f"   Tasks: {best.get('tasks', 'N/A')}")
    print(f"   Weights: {best.get('task_weights', 'N/A')}")
    print()
    
    # Compare to baseline
    baseline = next((e for e in experiments if 'baseline' in e['name']), None)
    if baseline:
        improvement = best.get('val_mAP', 0) - baseline.get('val_mAP', 0)
        pct_improvement = (improvement / baseline.get('val_mAP', 1)) * 100
        print(f"📈 Improvement over baseline:")
        print(f"   Baseline mAP: {baseline.get('val_mAP', 0):.4f}")
        print(f"   Best mAP: {best.get('val_mAP', 0):.4f}")
        print(f"   Absolute gain: +{improvement:.4f}")
        print(f"   Relative gain: +{pct_improvement:.2f}%")
        print()
    
    # Task contribution analysis
    print("📊 Task Contribution:")
    if phase1:
        se_only = next((e for e in phase1 if 'se_only' in e['name']), None)
        se_maccs = next((e for e in phase1 if 'se_plus_maccs' in e['name']), None)
        se_atc = next((e for e in phase1 if 'se_plus_atc' in e['name']), None)
        
        if se_only:
            print(f"   Side effects only: mAP {se_only.get('val_mAP', 0):.4f}")
            
            if se_maccs:
                gain = se_maccs.get('val_mAP', 0) - se_only.get('val_mAP', 0)
                print(f"   + MACCS: mAP {se_maccs.get('val_mAP', 0):.4f} ({gain:+.4f})")
            
            if se_atc:
                gain = se_atc.get('val_mAP', 0) - se_only.get('val_mAP', 0)
                print(f"   + ATC: mAP {se_atc.get('val_mAP', 0):.4f} ({gain:+.4f})")
    print()
    
    # Save summary to file
    summary_path = results_path / 'SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("ABLATION STUDY SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Best Configuration: {best['name']}\n")
        f.write(f"Val mAP: {best.get('val_mAP', 0):.4f}\n")
        f.write(f"Val P@50: {best.get('val_P@50', 0):.4f}\n")
        f.write(f"Val AUROC: {best.get('val_AUROC', 0):.4f}\n")
        f.write(f"Tasks: {best.get('tasks', 'N/A')}\n")
        f.write(f"Weights: {best.get('task_weights', 'N/A')}\n")
    
    print(f"Summary saved to: {summary_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument('results_dir', help='Directory containing ablation log files')
    args = parser.parse_args()
    
    analyze_ablations(args.results_dir)


if __name__ == '__main__':
    main()
