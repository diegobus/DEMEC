#!/bin/bash
# Ablation Study Script for Multi-Task Learning
# This script runs systematic ablation experiments to determine which auxiliary tasks help

set -e  # Exit on error

# Configuration
EPOCHS=50
BATCH_SIZE=32
MODEL="gcn"
HIDDEN_DIM=64
NUM_LAYERS=5
DROPOUT=0.2
LR=1e-3
SEED=42

# Output directory
OUTPUT_DIR="results/ablations_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Multi-Task Learning Ablation Study"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo ""

# Function to run experiment and save results
run_experiment() {
    local exp_name=$1
    local tasks=$2
    local task_weights=$3
    local extra_args=$4
    
    echo "----------------------------------------"
    echo "Experiment: $exp_name"
    echo "Tasks: $tasks"
    echo "Task weights: $task_weights"
    echo "----------------------------------------"
    
    local output_file="$OUTPUT_DIR/${exp_name}.log"
    
    if [ -z "$task_weights" ]; then
        python src/demec/training/train.py \
            --model $MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --hidden_dim $HIDDEN_DIM \
            --num_layers $NUM_LAYERS \
            --dropout $DROPOUT \
            --lr $LR \
            --seed $SEED \
            --tasks "$tasks" \
            $extra_args \
            2>&1 | tee "$output_file"
    else
        python src/demec/training/train.py \
            --model $MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --hidden_dim $HIDDEN_DIM \
            --num_layers $NUM_LAYERS \
            --dropout $DROPOUT \
            --lr $LR \
            --seed $SEED \
            --tasks "$tasks" \
            --task_weights "$task_weights" \
            $extra_args \
            2>&1 | tee "$output_file"
    fi
    
    echo ""
    echo "Results saved to: $output_file"
    echo ""
}

# ============================================================================
# Phase 1: Task Contribution (Which tasks help?)
# ============================================================================

echo "=========================================="
echo "PHASE 1: Task Contribution"
echo "=========================================="
echo ""

# Baseline: Side effects only
run_experiment "01_baseline_se_only" "side_effects" "" ""

# Side effects + MACCS
run_experiment "02_se_plus_maccs" "side_effects,maccs" "" ""

# Side effects + ATC
run_experiment "03_se_plus_atc" "side_effects,atc" "" ""

# All tasks (current default)
run_experiment "04_all_tasks" "side_effects,atc,maccs" "" ""

# ============================================================================
# Phase 2: Task Weighting (How to balance tasks?)
# ============================================================================

echo "=========================================="
echo "PHASE 2: Task Weighting"
echo "=========================================="
echo ""

# Equal weighting (baseline)
run_experiment "05_equal_weights" "side_effects,atc,maccs" "side_effects:1.0,atc:1.0,maccs:1.0" ""

# Prioritize side effects
run_experiment "06_prioritize_se" "side_effects,atc,maccs" "side_effects:1.0,atc:0.3,maccs:0.3" ""

# MACCS as auxiliary only
run_experiment "07_maccs_auxiliary" "side_effects,atc,maccs" "side_effects:1.0,atc:0.3,maccs:0.1" ""

# Strong MACCS regularization
run_experiment "08_strong_maccs" "side_effects,maccs" "side_effects:1.0,maccs:0.5" ""

# ============================================================================
# Phase 3: Focal Loss Alpha (Better class balance?)
# ============================================================================

echo "=========================================="
echo "PHASE 3: Focal Loss Alpha"
echo "=========================================="
echo ""

# Default alpha (0.25)
run_experiment "09_focal_alpha_025" "side_effects" "" ""

# Higher alpha (more weight on positives)
run_experiment "10_focal_alpha_050" "side_effects" "" "--focal_alpha 0.5"

run_experiment "11_focal_alpha_075" "side_effects" "" "--focal_alpha 0.75"

# ============================================================================
# Phase 4: Gradient Clipping (More stable training?)
# ============================================================================

echo "=========================================="
echo "PHASE 4: Gradient Clipping"
echo "=========================================="
echo ""

# No clipping (baseline)
run_experiment "12_no_clipping" "side_effects,maccs" "side_effects:1.0,maccs:0.3" ""

# With clipping
run_experiment "13_clip_norm_1" "side_effects,maccs" "side_effects:1.0,maccs:0.3" "--clip_grad_norm 1.0"

# ============================================================================
# Phase 5: Model Capacity (Does size matter for multi-task?)
# ============================================================================

echo "=========================================="
echo "PHASE 5: Model Capacity"
echo "=========================================="
echo ""

# Small model (64 dim)
run_experiment "14_capacity_64" "side_effects,maccs" "side_effects:1.0,maccs:0.3" ""

# Medium model (128 dim)
HIDDEN_DIM=128
DROPOUT=0.3
LR=5e-4
run_experiment "15_capacity_128" "side_effects,maccs" "side_effects:1.0,maccs:0.3" ""

# Large model (256 dim)
HIDDEN_DIM=256
DROPOUT=0.4
LR=3e-4
run_experiment "16_capacity_256" "side_effects,maccs" "side_effects:1.0,maccs:0.3" ""

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_ablations.py $OUTPUT_DIR"
echo ""
