#!/bin/bash -l

# Inference Runner Script for Fine-Tuned LLM Morality Models
# This script evaluates all 4 trained model checkpoints against a Random opponent
# across multiple game types (IPD, ISH, IVD, ICN, BOS, ICD)

echo "========================================"
echo "LLM Morality Model Inference Evaluation"
echo "========================================"
echo ""
echo "This script will evaluate 4 model checkpoints:"
echo "  1. PART2 - Game rewards only"
echo "  2. PART3-De - Deontological moral rewards"
echo "  3. PART3-Ut - Utilitarian moral rewards"
echo "  4. PART4 - Combined game + deontological rewards"
echo ""
echo "Outputs will be saved to: /LLM_morality_output/"
echo ""

# Common parameters (matching training configuration)
BASE_MODEL="google/gemma-2-2b-it"
OPP_STRAT="TFT"
RUN_IDX=1
NUM_EPISODES_TRAINED=1000
CD_TOKENS="action12"
R_ILLEGAL=6
BATCH_SIZE=5
NUM_EPISODES_EVAL=10

# You can customize evaluation settings here
# Increase NUM_EPISODES_EVAL for more thorough evaluation (default: 10)
# Adjust BATCH_SIZE if running into GPU memory issues (default: 5)

echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Training opponent: $OPP_STRAT"
echo "  Run index: $RUN_IDX"
echo "  Episodes trained: $NUM_EPISODES_TRAINED"
echo "  Action tokens: $CD_TOKENS"
echo "  Evaluation episodes: $NUM_EPISODES_EVAL"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "========================================"
echo ""

# Function to run inference for a single model
run_inference() {
    local parts_detail=$1
    local option=$2
    local model_name=$3

    echo "[$model_name] Starting evaluation..."
    echo "  Parameters: PARTs_detail=$parts_detail, option=$option"

    python src/inference_vsRandom.py \
        --base_model_id "$BASE_MODEL" \
        --opp_strat "$OPP_STRAT" \
        --run_idx "$RUN_IDX" \
        --num_episodes_trained "$NUM_EPISODES_TRAINED" \
        --PARTs_detail "$parts_detail" \
        --CD_tokens "$CD_TOKENS" \
        --r_illegal "$R_ILLEGAL" \
        --option "$option" \
        --BATCH_SIZE_eval "$BATCH_SIZE" \
        --NUM_EPISODES_eval "$NUM_EPISODES_EVAL"

    if [ $? -eq 0 ]; then
        echo "[$model_name] Evaluation completed successfully!"
    else
        echo "[$model_name] ERROR: Evaluation failed!"
        exit 1
    fi
    echo ""
}

# Run inference for all 4 models
echo "Starting inference evaluations..."
echo ""

# Model 1: PART2 - Game rewards only
run_inference "_PT2" "COREDe" "PART2 (Game Rewards)"

# Model 2: PART3-De - Deontological moral rewards
run_inference "_PT3" "COREDe" "PART3-De (Deontological)"

# Model 3: PART3-Ut - Utilitarian moral rewards
run_inference "_PT3" "COREUt" "PART3-Ut (Utilitarian)"

# Model 4: PART4 - Combined game + deontological rewards
run_inference "_PT4" "COREDe" "PART4 (Game + Deontological)"

echo "========================================"
echo "All evaluations completed successfully!"
echo "========================================"
echo ""
echo "Results are saved in: /LLM_morality_output/EVALaction12vsRandom_samestate_orderoriginal/"
echo ""
echo "Next steps:"
echo "  1. Check the output directory for CSV files with evaluation results"
echo "  2. Use plotting.py to visualize the results"
echo "  3. See INFERENCE_GUIDE.md for more details"
echo ""
