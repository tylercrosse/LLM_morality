#!/bin/bash
# Run all causal routing experiments SEQUENTIALLY (one after another)
# This avoids GPU memory issues from running multiple experiments in parallel

set -e

LOG_DIR="$HOME/LLM_morality/mech_interp_outputs/causal_routing/logs"
SCRIPT_DIR="$HOME/LLM_morality/scripts/mech_interp"

# Create log directory
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "Causal Routing Experiments - Sequential Execution"
echo "================================================================================"
echo ""
echo "Running experiments one at a time to avoid GPU memory issues"
echo ""
echo "Order:"
echo "  1. Frankenstein (LoRA weight transplant)      ~30-45 min"
echo "  2. Activation Steering                        ~30-45 min"
echo "  3. Path Patching                              ~60-90 min"
echo ""
echo "Total estimated time: 2-3 hours (sequential)"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo "================================================================================"
echo ""

# Experiment 1: Frankenstein
echo "=== [1/3] Starting Frankenstein Experiment at $(date) ==="
python "$SCRIPT_DIR/run_frankenstein.py" 2>&1 | tee "$LOG_DIR/frankenstein.log"
FRANKENSTEIN_EXIT=$?
echo "=== Frankenstein completed at $(date) with exit code $FRANKENSTEIN_EXIT ===" | tee -a "$LOG_DIR/frankenstein.log"
echo ""

if [ $FRANKENSTEIN_EXIT -ne 0 ]; then
    echo "⚠️  Frankenstein failed with exit code $FRANKENSTEIN_EXIT"
    echo "Check log: $LOG_DIR/frankenstein.log"
    echo ""
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Stopping experiments"
        exit $FRANKENSTEIN_EXIT
    fi
fi

# Experiment 2: Activation Steering
echo "=== [2/3] Starting Activation Steering at $(date) ==="
python "$SCRIPT_DIR/run_activation_steering.py" 2>&1 | tee "$LOG_DIR/steering.log"
STEERING_EXIT=$?
echo "=== Activation Steering completed at $(date) with exit code $STEERING_EXIT ===" | tee -a "$LOG_DIR/steering.log"
echo ""

if [ $STEERING_EXIT -ne 0 ]; then
    echo "⚠️  Activation Steering failed with exit code $STEERING_EXIT"
    echo "Check log: $LOG_DIR/steering.log"
    echo ""
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Stopping experiments"
        exit $STEERING_EXIT
    fi
fi

# Experiment 3: Path Patching
echo "=== [3/3] Starting Path Patching at $(date) ==="
python "$SCRIPT_DIR/run_path_patching.py" 2>&1 | tee "$LOG_DIR/path_patching.log"
PATH_PATCHING_EXIT=$?
echo "=== Path Patching completed at $(date) with exit code $PATH_PATCHING_EXIT ===" | tee -a "$LOG_DIR/path_patching.log"
echo ""

# Final summary
echo "================================================================================"
echo "All Experiments Complete!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  Frankenstein:        Exit code $FRANKENSTEIN_EXIT"
echo "  Activation Steering: Exit code $STEERING_EXIT"
echo "  Path Patching:       Exit code $PATH_PATCHING_EXIT"
echo ""
echo "Logs:"
echo "  $LOG_DIR/frankenstein.log"
echo "  $LOG_DIR/steering.log"
echo "  $LOG_DIR/path_patching.log"
echo ""
echo "Output files:"
ls -lh "$HOME/LLM_morality/mech_interp_outputs/causal_routing/" | grep -v "^d" | grep -v "^total"
echo ""
echo "================================================================================"

# Exit with failure if any experiment failed
if [ $FRANKENSTEIN_EXIT -ne 0 ] || [ $STEERING_EXIT -ne 0 ] || [ $PATH_PATCHING_EXIT -ne 0 ]; then
    exit 1
fi

exit 0
