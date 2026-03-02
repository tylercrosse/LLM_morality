#!/bin/bash
# Check status of causal routing experiments without attaching to tmux

LOG_DIR="$HOME/LLM_morality/mech_interp_outputs/causal_routing/logs"
SESSION_NAME="causal_routing"

echo "================================================================================"
echo "Causal Routing Experiments - Status Check"
echo "================================================================================"
echo "Time: $(date)"
echo ""

# Check if tmux session exists
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "❌ Session '$SESSION_NAME' is not running"
    echo ""
    echo "To start experiments:"
    echo "  ./scripts/mech_interp/run_causal_experiments_tmux.sh"
    exit 1
fi

echo "✓ Session '$SESSION_NAME' is running"
echo ""

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "⚠️  Log directory not found: $LOG_DIR"
    exit 1
fi

# Function to get last line of log and check for completion
check_experiment() {
    local name=$1
    local logfile=$2

    echo "--- $name ---"

    if [ ! -f "$logfile" ]; then
        echo "  Status: Not started (no log file)"
        return
    fi

    # Check if completed
    if grep -q "completed at" "$logfile"; then
        local completion_time=$(grep "completed at" "$logfile" | tail -n 1 | sed 's/.*completed at //')
        echo "  Status: ✓ COMPLETED at $completion_time"

        # Check exit code
        local exit_code=$(grep "Exit code:" "$logfile" | tail -n 1 | sed 's/.*Exit code: //')
        if [ "$exit_code" = "0" ]; then
            echo "  Exit code: 0 (success)"
        else
            echo "  Exit code: $exit_code (⚠️  error)"
        fi
    else
        echo "  Status: ⏳ RUNNING"

        # Show last 3 lines of output
        echo "  Recent output:"
        tail -n 3 "$logfile" | sed 's/^/    /'
    fi

    # Show file size (approximate progress indicator)
    local size=$(du -h "$logfile" | cut -f1)
    echo "  Log size: $size"
    echo ""
}

# Check each experiment
check_experiment "Frankenstein" "$LOG_DIR/frankenstein.log"
check_experiment "Activation Steering" "$LOG_DIR/steering.log"
check_experiment "Path Patching" "$LOG_DIR/path_patching.log"

echo "================================================================================"
echo "Commands:"
echo "================================================================================"
echo ""
echo "Watch live progress:"
echo "  tail -f $LOG_DIR/frankenstein.log"
echo "  tail -f $LOG_DIR/steering.log"
echo "  tail -f $LOG_DIR/path_patching.log"
echo ""
echo "Attach to tmux session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Kill all experiments:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
