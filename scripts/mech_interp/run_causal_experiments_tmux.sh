#!/bin/bash
# Run all causal routing experiments in parallel tmux sessions
# This allows you to detach and leave while experiments run in background

set -e

SESSION_NAME="causal_routing"
LOG_DIR="$HOME/LLM_morality/mech_interp_outputs/causal_routing/logs"
SCRIPT_DIR="$HOME/LLM_morality/scripts/mech_interp"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session and restart: tmux kill-session -t $SESSION_NAME && $0"
    exit 1
fi

echo "================================================================================"
echo "Starting Causal Routing Experiments in tmux"
echo "================================================================================"
echo ""
echo "Session name: $SESSION_NAME"
echo "Log directory: $LOG_DIR"
echo ""
echo "Experiments:"
echo "  Window 0: Frankenstein (LoRA weight transplant)"
echo "  Window 1: Activation Steering"
echo "  Window 2: Path Patching"
echo ""
echo "================================================================================"
echo ""

# Create new detached session with first window for Frankenstein
tmux new-session -d -s "$SESSION_NAME" -n "frankenstein"

# Set up Frankenstein experiment (Window 0)
tmux send-keys -t "$SESSION_NAME:0" "cd $HOME/LLM_morality" Enter
tmux send-keys -t "$SESSION_NAME:0" "echo '=== Starting Frankenstein Experiment at \$(date) ===' | tee $LOG_DIR/frankenstein.log" Enter
tmux send-keys -t "$SESSION_NAME:0" "python $SCRIPT_DIR/run_frankenstein.py 2>&1 | tee -a $LOG_DIR/frankenstein.log" Enter
tmux send-keys -t "$SESSION_NAME:0" "echo '=== Frankenstein completed at \$(date) ===' | tee -a $LOG_DIR/frankenstein.log" Enter
tmux send-keys -t "$SESSION_NAME:0" "echo 'Exit code: \$?' | tee -a $LOG_DIR/frankenstein.log" Enter

# Create window for Activation Steering (Window 1)
tmux new-window -t "$SESSION_NAME:1" -n "steering"
tmux send-keys -t "$SESSION_NAME:1" "cd $HOME/LLM_morality" Enter
tmux send-keys -t "$SESSION_NAME:1" "echo '=== Starting Activation Steering at \$(date) ===' | tee $LOG_DIR/steering.log" Enter
tmux send-keys -t "$SESSION_NAME:1" "python $SCRIPT_DIR/run_activation_steering.py 2>&1 | tee -a $LOG_DIR/steering.log" Enter
tmux send-keys -t "$SESSION_NAME:1" "echo '=== Activation Steering completed at \$(date) ===' | tee -a $LOG_DIR/steering.log" Enter
tmux send-keys -t "$SESSION_NAME:1" "echo 'Exit code: \$?' | tee -a $LOG_DIR/steering.log" Enter

# Create window for Path Patching (Window 2)
tmux new-window -t "$SESSION_NAME:2" -n "path_patching"
tmux send-keys -t "$SESSION_NAME:2" "cd $HOME/LLM_morality" Enter
tmux send-keys -t "$SESSION_NAME:2" "echo '=== Starting Path Patching at \$(date) ===' | tee $LOG_DIR/path_patching.log" Enter
tmux send-keys -t "$SESSION_NAME:2" "python $SCRIPT_DIR/run_path_patching.py 2>&1 | tee -a $LOG_DIR/path_patching.log" Enter
tmux send-keys -t "$SESSION_NAME:2" "echo '=== Path Patching completed at \$(date) ===' | tee -a $LOG_DIR/path_patching.log" Enter
tmux send-keys -t "$SESSION_NAME:2" "echo 'Exit code: \$?' | tee -a $LOG_DIR/path_patching.log" Enter

# Create monitoring window (Window 3)
tmux new-window -t "$SESSION_NAME:3" -n "monitor"
tmux send-keys -t "$SESSION_NAME:3" "cd $HOME/LLM_morality" Enter
tmux send-keys -t "$SESSION_NAME:3" "clear" Enter
tmux send-keys -t "$SESSION_NAME:3" "cat << 'EOF'
================================================================================
Causal Routing Experiments - Monitor
================================================================================

Quick Commands:
  tail -f $LOG_DIR/frankenstein.log     # Watch Frankenstein progress
  tail -f $LOG_DIR/steering.log         # Watch Steering progress
  tail -f $LOG_DIR/path_patching.log    # Watch Path Patching progress

  # Check all logs
  tail -n 20 $LOG_DIR/*.log

  # Switch between experiment windows:
  Ctrl+b 0  - Frankenstein window
  Ctrl+b 1  - Steering window
  Ctrl+b 2  - Path Patching window
  Ctrl+b 3  - This monitor window

  # Detach from session (leave it running):
  Ctrl+b d

  # Kill session (stop all experiments):
  tmux kill-session -t $SESSION_NAME

================================================================================

Checking experiment status...
EOF
" Enter

tmux send-keys -t "$SESSION_NAME:3" "sleep 2 && watch -n 10 'echo \"=== Experiment Status at \$(date) ===\"  && echo \"\" && for log in $LOG_DIR/*.log; do echo \"--- \$(basename \$log) ---\"; tail -n 3 \$log; echo \"\"; done'" Enter

# Select the monitor window by default
tmux select-window -t "$SESSION_NAME:3"

# Print instructions
echo "✓ tmux session '$SESSION_NAME' created with 4 windows:"
echo ""
echo "  Window 0 (frankenstein):  Running Frankenstein experiment"
echo "  Window 1 (steering):      Running Activation Steering experiment"
echo "  Window 2 (path_patching): Running Path Patching experiment"
echo "  Window 3 (monitor):       Status monitor (you are here)"
echo ""
echo "================================================================================"
echo "Next Steps:"
echo "================================================================================"
echo ""
echo "1. ATTACH to the session to watch progress:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "2. SWITCH between windows (after attaching):"
echo "   Ctrl+b 0  →  Frankenstein"
echo "   Ctrl+b 1  →  Steering"
echo "   Ctrl+b 2  →  Path Patching"
echo "   Ctrl+b 3  →  Monitor"
echo ""
echo "3. DETACH and go home (experiments keep running):"
echo "   Ctrl+b d"
echo ""
echo "4. CHECK logs remotely (without attaching):"
echo "   tail -f $LOG_DIR/frankenstein.log"
echo "   tail -f $LOG_DIR/steering.log"
echo "   tail -f $LOG_DIR/path_patching.log"
echo ""
echo "5. REATTACH later (when you come back):"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "6. KILL session (stop all experiments):"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "================================================================================"
echo "Estimated Runtime: 3-4 hours total"
echo "================================================================================"
echo ""
echo "Logs are being written to: $LOG_DIR/"
echo ""
echo "✓ Ready! Attach now with: tmux attach -t $SESSION_NAME"
echo ""
