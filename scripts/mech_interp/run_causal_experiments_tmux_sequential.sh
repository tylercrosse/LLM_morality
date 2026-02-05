#!/bin/bash
# Run all causal routing experiments SEQUENTIALLY in a tmux session
# This allows you to detach while avoiding GPU memory issues

set -e

SESSION_NAME="causal_routing"
SCRIPT_DIR="$HOME/LLM_morality/scripts/mech_interp"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session and restart: tmux kill-session -t $SESSION_NAME && $0"
    exit 1
fi

echo "================================================================================"
echo "Starting Causal Routing Experiments in tmux (Sequential Mode)"
echo "================================================================================"
echo ""
echo "Session name: $SESSION_NAME"
echo ""
echo "Experiments will run ONE AT A TIME (avoids GPU memory issues):"
echo "  1. Frankenstein (LoRA weight transplant)      ~30-45 min"
echo "  2. Activation Steering                        ~30-45 min"
echo "  3. Path Patching                              ~60-90 min"
echo ""
echo "Total estimated time: 2-3 hours"
echo ""
echo "================================================================================"
echo ""

# Create new detached session
tmux new-session -d -s "$SESSION_NAME" -n "experiments"

# Run sequential script in the session
tmux send-keys -t "$SESSION_NAME:0" "cd $HOME/LLM_morality" Enter
tmux send-keys -t "$SESSION_NAME:0" "$SCRIPT_DIR/run_causal_experiments_sequential.sh" Enter

# Print instructions
echo "✓ tmux session '$SESSION_NAME' created"
echo ""
echo "================================================================================"
echo "Next Steps:"
echo "================================================================================"
echo ""
echo "1. ATTACH to watch progress:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "2. DETACH and go home (experiments keep running):"
echo "   Ctrl+b d"
echo ""
echo "3. CHECK progress (without attaching):"
echo "   tail -f ~/LLM_morality/mech_interp_outputs/causal_routing/logs/frankenstein.log"
echo "   ./scripts/mech_interp/check_experiment_status.sh"
echo ""
echo "4. REATTACH later:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "5. KILL session (stop experiments):"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "================================================================================"
echo ""
echo "Running sequentially to avoid GPU memory issues"
echo "Logs: ~/LLM_morality/mech_interp_outputs/causal_routing/logs/"
echo ""
echo "✓ Ready! Attach now with: tmux attach -t $SESSION_NAME"
echo ""
