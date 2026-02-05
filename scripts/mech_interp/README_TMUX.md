# Running Causal Routing Experiments in Background

This guide shows how to run the three causal routing experiments in parallel tmux sessions, allowing you to detach and leave while they run.

## Quick Start (Recommended: Sequential Mode)

```bash
# 1. Start experiments in background tmux session (runs one at a time - GPU safe)
./scripts/mech_interp/run_causal_experiments_tmux_sequential.sh

# 2. Attach to watch progress (optional)
tmux attach -t causal_routing

# 3. Detach and go home (experiments keep running)
# Press: Ctrl+b, then d

# 4. Come back later and check status
./scripts/mech_interp/check_experiment_status.sh

# 5. Reattach to see live output
tmux attach -t causal_routing
```

## Alternative: Run Without tmux (Foreground)

```bash
# Run sequentially in foreground (you'll see output directly)
./scripts/mech_interp/run_causal_experiments_sequential.sh
```

## Experiments Running

### Sequential Mode (Recommended - GPU Safe)

Runs experiments **one at a time** to avoid GPU memory issues:

1. **Frankenstein**: LoRA weight transplant - ~30-45 min
2. **Activation Steering**: Steering L2_MLP activations - ~30-45 min
3. **Path Patching**: L2→L9 pathway causality - ~60-90 min

**Total time: 2-3 hours (sequential)**

Each experiment loads multiple models, so running sequentially prevents OOM errors.

### Parallel Mode (Advanced - High Memory)

⚠️ **Only use if you have >20GB GPU memory**

The parallel version (`run_causal_experiments_tmux.sh`) launches all 3 experiments simultaneously in separate tmux windows. Faster but may cause GPU OOM errors.

**Total time: ~90 min (limited by slowest experiment)**

## tmux Navigation

After attaching with `tmux attach -t causal_routing`:

| Keys | Action |
|------|--------|
| `Ctrl+b 0` | Switch to Frankenstein window |
| `Ctrl+b 1` | Switch to Steering window |
| `Ctrl+b 2` | Switch to Path Patching window |
| `Ctrl+b 3` | Switch to Monitor window |
| `Ctrl+b d` | Detach (leave running) |
| `Ctrl+b [` | Scroll mode (use arrows, q to exit) |

## Checking Status Without Attaching

```bash
# Quick status check
./scripts/mech_interp/check_experiment_status.sh

# Watch live log output
tail -f ~/LLM_morality/mech_interp_outputs/causal_routing/logs/frankenstein.log
tail -f ~/LLM_morality/mech_interp_outputs/causal_routing/logs/steering.log
tail -f ~/LLM_morality/mech_interp_outputs/causal_routing/logs/path_patching.log

# See last 20 lines of all logs
tail -n 20 ~/LLM_morality/mech_interp_outputs/causal_routing/logs/*.log
```

## Output Locations

**Logs** (stdout/stderr):
- `~/LLM_morality/mech_interp_outputs/causal_routing/logs/frankenstein.log`
- `~/LLM_morality/mech_interp_outputs/causal_routing/logs/steering.log`
- `~/LLM_morality/mech_interp_outputs/causal_routing/logs/path_patching.log`

**Results** (CSV, PNG, JSON):
- `~/LLM_morality/mech_interp_outputs/causal_routing/frankenstein_*.csv`
- `~/LLM_morality/mech_interp_outputs/causal_routing/steering_*.csv`
- `~/LLM_morality/mech_interp_outputs/causal_routing/path_patch_*.csv`
- `~/LLM_morality/mech_interp_outputs/causal_routing/*.png`

## Common Scenarios

### Scenario 1: Start and leave immediately

```bash
./scripts/mech_interp/run_causal_experiments_tmux.sh
# Session starts, you can leave immediately
# Experiments run in background
```

### Scenario 2: Watch for a bit, then leave

```bash
./scripts/mech_interp/run_causal_experiments_tmux.sh
tmux attach -t causal_routing

# Watch for a few minutes, then:
# Press Ctrl+b, then d
# Now you can close terminal and go home
```

### Scenario 3: Check status while away

```bash
# SSH back in, or in another terminal
./scripts/mech_interp/check_experiment_status.sh

# Or watch live:
tail -f ~/LLM_morality/mech_interp_outputs/causal_routing/logs/frankenstein.log
```

### Scenario 4: Come back and analyze results

```bash
# Check if experiments finished
./scripts/mech_interp/check_experiment_status.sh

# If all completed, results are in:
ls -lh ~/LLM_morality/mech_interp_outputs/causal_routing/

# You can safely kill the session now:
tmux kill-session -t causal_routing
```

## Troubleshooting

### "Session already exists"

If you get this error, either:
- Attach to the existing session: `tmux attach -t causal_routing`
- Kill it and restart: `tmux kill-session -t causal_routing && ./run_causal_experiments_tmux.sh`

### Experiment failed with error

1. Attach to the session: `tmux attach -t causal_routing`
2. Switch to the failed experiment's window (Ctrl+b 0/1/2)
3. Check error message in terminal
4. Check full log: `cat ~/LLM_morality/mech_interp_outputs/causal_routing/logs/<experiment>.log`

### GPU out of memory

The experiments run sequentially within each window (not all at once), so GPU memory should be OK. If you get OOM:
- Check what else is running: `nvidia-smi`
- Kill session and run one at a time manually

### Want to pause/resume

tmux doesn't have native pause, but you can:
- Kill the session: `tmux kill-session -t causal_routing`
- Restart later with the same script
- Individual experiments will resume from scratch (they don't checkpoint)

## tmux Cheat Sheet

```bash
# List all sessions
tmux ls

# Attach to session
tmux attach -t causal_routing

# Kill session
tmux kill-session -t causal_routing

# Kill all tmux sessions
tmux kill-server

# Create new window (while attached)
Ctrl+b c

# Rename window (while attached)
Ctrl+b ,

# Split pane horizontally
Ctrl+b "

# Split pane vertically
Ctrl+b %
```

## Success Criteria

After all experiments complete (~3-4 hours), you should see:

```bash
$ ls mech_interp_outputs/causal_routing/
frankenstein_PT3_COREDe_to_PT2_COREDe_L2.csv
frankenstein_PT2_COREDe_to_PT3_COREDe_L2.csv
frankenstein_comparison.png
frankenstein_per_scenario_heatmap.png
steering_vector_L2_mlp_De_minus_Strategic.pt
steering_sweep_PT2_COREDe_L2_mlp.csv
steering_sweep_PT2_COREDe_L2_mlp.png
path_patch_PT3_COREDe_to_PT2_COREDe_L2-L9_residual.csv
progressive_patch_comparison.png
... (and more)
```

**Expected results:**
- Frankenstein: 3-4 hypotheses supported (cooperation shifts >5%)
- Steering: Monotonic relationship between strength and cooperation
- Path Patching: Effect size >30% (much larger than single-component's 0%)

## Remote SSH Persistence

If running over SSH and worried about disconnection:

```bash
# Before starting, ensure tmux is available
which tmux

# Start experiments
./scripts/mech_interp/run_causal_experiments_tmux.sh

# Detach immediately (don't attach first)
# Session is running in background

# You can now safely disconnect SSH
# When you reconnect, the session will still be running
```

Even if your SSH connection drops, tmux keeps the session alive.

## GPU Monitoring

While experiments run, you can monitor GPU usage in another terminal:

```bash
watch -n 5 nvidia-smi
```

Expected GPU usage:
- Memory: ~8-12 GB (Gemma-2-2b-it with LoRA)
- Utilization: 80-100% during forward passes
- Should release memory between experiments

## Next Steps After Completion

Once all experiments finish:

1. Check results: `./scripts/mech_interp/check_experiment_status.sh`
2. Review outputs: `ls -lh mech_interp_outputs/causal_routing/`
3. Analyze CSV files and plots
4. Update MECH_INTERP_RESEARCH_LOG.md with findings
5. Update WRITE_UP.md with causal evidence section
6. Kill tmux session: `tmux kill-session -t causal_routing`
