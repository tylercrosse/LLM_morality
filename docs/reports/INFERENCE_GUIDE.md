# Inference Guide for Fine-Tuned LLM Morality Models

This guide explains how to run inference and evaluation on your fine-tuned models.

## Quick Start

To evaluate all 4 trained models, simply run:

```bash
./run_inference.sh
```

This will evaluate all your models against a Random opponent and save results to `/LLM_morality_output/`.

## Your Trained Models

You have 4 model checkpoints in the `/models/` directory:

| Model | Training Focus | Model Directory |
|-------|----------------|-----------------|
| **PART2** | Game rewards only (IPD payoffs) | `gemma-2-2b-it_FT_PT2_oppTFT_run1_1000ep_COREDe/` |
| **PART3-De** | Deontological moral rewards | `gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREDe/` |
| **PART3-Ut** | Utilitarian moral rewards | `gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREUt/` |
| **PART4** | Combined game + deontological | `gemma-2-2b-it_FT_PT4_oppTFT_run1_1000ep_COREDe/` |

All models were trained with:
- Base model: `google/gemma-2-2b-it`
- Opponent: TFT (Tit-for-Tat)
- Training episodes: 1000
- Action tokens: action12
- Run index: 1

## Understanding the Models

### PART2: Game Rewards
- Trained with IPD game payoffs only
- Learns to maximize points in the Prisoner's Dilemma
- No explicit moral constraints

### PART3-De: Deontological Morality
- Trained with deontological moral rewards
- Punished for defecting against cooperators
- Follows rule-based moral principles

### PART3-Ut: Utilitarian Morality
- Trained with utilitarian rewards
- Maximizes collective payoff (both players' rewards)
- Focuses on greatest good for all

### PART4: Combined
- Trained with both game rewards AND deontological moral rewards
- Balances winning the game with moral behavior

## Evaluation Process

The [src/inference_vsRandom.py](src/inference_vsRandom.py) script evaluates models across multiple game types:

1. **IPD** - Iterated Prisoner's Dilemma (training domain)
2. **ISH** - Iterated Stag Hunt
3. **IVD** - Iterated Volunteer's Dilemma
4. **ICN** - Iterated Chicken Game
5. **BOS** - Battle of the Sexes
6. **ICD** - Iterated Coordination Game

For each game, the script:
- Compares fine-tuned model responses vs base (reference) model responses
- Tests with a Random opponent (hard-coded in the script)
- Records actions taken, rewards earned, and full text responses
- Saves detailed CSV files with all metrics

## Output Structure

Results are saved to:
```
/LLM_morality_output/EVALaction12vsRandom_samestate_orderoriginal/
├── gemma-2-2b-it_FT_PT2_oppTFT_run1_1000ep_COREDe/
│   ├── EVAL After FT _PT2 - independent eval IPD.csv
│   ├── EVAL After FT _PT2 - independent eval ISH.csv
│   ├── EVAL After FT _PT2 - independent eval IVD.csv
│   ├── EVAL After FT _PT2 - independent eval ICN.csv
│   ├── EVAL After FT _PT2 - independent eval BOS.csv
│   └── EVAL After FT _PT2 - independent eval ICD.csv
├── gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREDe/
│   └── ... (same game CSVs)
├── gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREUt/
│   └── ... (same game CSVs)
└── gemma-2-2b-it_FT_PT4_oppTFT_run1_1000ep_COREDe/
    └── ... (same game CSVs)
```

Each CSV file contains:
- Episode number
- Prompts given to the model
- Model responses (before and after fine-tuning)
- Actions taken (C/D/illegal)
- Rewards earned (game, deontological, utilitarian, combined)
- Opponent actions

## Manual Inference Commands

If you want to run inference manually for a specific model:

### PART2 (Game Rewards)
```bash
python src/inference_vsRandom.py \
  --base_model_id "google/gemma-2-2b-it" \
  --opp_strat "TFT" \
  --run_idx 1 \
  --num_episodes_trained 1000 \
  --PARTs_detail "_PT2" \
  --CD_tokens "action12" \
  --r_illegal 6 \
  --option "COREDe" \
  --BATCH_SIZE_eval 5 \
  --NUM_EPISODES_eval 10
```

### PART3-De (Deontological)
```bash
python src/inference_vsRandom.py \
  --base_model_id "google/gemma-2-2b-it" \
  --opp_strat "TFT" \
  --run_idx 1 \
  --num_episodes_trained 1000 \
  --PARTs_detail "_PT3" \
  --CD_tokens "action12" \
  --r_illegal 6 \
  --option "COREDe" \
  --BATCH_SIZE_eval 5 \
  --NUM_EPISODES_eval 10
```

### PART3-Ut (Utilitarian)
```bash
python src/inference_vsRandom.py \
  --base_model_id "google/gemma-2-2b-it" \
  --opp_strat "TFT" \
  --run_idx 1 \
  --num_episodes_trained 1000 \
  --PARTs_detail "_PT3" \
  --CD_tokens "action12" \
  --r_illegal 6 \
  --option "COREUt" \
  --BATCH_SIZE_eval 5 \
  --NUM_EPISODES_eval 10
```

### PART4 (Game + Deontological)
```bash
python src/inference_vsRandom.py \
  --base_model_id "google/gemma-2-2b-it" \
  --opp_strat "TFT" \
  --run_idx 1 \
  --num_episodes_trained 1000 \
  --PARTs_detail "_PT4" \
  --CD_tokens "action12" \
  --r_illegal 6 \
  --option "COREDe" \
  --BATCH_SIZE_eval 5 \
  --NUM_EPISODES_eval 10
```

## Customizing Evaluation

You can modify the evaluation by changing parameters:

### Increase Evaluation Episodes
For more robust statistics, increase `--NUM_EPISODES_eval`:
```bash
--NUM_EPISODES_eval 100
```

### Adjust Batch Size
If you run into GPU memory issues, reduce `--BATCH_SIZE_eval`:
```bash
--BATCH_SIZE_eval 3
```

### Test Different Action Tokens
To test generalization to new action tokens:
```bash
--CD_tokens "action34"
```

### Test Different Payoff Matrix Orders
To test robustness to prompt variations:
```bash
--order_CD "permuted1"    # Permuted order 1
--order_CD "permuted2"    # Permuted order 2
--order_CD "reversed"     # Reversed C/D meaning
```

## Command-Line Parameters

| Parameter | Description | Your Value |
|-----------|-------------|------------|
| `--base_model_id` | HuggingFace model ID | `"google/gemma-2-2b-it"` |
| `--opp_strat` | Opponent strategy during training | `"TFT"` |
| `--run_idx` | Training run/seed number | `1` |
| `--num_episodes_trained` | Episodes model was trained on | `1000` |
| `--PARTs_detail` | Training part (_PT2, _PT3, _PT4) | Varies by model |
| `--CD_tokens` | Action token representation | `"action12"` |
| `--r_illegal` | Penalty for illegal actions | `6` |
| `--option` | Model naming option (COREDe/COREUt) | Varies by model |
| `--BATCH_SIZE_eval` | Batch size for evaluation | `5` (default) |
| `--NUM_EPISODES_eval` | Episodes to evaluate | `10` (default) |
| `--order_CD` | Payoff matrix order | `"original"` (default) |

## Analyzing Results

### Using plotting.py

The [plotting.py](plotting.py) script provides visualization utilities:

```python
import plotting

# Load and analyze evaluation results
# (See plotting.py for specific functions)
```

### CSV Analysis

Each CSV file contains columns like:
- `episode` - Episode number
- `query_text` - The prompt given to the model
- `response_ref` - Base model response
- `response_ft` - Fine-tuned model response
- `action_M` - Action taken by player M
- `action_O` - Action taken by opponent
- `reward_game` - Game payoff reward
- `reward_De` - Deontological reward
- `reward_Ut` - Utilitarian reward
- `reward_combined` - Combined reward

You can load these CSVs with pandas:
```python
import pandas as pd

df = pd.read_csv('/LLM_morality_output/EVALaction12vsRandom_samestate_orderoriginal/gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREDe/EVAL After FT _PT3 - independent eval IPD.csv')

# Calculate cooperation rate
coop_rate = (df['action_M'] == 'C').mean()
print(f"Cooperation rate: {coop_rate:.2%}")
```

## Key Metrics to Analyze

1. **Cooperation Rate**: Percentage of cooperative (C) actions
2. **Average Rewards**: Mean rewards across episodes
3. **Illegal Action Rate**: Percentage of invalid token generations
4. **Generalization**: Performance on non-IPD games (ISH, IVD, etc.)
5. **Before vs After**: Comparing reference model vs fine-tuned model behavior

## System Requirements

- GPU with sufficient memory (models use 4-bit quantization)
- Python environment with all dependencies from [requirements.txt](requirements.txt)
- Access to `/models/` directory with trained model checkpoints
- HuggingFace token (if downloading base model)

## Troubleshooting

### GPU Out of Memory
- Reduce `--BATCH_SIZE_eval` to 3 or even 1
- Close other GPU-intensive processes
- The script uses 4-bit quantization to minimize memory usage

### Model Not Found
- Check that model directories exist in `/models/`
- Verify model names match the pattern: `gemma-2-2b-it_FT_{PARTs_detail}_oppTFT_run{run_idx}_{num_episodes}ep_{option}/`

### Long Evaluation Times
- Each model evaluation can take significant time due to multiple game types
- Consider running evaluations in parallel or on separate GPU nodes
- Reduce `--NUM_EPISODES_eval` for quicker tests

## Advanced Usage

### Running Specific Game Types
The inference script evaluates all game types by default. To modify this, you would need to edit [src/inference_vsRandom.py](src/inference_vsRandom.py) and comment out specific game evaluation sections.

### Testing with Different Opponents
The test-time opponent is hard-coded to "Random" in line 916 of inference_vsRandom.py. To change this:
1. Edit `opponent_strategy_foreval = 'Random'` to another strategy
2. Options: 'Random', 'TFT', 'AC' (Always Cooperate), 'AD' (Always Defect), 'LLM'

### Reference Model with Value Prompts
To evaluate the base model with explicit value prompts:
```bash
--ref_value_only "De"  # or "Ut"
```

## Further Reading

- [README.md](README.md) - Project overview and paper citation
- [JOBS_gemma_final.txt](JOBS_gemma_final.txt) - Complete list of training and inference jobs from the paper
- [inference_main.sh](inference_main.sh) - Advanced inference patterns with different orders
- [modal_train.py](modal_train.py) - Training infrastructure used to create your models

## Questions?

If you have questions about:
- **Training parameters**: See [modal_train.py](modal_train.py)
- **Inference parameters**: See [src/inference_vsRandom.py](src/inference_vsRandom.py) lines 888-909
- **Game mechanics**: See [src/fine_tune.py](src/fine_tune.py) for reward function definitions
- **Paper details**: See the [arXiv paper](https://arxiv.org/abs/2410.01639)
