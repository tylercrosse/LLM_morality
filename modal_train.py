import os
import re
import subprocess
from pathlib import Path

import modal


REPO_DIR = Path(__file__).resolve().parent
REMOTE_REPO_DIR = "/root/project"
MODEL_DIR = "/models"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04")
    .apt_install("git", "curl", "python3", "python3-venv", "python3-pip", "python-is-python3")
    .add_local_dir(
        str(REPO_DIR),
        remote_path=REMOTE_REPO_DIR,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            "logs",
            "pics",
        ],
        copy=True,
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --no-cache-dir uv",
        "python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.1.2+cu121",
        f"python -m pip install --no-cache-dir -r {REMOTE_REPO_DIR}/requirements.txt",
        "python -m pip install --no-cache-dir "
        "'trl @ git+https://github.com/huggingface/trl.git@78f8228874d5cf9c0e68952533cb377202e1eb22'",
    )
)

app = modal.App("llm-morality-train", image=image)

model_volume = modal.Volume.from_name("llm-morality-models", create_if_missing=True)


def _patch_tokens():
    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not hf_token or not wandb_api_key:
        raise RuntimeError("Missing HF_TOKEN and/or WANDB_API_KEY in environment.")

    target = Path(REMOTE_REPO_DIR) / "src" / "fine_tune.py"
    text = target.read_text()
    text = re.sub(r'hf_token\s*=\s*".*?"', f'hf_token = "{hf_token}"', text)
    text = re.sub(r'WANDB_API_KEY\s*=\s*".*?"', f'WANDB_API_KEY = "{wandb_api_key}"', text)
    target.write_text(text)


def _base_args(run_idx: int, num_episodes: int):
    return [
        "python",
        "src/fine_tune.py",
        "--base_model_id",
        "google/gemma-2-2b-it",
        "--opp_strat",
        "TFT",
        "--run",
        str(run_idx),
        "--batch_size",
        "5",
        "--num_episodes",
        str(num_episodes),
        "--payoff_version",
        "smallerR",
        "--CD_tokens",
        "action12",
        "--LoRA",
        "True",
        "--option",
        "CORE",
        "--Rscaling",
        "True",
        "--r_illegal",
        "6",
        "--r_punishment",
        "3",
        "--LoRA_rank",
        "64",
        "--gradient_accumulation_steps",
        "4",
    ]


def _commands_for_runset(run_set: str, run_idx: int):
    if run_set == "smoke":
        return [
            _base_args(run_idx, 10) + ["--do_PART2", "True"],
            _base_args(run_idx, 10) + ["--do_PART3", "True", "--moral_type", "De"],
            _base_args(run_idx, 10) + ["--do_PART3", "True", "--moral_type", "Ut"],
        ]
    if run_set == "minimal":
        return [
            _base_args(run_idx, 1000) + ["--do_PART2", "True"],
            _base_args(run_idx, 1000) + ["--do_PART3", "True", "--moral_type", "De"],
            _base_args(run_idx, 1000) + ["--do_PART3", "True", "--moral_type", "Ut"],
        ]
    if run_set == "minimal_plus":
        return [
            _base_args(run_idx, 1000) + ["--do_PART2", "True"],
            _base_args(run_idx, 1000) + ["--do_PART3", "True", "--moral_type", "De"],
            _base_args(run_idx, 1000) + ["--do_PART3", "True", "--moral_type", "Ut"],
            _base_args(run_idx, 1000) + ["--do_PART4", "True"],
        ]
    if run_set == "unlearning_de":
        return [
            _base_args(run_idx, 500)
            + ["--do_PART2", "True", "--do_PART3", "True", "--moral_type", "De"]
        ]
    if run_set == "unlearning_ut":
        return [
            _base_args(run_idx, 500)
            + ["--do_PART2", "True", "--do_PART3", "True", "--moral_type", "Ut"]
        ]
    raise ValueError(f"Unknown run_set: {run_set}")


@app.function(
    gpu="L40S",
    timeout=60 * 60 * 8,
    volumes={MODEL_DIR: model_volume},
    secrets=[
        modal.Secret.from_dotenv(),
    ],
)
def train_one(run_set: str = "minimal", run_idx: int = 1, cmd_idx: int = 0):
    os.chdir(REMOTE_REPO_DIR)
    os.makedirs(MODEL_DIR, exist_ok=True)

    _patch_tokens()

    commands = _commands_for_runset(run_set, run_idx)
    if cmd_idx < 0 or cmd_idx >= len(commands):
        raise ValueError(f"cmd_idx out of range: {cmd_idx}")

    cmd = commands[cmd_idx]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    model_volume.commit()


@app.local_entrypoint()
def main(run_set: str = "minimal", run_idx: int = 1, parallel: bool = True):
    cmd_count = len(_commands_for_runset(run_set, run_idx))
    if parallel and cmd_count > 1:
        list(
            train_one.map(
                [run_set] * cmd_count,
                [run_idx] * cmd_count,
                list(range(cmd_count)),
            )
        )
    else:
        for cmd_idx in range(cmd_count):
            train_one.remote(run_set=run_set, run_idx=run_idx, cmd_idx=cmd_idx)
