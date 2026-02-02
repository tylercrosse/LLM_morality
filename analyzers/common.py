from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_LABELS = {
    "PT2_COREDe": "Game payoffs",
    "PT3_COREDe": "Deontological",
    "PT3_COREUt": "Utilitarian",
    "PT4_COREDe": "Game + Deontological",
}
GAMES = ["IPD", "ISH", "ICN", "ICD", "BOS"]
PROMPTS = [
    ("structured_IPD", "game"),
    ("unstructured_IPD", "question"),
    ("poetic_IPD", "moral"),
    ("explicit_IPD", "explicit IPD"),
]

MODEL_PLOT_ORDER = ["No fine-tuning", "Game payoffs", "Deontological", "Utilitarian", "Game + Deontological"]
MODEL_COLORS = {
    "No fine-tuning": "#9AA0A6",
    "Game payoffs": "#4C78A8",
    "Deontological": "#59A14F",
    "Utilitarian": "#F28E2B",
    "Game + Deontological": "#B07AA1",
}
ACTION_COLORS = {
    "C | C": "#28641E",
    "C | D": "#B0DC82",
    "D | C": "#FBE6F1",
    "D | D": "#8E0B52",
    "illegal | C": "#A9A9A9",
    "illegal | D": "#7A7A7A",
}


def apply_publication_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "legend.title_fontsize": 13,
        }
    )


def model_order(results_dir):
    results_dir = Path(results_dir)
    return [k for k in MODEL_LABELS if (results_dir / k / "run1").exists()]


def parse_action(text):
    if not isinstance(text, str):
        return "illegal"
    m = re.search(r"action(1|2)", text.lower())
    if not m:
        return "illegal"
    return "C" if m.group(1) == "1" else "D"


def extract_prev_move_from_query(query):
    if not isinstance(query, str):
        return "illegal"
    m = re.search(r"they played\\s+(action1|action2)", query.lower())
    if not m:
        return "illegal"
    return "C" if m.group(1) == "action1" else "D"


def read_game_csv(results_dir, model_key, game):
    run_dir = Path(results_dir) / model_key / "run1"
    files = list(run_dir.glob(f"*independent eval {game}.csv"))
    return pd.read_csv(files[0]) if files else None


def read_unrelated_csv(results_dir, model_key):
    run_dir = Path(results_dir) / model_key / "run1"
    files = list(run_dir.glob("*independent eval 4 unrelated queries.csv"))
    return pd.read_csv(files[0]) if files else None


def read_unstructured_csv(results_dir, model_key):
    run_dir = Path(results_dir) / model_key / "run1"
    files = list(run_dir.glob("*independent eval 2 unstructured IPD queries.csv"))
    return pd.read_csv(files[0]) if files else None
