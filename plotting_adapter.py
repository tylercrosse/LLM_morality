import ast
import importlib.util
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
import types

import matplotlib.pyplot as plt
import pandas as pd


class PlottingAdapter:
    """Adapts ./results layout to plotting.py expectations without mutating originals."""

    def __init__(
        self,
        results_dir="./results",
        opponent="TFT",
        num_episodes_trained=1000,
        model_name="gemma-2-2b-it",
        eval_dir="publication_figures",
        plotting_path="plotting.py",
    ):
        self.results_dir = Path(results_dir).resolve()
        self.opponent = opponent
        self.num_episodes_trained = num_episodes_trained
        self.model_name = model_name
        self.eval_dir = eval_dir
        self.plotting_path = Path(plotting_path).resolve()
        self.config = self._auto_detect_config()
        self._plotting_module = None

    def _auto_detect_config(self):
        config = {}
        if not self.results_dir.exists():
            return config
        for model_dir in self.results_dir.iterdir():
            if not model_dir.is_dir():
                continue
            name = model_dir.name
            parts = name.split("_")
            if len(parts) < 2 or not parts[0].startswith("PT"):
                continue
            part = parts[0][2:]
            if "COREDe" in name:
                moral = "De"
                option = "COREDe"
            elif "COREUt" in name:
                moral = "Ut"
                option = "COREUt"
            else:
                continue
            config[name] = {
                "part": part,
                "part_detail": f"_PT{part}",
                "moral": moral,
                "option": option,
                "model_dir": name,
            }
        return config

    def _expected_model_dir(self, part_detail, option, num_episodes):
        return f"{self.model_name}_FT_{part_detail}_opp{self.opponent}_{num_episodes}ep_{option}"

    def _ensure_symlink(self, target, source):
        target = Path(target)
        if target.exists() or target.is_symlink():
            return
        target.symlink_to(source)

    def _transform_unstructured(self, src_path, dst_path):
        df = pd.read_csv(src_path)
        mappings = {
            "structured_IPD": ("response (before) - game", "response (after) - game"),
            "unstructured_IPD": ("response (before) - question", "response (after) - question"),
            "poetic_IPD": ("response (before) - moral", "response (after) - moral"),
            "explicit_IPD": ("response (before) - explicit IPD", "response (after) - explicit IPD"),
        }
        for key, (before_col, after_col) in mappings.items():
            if before_col in df.columns:
                df[f"response (before) - {key}"] = df[before_col]
            if after_col in df.columns:
                df[f"response (after) - {key}"] = df[after_col]
        df.to_csv(dst_path, index=False)

    def _prepare_run_dir(self, source_run_dir, target_run_dir, source_part_detail, target_part_detail):
        source_run_dir = Path(source_run_dir)
        target_run_dir = Path(target_run_dir)
        target_run_dir.mkdir(parents=True, exist_ok=True)
        for item in source_run_dir.iterdir():
            if not item.is_file():
                continue
            target_name = item.name.replace(
                f"EVAL After FT {source_part_detail}",
                f"EVAL After FT {target_part_detail}",
            )
            target_path = target_run_dir / target_name
            if "2 unstructured IPD queries.csv" in item.name:
                self._transform_unstructured(item, target_path)
            else:
                self._ensure_symlink(target_path, item)

    def _build_symlink_tree(self, temp_root):
        temp_root = Path(temp_root)
        run_ids = ["run1", "run2", "run3", "run5", "run6"]

        def add_model(part_detail, option, num_episodes, source_key, source_part_detail=None):
            if source_key not in self.config:
                return
            source_run = self.results_dir / source_key / "run1"
            if not source_run.exists():
                return
            if source_part_detail is None:
                source_part_detail = part_detail
            target_model_dir = temp_root / self._expected_model_dir(part_detail, option, num_episodes)
            run1_dir = target_model_dir / "run1"
            self._prepare_run_dir(source_run, run1_dir, source_part_detail, part_detail)
            for run_id in run_ids[1:]:
                self._ensure_symlink(target_model_dir / run_id, run1_dir)

        add_model("_PT2", "COREDe", self.num_episodes_trained, "PT2_COREDe")
        add_model("_PT3", "COREDe", self.num_episodes_trained, "PT3_COREDe")
        add_model("_PT3", "COREUt", self.num_episodes_trained, "PT3_COREUt")
        add_model("_PT4", "COREDe", self.num_episodes_trained, "PT4_COREDe")

        # PT3after2 placeholders expected by plotting.py; map to PT3 runs
        add_model("_PT3after2", "COREDe", 500, "PT3_COREDe", source_part_detail="_PT3")
        add_model("_PT3after2", "COREUt", 500, "PT3_COREUt", source_part_detail="_PT3")

    def _load_plotting_module(self):
        if self._plotting_module is not None:
            return self._plotting_module
        os.environ.setdefault("MPLBACKEND", "Agg")
        src = self.plotting_path.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            # plotting.py contains scratchpad text near the end; parse only the valid prefix.
            valid_prefix = "\n".join(src.splitlines()[: max(exc.lineno - 1, 0)])
            tree = ast.parse(valid_prefix)
        allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.ClassDef)
        filtered = []
        for node in tree.body:
            if not isinstance(node, allowed):
                continue
            if isinstance(node, ast.Import):
                names = []
                for alias in node.names:
                    if importlib.util.find_spec(alias.name) is not None:
                        names.append(alias)
                if not names:
                    continue
                node.names = names
            if isinstance(node, ast.ImportFrom):
                if node.module and importlib.util.find_spec(node.module) is None:
                    continue
            filtered.append(node)
        tree.body = filtered
        module = types.ModuleType("plotting_slim")
        code = compile(tree, str(self.plotting_path), "exec")
        exec(code, module.__dict__)
        self._plotting_module = module
        return module

    @contextmanager
    def mock_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            self._build_symlink_tree(temp_root)
            cwd = os.getcwd()
            os.chdir(temp_root)
            try:
                yield temp_root
            finally:
                os.chdir(cwd)

    def run_plotting(self, func, output_dir, **kwargs):
        output_dir = Path(output_dir).resolve()
        module = self._load_plotting_module()
        module.SAVE_FIGURES_PATH = str(output_dir)
        module.EVALS_dir = self.eval_dir
        (output_dir / "RESULTS" / self.eval_dir).mkdir(parents=True, exist_ok=True)
        (output_dir / self.eval_dir).mkdir(parents=True, exist_ok=True)

        original_savefig = plt.savefig

        def savefig_compat(*args, **kw):
            # plotting.py sometimes passes unsupported kwargs to savefig.
            kw.pop("tight_layout", None)
            return original_savefig(*args, **kw)

        plt.savefig = savefig_compat
        try:
            with self.mock_directory_structure():
                return func(**kwargs)
        finally:
            plt.savefig = original_savefig
