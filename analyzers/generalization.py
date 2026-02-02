from pathlib import Path
import shutil


def analyze_cross_game_generalization(adapter, output_dir):
    """Use original plotting style, but only 5 real conditions."""
    output_dir = Path(output_dir).resolve()
    module = adapter._load_plotting_module()

    def _plot_5model_generalization():
        import matplotlib.pyplot as plt
        import pandas as pd

        games = ["IPD", "ISH", "ICN", "BOS", "ICD"]
        iterables = [
            [
                "No fine-tuning",
                "Game payoffs",
                "Deontological",
                "Utilitarian",
                "Game + \nDeontological",
            ],
            ["C | C", "C | D", "D | C", "D | D", "illegal | C", "illegal | D"],
        ]
        multiindex = pd.MultiIndex.from_product(iterables, names=["PART", "Action Types"])
        all_runs = pd.DataFrame(index=games, columns=multiindex)

        num_episodes_eval = 10

        for game in games:
            allruns_singlepart = module.process_actions_eval(
                "_PT2",
                adapter.opponent,
                1000,
                "COREDe",
                game,
                "",
                "action1",
                "action2",
                num_episodes_eval,
                before_or_after="before",
            )
            all_runs.loc[game]["No fine-tuning"] = allruns_singlepart.mean(axis=1).values

        for game in games:
            all_runs.loc[game]["Game payoffs"] = module.process_actions_eval(
                "_PT2",
                adapter.opponent,
                1000,
                "COREDe",
                game,
                "",
                "action1",
                "action2",
                num_episodes_eval,
                before_or_after="after",
            ).mean(axis=1).values
            all_runs.loc[game]["Deontological"] = module.process_actions_eval(
                "_PT3",
                adapter.opponent,
                1000,
                "COREDe",
                game,
                "",
                "action1",
                "action2",
                num_episodes_eval,
                before_or_after="after",
            ).mean(axis=1).values
            all_runs.loc[game]["Utilitarian"] = module.process_actions_eval(
                "_PT3",
                adapter.opponent,
                1000,
                "COREUt",
                game,
                "",
                "action1",
                "action2",
                num_episodes_eval,
                before_or_after="after",
            ).mean(axis=1).values
            all_runs.loc[game]["Game + \nDeontological"] = module.process_actions_eval(
                "_PT4",
                adapter.opponent,
                1000,
                "COREDe",
                game,
                "",
                "action1",
                "action2",
                num_episodes_eval,
                before_or_after="after",
            ).mean(axis=1).values

        all_runs.index = [
            "Iterated Prisoner's Dilemma",
            "Iterated Stag Hunt",
            "Iterated Chicken",
            "Iterated Bach-or-Stravinsky",
            "Iterated Defective Coordination",
        ]

        fig, axs = plt.subplots(1, 5, figsize=(48, 7), sharey=True)
        fontsize = 70
        colors = [
            "#28641E",
            "#B0DC82",
            "#FBE6F1",
            "#8E0B52",
            "#A9A9A9",
            "#A9A9A9",
            "#A9A9A9",
            "#A9A9A9",
            "#A9A9A9",
        ]

        all_runs["No fine-tuning"].plot(kind="bar", stacked=True, ax=axs[0], legend=False, color=colors, fontsize=fontsize)
        axs[0].set_title("No fine-tuning\n", fontsize=fontsize)

        all_runs["Game payoffs"].plot(kind="bar", stacked=True, ax=axs[1], legend=False, color=colors, fontsize=fontsize)
        axs[1].set_title("Game \npayoffs", fontsize=fontsize)

        all_runs["Deontological"].plot(kind="bar", stacked=True, ax=axs[2], legend=False, color=colors, fontsize=fontsize)
        axs[2].set_title("Deontological\n", fontsize=fontsize)

        all_runs["Utilitarian"].plot(kind="bar", stacked=True, ax=axs[3], legend=False, color=colors, fontsize=fontsize)
        axs[3].set_title("Utilitarian\n", fontsize=fontsize)

        all_runs["Game + \nDeontological"].plot(kind="bar", stacked=True, ax=axs[4], legend=False, color=colors, fontsize=fontsize)
        axs[4].set_title("Game + \nDeontological", fontsize=fontsize)

        plt.legend(bbox_to_anchor=(1.55, 0.5), loc="center", fontsize=fontsize)
        axs[0].set_ylabel("M's action | O's prev. move \n (% of test time responses)", fontsize=fontsize)
        axs[2].set_xlabel("Iterated Game", fontsize=fontsize)
        plt.suptitle(
            f"Action choices on five iterated matrix games \n (all models trained vs {adapter.opponent} opponent)",
            fontsize=fontsize + 10,
            y=1.49,
        )

        plt.tight_layout()
        plt.savefig(
            f"{module.SAVE_FIGURES_PATH}/RESULTS/{module.EVALS_dir}/othergames_actiontypes_opp{adapter.opponent}_5models.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()

    adapter.run_plotting(_plot_5model_generalization, output_dir=output_dir)

    src = output_dir / "RESULTS" / adapter.eval_dir / f"othergames_actiontypes_opp{adapter.opponent}_5models.pdf"
    dst = output_dir / "cross_game_generalization_publication.pdf"
    if src.exists():
        shutil.copy2(src, dst)
    return dst
