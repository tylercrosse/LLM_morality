#!/bin/bash -l

mkdir models
modal volume get llm-morality-models gemma-2-2b-it models/
modal volume get llm-morality-models gemma-2-2b-it_FT_PT2_oppTFT_run1_1000ep_COREDe models/
modal volume get llm-morality-models gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREDe models/
modal volume get llm-morality-models gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREUt models/
modal volume get llm-morality-models gemma-2-2b-it_FT_PT4_oppTFT_run1_1000ep_COREDe models/