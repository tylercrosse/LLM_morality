# Prompt Sensitivity Validation

- Input: `mech_interp_outputs/logit_lens/decision_statistics_by_variant.csv`
- Rows: 75

## Prompt-Variant Stability
- Scenario-model cells with action flips across variants: **1/25**

## Model-Level Sensitivity (seq_p_action2 std)
- PT2_COREDe: 0.0003
- PT3_COREDe: 0.0002
- PT3_COREUt: 0.0344
- PT4_COREDe: 0.4541
- base: 0.1784

## Key Global Pairwise Tests (seq_p_action2)
- PT2_COREDe - PT3_COREDe: diff=0.9994, p=0.000050, CI=[0.9992, 0.9996]
- PT2_COREDe - PT3_COREUt: diff=0.9295, p=0.000050, CI=[0.9128, 0.9461]
- PT2_COREDe - PT4_COREDe: diff=0.5881, p=0.000050, CI=[0.3539, 0.8185]
- PT3_COREDe - PT3_COREUt: diff=-0.0699, p=0.000050, CI=[-0.0866, -0.0534]
- PT3_COREUt - PT4_COREDe: diff=-0.3414, p=0.014049, CI=[-0.5732, -0.1137]