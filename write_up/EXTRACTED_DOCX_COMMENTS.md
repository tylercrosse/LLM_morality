# Extracted comments from `LLM Morality Mech Interp - ARENA Capstone.docx`

Source: `write_up/LLM Morality Mech Interp - ARENA Capstone.docx`

## Comments

1. Anchor: `zero produced a behavioral flip.`
   Author: Tyler Crosse
   Date: 2026-03-06T23:21:45Z
   Comment: I go back and forth on if it's appropriate to use bold for interesting points.

2. Anchor: `Linear probes.`
   Author: Tyler Crosse
   Date: 2026-03-06T23:24:05Z
   Comment: I think this negative result is correct, although it gives me a nagging feeling that I could have screwed something up in the experiment. In general, I feel like the negative results may need more evidence. 🤷‍♂️

3. Anchor: `We computed Pearson correlations for every pair of the 52 components (26 attention + 26 MLP) across the 15 evaluation prompts, then compared the resulting correlation matrices between the Deontological and Utilitarian models.`
   Author: Tyler Crosse
   Date: 2026-03-06T23:58:51Z
   Comment: Explain why I didn't do a factor analysis of all the models?

4. Anchor: `[no anchored text extracted]`
   Author: Tyler Crosse
   Date: 2026-03-06T23:57:32Z
   Comment: I still don't love this figure. It's not very obvious what it's supposed to show beyond that the correlations are quite noisy.

5. Anchor: `L2_MLP produces a +0.56% cooperation shift, which is effectively zero. L16_MLP and L17_MLP produce +26.2% and +29.6%, respectively. They are 46-52× more effective.`
   Author: Tyler Crosse
   Date: 2026-03-06T23:12:10Z
   Comment: This needs to be tweaked to explain why the comparison with L2 is relevant. It's hinted at in the section below but the connection isn't clear.

6. Anchor: `[no anchored text extracted]`
   Author: Tyler Crosse
   Date: 2026-03-06T23:59:57Z
   Comment: This could maybe just be a table.

7. Anchor: `We tested progressive path patching from the Deontological model into the Strategic model, extending the patch endpoint from L2→L2 through L2→L9.`
   Author: Tyler Crosse
   Date: 2026-03-06T23:26:55Z
   Comment: The emphasis on Layer 2 here is also a bit underdeveloped in the current draft. That was mostly an initial hypothesis that got discarded. TODO do another pass to try to disentangle this/clean it up.

8. Anchor: `Figure 6b: Attention pathways contribute ~3× more causal impact than MLP pathways.`
   Author: Tyler Crosse
   Date: 2026-03-06T23:52:08Z
   Comment: Make another figure of this result that's clearer? The Attention vs. MLP difference seems small relative to the Full Residual

9. Anchor: `Pathway-level interventions produce effects 61.7× larger than any individual component patch, which is consistent with the washout pattern. The effect saturates by L5, suggesting the L2-L5 window contains the primary causal pathway.`
   Author: Tyler Crosse
   Date: 2026-03-06T23:28:55Z
   Comment: Improve this explanation.

10. Anchor: `L2→L9`
    Author: Tyler Crosse
    Date: 2026-03-06T23:53:20Z
    Comment: Same comment about explaining or removing references to Layer 2. This point is no longer supported by the current draft.

11. Anchor: `Increase validation sample counts to tighten rate estimates`
    Author: Tyler Crosse
    Date: 2026-03-06T23:54:13Z
    Comment: This is a bit too vague.

12. Anchor: `L2→L9`
    Author: Tyler Crosse
    Date: 2026-03-06T23:54:42Z
    Comment: Qualify Layer 2 here too.

## Notes

- `write_up/LLM Morality Mech Interp - ARENA Capstone.docx` contains 12 comments.
- `write_up/WRITE_UP_LW_google_docs.docx` contains a `word/comments.xml` part but no actual comment entries.
