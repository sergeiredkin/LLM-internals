---
type: concept
---
# Sampling
**One line:** all knobs (temp/top-k/top-p/rep-penalty) reshape one logits vector pre-softmax.
**The click:** temperature → entropy; top-p set size varies per step; greedy = loops.
**Learned in:** [[07-sampling]]
**In the GPT:** generation; full suite incl. top-p + rep penalty.
