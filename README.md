# Close Call Collection

End-to-end semi-automated annotation pipeline for VTTI safety-critical event detection in BDD100K.

## Project Overview

This repository contains an end-to-end pipeline for detecting and semi-automatically annotating safety-critical events (SCEs) in BDD100K videos. The core model is a four-head MMAction2 SlowFast + kinematics fusion pipeline that predicts:

\[
\text{Event Type} \in \{\text{Conflict}, \text{Bump}, \text{Hard Brake}, \text{Not an SCE}\}
\]

\[
\text{Grouped Conflict Type}, \quad \text{17-Way Conflict Type}, \quad \widehat{\text{BDD\_START}}
\]

The intended workflow is:

\[
\texttt{BDD100K\_batch\_auto\_download.py}
\rightarrow
\texttt{mmaction2\_joint\_multitask\_final.py}
\rightarrow
\texttt{pull\_unseen\_batch.py}
\rightarrow
\texttt{deploy\_unseen\_multitask.py}
\rightarrow
\texttt{AutoLabeler.py}
\rightarrow
\texttt{manual\_review.py}
\rightarrow
\texttt{mmaction2\_joint\_multitask\_final.py}
\]

The loop is repeated after manual review so that newly accepted labels can be added back into the training set.

---

## Files Not Stored in GitHub

Large model and video/data artifacts are intentionally not stored in GitHub. These files must be provided separately through OneDrive, local storage, ARC, or another shared storage location.

Required files/folders:

```text
data/
  X_ts.npy
  X_ctx.csv
  meta.csv
  bdd_sce.csv

  annotated_videos_only/
    *.mov

  saved_model/
    joint_multitask_four_head_best.pth
    joint_splits.csv