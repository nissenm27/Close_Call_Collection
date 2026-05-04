# Close Call Collection

End-to-end semi-automated annotation pipeline for VTTI safety-critical event detection in BDD100K.

## Project Overview

This repository contains an end-to-end pipeline for detecting and semi-automatically annotating safety-critical events (SCEs) in BDD100K videos. The core model is a four-head MMAction2 SlowFast + kinematics fusion pipeline that predicts:

$$
\text{Event Type} \in \{\text{Conflict},\ \text{Bump},\ \text{Hard Brake},\ \text{Not an SCE}\}
$$

$$
\text{Grouped Conflict Type}, \quad \text{17-Way Conflict Type}, \quad \widehat{\text{BDD\_START}}
$$

The intended workflow is:

```text
BDD100K_batch_auto_download.py
        ↓
mmaction2_joint_multitask_final.py
        ↓
pull_unseen_batch.py
        ↓
deploy_unseen_multitask.py
        ↓
AutoLabeler.py
        ↓
manual_review.py
        ↓
mmaction2_joint_multitask_final.py


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