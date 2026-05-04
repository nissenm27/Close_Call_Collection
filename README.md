# Close Call Collection

End-to-end semi-automated annotation pipeline for VTTI safety-critical event detection in BDD100K.

## Project Overview

This repository contains an end-to-end pipeline for detecting and semi-automatically annotating safety-critical events (SCEs) in BDD100K videos. The core model is a four-head MMAction2 SlowFast + kinematics fusion pipeline that predicts:

- **Event Type:** `Conflict`, `Bump`, `Hard Brake`, or `Not an SCE`
- **Grouped Conflict Type**
- **17-Way Conflict Type**
- **Predicted Event Start Time:** `BDD_START`

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
```

The first run is meant to skip over steps 3 and 4. Afterwards, we ignore the first two steps and the loop is repeated after manual review so that newly accepted labels can be added back into the training set.

---

## Files Not Stored in GitHub

Large model and video/data artifacts are intentionally not stored in GitHub. These files must be provided separately through OneDrive, local storage, ARC, or another shared storage location.

Required files/folders:

```text
data/

  annotated_videos_only/
    *.mov

  saved_model/
    joint_multitask_four_head_best.pth
```

## Path Configuration Note

Some scripts were originally developed across different machines/environments, so a few paths may need to be checked before running the full pipeline.

The intended repository-relative structure is:

```text
data/
  X_ts.npy
  X_ctx.csv
  meta.csv
  bdd_sce.csv
  annotated_videos_only/
  saved_model/
  unseen_batches/
  results/