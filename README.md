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
audit_interface.py
        ↓
mmaction2_joint_multitask_final.py
```

On the initial training run, the unseen-batch pull and deployment steps are skipped because no feedback-loop batch has been generated yet. After the first trained model is available, the repeated loop begins at `pull_unseen_batch.py`, followed by deployment, auto-labeling, audit review, and retraining. At this point, the first two setup steps usually do not need to be repeated unless the base annotated-video set or training data is being rebuilt.

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

The intended shared structure is:

```text
data/
  X_ts.npy
  X_ctx.csv
  meta.csv
  bdd_sce.csv
  downloaded_videos_meta.csv

  annotated_videos_only/
    *.mov

  saved_model/
    joint_multitask_four_head_best.pth
    joint_splits.csv

  unseen_batches/
    batch_020/
      videos/
        *.mov
      batch_020_unseen_manifest.csv

  results/
    batch_020/
      batch_020_inference_manifest.csv
      batch_020_model_predictions.csv
      bdd_sce_output.csv
      audit_list.csv
      sponsor_report_full.csv
```

## Script-by-Script Path Notes

Before running the pipeline, confirm that the scripts use repository-relative paths instead of user-specific absolute paths. The intended convention is:

- `data/saved_model/` stores reusable model artifacts.
- `data/unseen_batches/batch_*/` stores raw pulled unseen videos.
- `data/results/batch_*/` stores deployment outputs, auto-label outputs, audit files, and sponsor-facing CSV outputs.

---

### `BDD100K_batch_auto_download.py`

This script rebuilds the original annotated-video collection by downloading BDD100K train batches and keeping only videos whose filename stem matches a labeled `BDD_ID`.

**Current path behavior to check:**

- The script may still use a user-specific absolute `DATA_DIR`.
- The script may still start at `START_BATCH = 16`.

**Recommended repo-relative configuration:**

```python
DATA_DIR = Path("data")
SAVED_FOLDER = DATA_DIR / "annotated_videos_only"
ORIGINAL_ANNOTATIONS_CSV = DATA_DIR / "bdd_sce.csv"
OPTIONAL_SOURCE_META_CSV = DATA_DIR / "meta.csv"
OUTPUT_META_CSV = DATA_DIR / "downloaded_videos_meta.csv"
```

If rebuilding the full annotated-video folder from scratch, set:

```python
START_BATCH = 0
```

If the annotated videos are already available through OneDrive or another shared folder, this script does not need to be rerun. In that case, copy the provided videos directly into:

```text
data/annotated_videos_only/
```

---

### `mmaction2_joint_multitask_final.py`

This is the main four-head training and evaluation script. It trains the MMAction2 SlowFast + kinematics fusion model with:

- event type
- grouped conflict type
- 17-way conflict type
- predicted `BDD_START`

**Required repo-relative inputs:**

```text
data/X_ts.npy
data/meta.csv
data/X_ctx.csv
data/bdd_sce.csv
data/annotated_videos_only/
```

**Current path behavior to check:**

- The script currently uses repo-relative data inputs, which is good.
- The script still saves training outputs under the experiment folder:

```text
data/mmaction2_slowfast_joint_multitask_four_head/
```

- The script also depends on local MMAction2 config/checkpoint paths:

```python
MMACTION_CONFIG_PATH
MMACTION_CHECKPOINT_PATH
```

After training, copy or move the deployable artifacts into:

```text
data/saved_model/joint_multitask_four_head_best.pth
data/saved_model/joint_splits.csv
```

The `.pth` checkpoint should not be committed to GitHub because it is too large for normal Git tracking.

---

### `pull_unseen_batch.py`

This script pulls raw unseen BDD100K videos for the next feedback-loop stage. It compares BDD video IDs against already-seen IDs from metadata files and `data/annotated_videos_only/`.

**Default behavior:**

- Pulls batch 20 unless changed.
- Writes unseen videos to:

```text
data/unseen_batches/batch_020/videos/
```

- Writes its own raw unseen manifest to:

```text
data/unseen_batches/batch_020/batch_020_unseen_manifest.csv
```

**Recommended usage:**

First run:

```bash
python pull_unseen_batch.py --dry-run
```

Then real run:

```bash
python pull_unseen_batch.py
```

If using a different batch `k`, update the batch number consistently so the folder follows:

```text
data/unseen_batches/batch_00k/videos/
```

This script owns raw unseen video collection. It does not run the model.

---

### `deploy_unseen_multitask.py`

This script loads the saved four-head checkpoint and runs inference on the unseen batch.

**Expected model artifacts:**

```text
data/saved_model/joint_multitask_four_head_best.pth
data/saved_model/joint_splits.csv
```

**Expected unseen video input:**

```text
data/unseen_batches/batch_020/videos/
```

**Expected deployment outputs:**

```text
data/results/batch_020/batch_020_inference_manifest.csv
data/results/batch_020/batch_020_model_predictions.csv
```

The deployment script creates its own inference manifest in the results directory. This manifest is just a CSV list of videos that will be scored by the model.

**Current path behavior to check:**

```python
BATCH_NUMBER = 20
SAVED_MODEL_DIR = Path("data/saved_model")
UNSEEN_VIDEO_DIR = Path(f"data/unseen_batches/batch_{BATCH_NUMBER:03d}/videos")
RESULTS_DIR = Path(f"data/results/batch_{BATCH_NUMBER:03d}")
```

If deploying a different batch `k`, change:

```python
BATCH_NUMBER = k
```

**Important limitation:**

The deployed model is not video-only. It requires matching kinematics and context rows from:

```text
data/meta.csv
data/X_ts.npy
data/X_ctx.csv
```

If a pulled unseen video does not have a matching `BDD_ID` in `data/meta.csv`, deployment will mark it as missing kinematics/context rather than scoring it.

---

### `AutoLabeler.py`

This script converts model predictions into annotation-style CSV outputs and separates high-confidence labels from clips needing audit review.

**Current path behavior to check:**

- The uploaded version still uses user-specific Windows paths.
- Replace those paths with repo-relative batch paths.

**Recommended batch-aware paths for batch 20:**

```python
INPUT_FILE = "data/results/batch_020/batch_020_model_predictions.csv"
OUTPUT_DIR = "data/results/batch_020"
MODEL_PATH = "data/results/batch_020/rf_autolabeler.pkl"
```

**Important column-name note:**

The current `AutoLabeler.py` was originally written for the held-out test prediction file from training. That file used columns like:

```text
Conflict
Bump
Hard Brake
Not SCE
conf17_Q
conf17_W
start_pred
```

The deployment script currently writes columns like:

```text
PROB_CONFLICT
PROB_BUMP
PROB_HARD_BRAKE
PROB_NOT_SCE
PROB_CONFLICT_17WAY_Q
PROB_CONFLICT_17WAY_W
MODEL_PRED_START_TIME_SEC
```

Therefore, before using `AutoLabeler.py` on deployed unseen-batch predictions, update the feature column names so they match `batch_020_model_predictions.csv`.

**Recommended deployment-compatible mapping:**

```text
Conflict      -> PROB_CONFLICT
Bump          -> PROB_BUMP
Hard Brake    -> PROB_HARD_BRAKE
Not SCE       -> PROB_NOT_SCE
conf17_Q      -> PROB_CONFLICT_17WAY_Q
conf17_W      -> PROB_CONFLICT_17WAY_W
start_pred    -> MODEL_PRED_START_TIME_SEC
```

Continue the same conflict-column pattern for all 17 conflict letters.

**Training mode note:**

- If running on unlabeled deployed predictions, set `TRAIN_MODE = False`.
- If `TRAIN_MODE = True`, the input CSV must contain true label columns such as `event_true_label`.

**Expected auto-labeler outputs:**

```text
data/results/batch_020/bdd_sce_output.csv
data/results/batch_020/audit_list.csv
data/results/batch_020/accuracy_diagnostics.csv
```

`accuracy_diagnostics.csv` is only produced if labels are available.

---

### `audit_interface.py`

This script provides the manual review interface between auto-labeling and retraining.

**Current path behavior to check:**

- The GUI defaults to files in `~/Downloads`.
- The setup screen lets the user browse for the video folder, audit CSV, and complete CSV.
- Because the file picker is interactive, this script does not need to be fully hard-coded, but the README should tell users which files to select.

**Recommended manual selections:**

Video folder:

```text
data/unseen_batches/batch_020/videos/
```

Audit CSV:

```text
data/results/batch_020/audit_list.csv
```

Complete CSV:

```text
data/results/batch_020/sponsor_report_full.csv
```

**Important column-name note:**

The current audit interface expects a prediction-name column such as:

```text
MODEL_PREDICTION_NAME
```

If the audit file comes from the current auto-labeler or deployment output, confirm that the column names match what `audit_interface.py` expects. If not, either rename the column in the audit CSV or update the script to read:

```text
MODEL_PRED_EVENT_TYPE
```

Manual review output should preserve or produce the key annotation fields:

```text
EVENT_ID
BDD_ID
EVENT_TYPE
CONFLICT_TYPE
BDD_START
```

After audit review, merge the reviewed rows with the accepted auto-label rows before adding them back into:

```text
data/bdd_sce.csv
```

---

### `EDA.R`

This script is not part of the required training, deployment, or feedback-loop pipeline.

Use it only for exploratory analysis. It can be run independently at any point, provided its file paths are updated to match the local repository structure.

---

### Batch Number Consistency

For any feedback-loop batch `k`, make sure the same batch number is used across:

```text
pull_unseen_batch.py
deploy_unseen_multitask.py
AutoLabeler.py
audit_interface.py
```

For example, batch 20 should consistently use:

```text
batch_020
```

The recommended result structure for each batch is:

```text
data/results/batch_020/
data/results/batch_021/
data/results/batch_022/
```

and so on.

---

### Feedback-Loop Reminder

After manual review, accepted automatic labels and corrected audit labels should be merged into the project annotation file:

```text
data/bdd_sce.csv
```

Any newly accepted videos that will be used for retraining should also be available in:

```text
data/annotated_videos_only/
```

Then rerun:

```bash
python mmaction2_joint_multitask_final.py
```

This closes the loop:

```text
pull unseen videos
        ↓
deploy model
        ↓
auto-label
        ↓
manual audit
        ↓
update labels
        ↓
retrain
```