#!/usr/bin/env python3
"""
Deploy the final four-head MMAction2 SlowFast + kinematics model on unseen BDD100K videos.

Hard-coded project layout
-------------------------
Run from the project/repo root:

    python deploy_unseen_multitask.py

Expected files:
    mmaction2_joint_multitask_final.py
    data/saved_model/joint_multitask_four_head_best.pth
    data/saved_model/joint_splits.csv
    data/X_ts.npy
    data/meta.csv
    data/X_ctx.csv
    data/unseen_batches/batch_020/videos/*.mov

Outputs created:
    data/results/batch_020/batch_020_inference_manifest.csv
    data/results/batch_020/batch_020_model_predictions.csv

What is a manifest?
-------------------
A manifest is just a CSV list of videos to run inference on. It usually contains:
    BDD_ID, VIDEO_PATH, status columns, and later prediction columns.

This script creates the manifest automatically from the unseen video folder if it
does not already exist in data/results/batch_020/.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# HARD-CODED PATHS
# ============================================================
# Change this one number if you pull a different unseen BDD batch.
BATCH_NUMBER = 20

# Main final training/evaluation pipeline file.
PIPELINE_FILE = Path("mmaction2_joint_multitask_final.py")

# Model artifacts you said you want to store here.
SAVED_MODEL_DIR = Path("data/saved_model")
CHECKPOINT_PATH = SAVED_MODEL_DIR / "joint_multitask_four_head_best.pth"
SPLITS_PATH = SAVED_MODEL_DIR / "joint_splits.csv"

# Unseen videos created by pull_unseen_batch.py.
# The pull script still owns raw video collection; this script owns inference/results.
UNSEEN_VIDEO_DIR = Path(f"data/unseen_batches/batch_{BATCH_NUMBER:03d}/videos")

# Batch-specific output directory.
RESULTS_DIR = Path(f"data/results/batch_{BATCH_NUMBER:03d}")
MANIFEST_PATH = RESULTS_DIR / f"batch_{BATCH_NUMBER:03d}_inference_manifest.csv"
OUTPUT_PATH = RESULTS_DIR / f"batch_{BATCH_NUMBER:03d}_model_predictions.csv"

# Inference settings.
BATCH_SIZE = 4
EVENT_CONFIDENCE_AUDIT_THRESHOLD = 0.80
CONFLICT_HEAD_AUDIT_THRESHOLD = 0.50
VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".MOV", ".MP4", ".MKV"}

# IMPORTANT:
# For truly unseen videos, we usually do not know the real BDD_START yet.
# The model predicts BDD_START, but the video sampler still needs a starting point.
# Defaulting to 0.0 means the video branch samples from the start of the clip.
DEFAULT_BDD_START_FOR_SAMPLING = 0.0


# ============================================================
# BASIC UTILS
# ============================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def get_visible_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


INFERENCE_NUM_WORKERS = min(12, max(get_visible_cpu_count() - 2, 0))
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()


def load_pipeline_module():
    """Load mmaction2_joint_multitask_final.py as a module without running its main()."""
    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {PIPELINE_FILE}. Run this script from the repo/project root "
            "or update PIPELINE_FILE at the top of this script."
        )

    spec = importlib.util.spec_from_file_location("final_pipeline", PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pipeline file: {PIPELINE_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["final_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# ============================================================
# MANIFEST CREATION
# ============================================================
def list_unseen_videos(video_dir: Path) -> List[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(
            f"Unseen video folder does not exist: {video_dir}\n"
            "Run pull_unseen_batch.py first, or update UNSEEN_VIDEO_DIR at the top of this script."
        )

    videos = [
        p for p in video_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def build_manifest_from_videos(video_dir: Path, manifest_path: Path) -> pd.DataFrame:
    videos = list_unseen_videos(video_dir)
    if len(videos) == 0:
        raise RuntimeError(f"No video files found in {video_dir}")

    rows = []
    for video_path in videos:
        rows.append({
            "BATCH": BATCH_NUMBER,
            "BDD_ID": video_path.stem,
            "SOURCE_FILENAME": video_path.name,
            "VIDEO_PATH": str(video_path),
            "BDD_START_FOR_SAMPLING": DEFAULT_BDD_START_FOR_SAMPLING,
            "AUTO_LABEL_STATUS": "READY_FOR_MODEL_INFERENCE",
            "NOTES": "",
        })

    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    log(f"Created inference manifest: {manifest_path}")
    log(f"Manifest rows: {len(df)}")
    return df


def resolve_video_path(bdd_id: str) -> Optional[Path]:
    for ext in [".mov", ".mp4", ".mkv", ".MOV", ".MP4", ".MKV"]:
        candidate = UNSEEN_VIDEO_DIR / f"{bdd_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_or_create_manifest() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists():
        log(f"Using existing inference manifest: {MANIFEST_PATH}")
        df = pd.read_csv(MANIFEST_PATH)
    else:
        df = build_manifest_from_videos(UNSEEN_VIDEO_DIR, MANIFEST_PATH)

    if "BDD_ID" not in df.columns:
        raise ValueError(f"Manifest must contain BDD_ID column: {MANIFEST_PATH}")

    if "VIDEO_PATH" not in df.columns:
        # Recover from BDD_ID only by assuming files are in UNSEEN_VIDEO_DIR.
        df["VIDEO_PATH"] = df["BDD_ID"].astype(str).map(lambda x: str(resolve_video_path(x)))

    if "BDD_START_FOR_SAMPLING" not in df.columns:
        df["BDD_START_FOR_SAMPLING"] = DEFAULT_BDD_START_FOR_SAMPLING

    df["BDD_ID"] = df["BDD_ID"].astype(str)
    df["VIDEO_PATH"] = df["VIDEO_PATH"].astype(str)
    df["BDD_START_FOR_SAMPLING"] = df["BDD_START_FOR_SAMPLING"].map(
        lambda x: safe_float(x, DEFAULT_BDD_START_FOR_SAMPLING)
    )

    return df


# ============================================================
# KINEMATICS / CONTEXT ALIGNMENT
# ============================================================
def make_context_matrix(ctx_df: pd.DataFrame) -> np.ndarray:
    """Match the context preprocessing from the final training script."""
    ctx_features = ctx_df.drop(
        columns=[c for c in ["BDD_ID", "EVENT_ID", "EVENT_TYPE", "y"] if c in ctx_df.columns]
    )
    categorical_cols = [c for c in ["weather", "scene", "timeofday"] if c in ctx_features.columns]
    ctx_features = pd.get_dummies(ctx_features, columns=categorical_cols, dummy_na=True)
    ctx_features = ctx_features.fillna(0)
    return ctx_features.to_numpy(dtype=np.float32)


def get_train_row_idx(meta_df: pd.DataFrame, splits_df: pd.DataFrame) -> np.ndarray:
    if "row_idx" in splits_df.columns:
        return splits_df.loc[splits_df["split"] == "train", "row_idx"].to_numpy(dtype=int)

    if "BDD_ID" in splits_df.columns:
        meta_idx = meta_df.copy()
        meta_idx["BDD_ID"] = meta_idx["BDD_ID"].astype(str)
        bdd_to_row = dict(zip(meta_idx["BDD_ID"], np.arange(len(meta_idx))))
        train_ids = splits_df.loc[splits_df["split"] == "train", "BDD_ID"].astype(str)
        return np.array([bdd_to_row[x] for x in train_ids if x in bdd_to_row], dtype=int)

    raise ValueError("Saved splits file must contain either row_idx or BDD_ID.")


def fit_transform_timeseries_from_saved_split(
    X_ts_all: np.ndarray,
    meta_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> np.ndarray:
    """Use the original training rows from saved joint_splits.csv to fit train-only scaling."""
    train_row_idx = get_train_row_idx(meta_df, splits_df)

    if len(train_row_idx) == 0:
        raise RuntimeError("No training rows found in saved splits file. Cannot fit scalers.")

    if X_ts_all.ndim != 3:
        raise ValueError(f"Expected X_ts shape [N, C, T], got {X_ts_all.shape}")

    n, c, t = X_ts_all.shape
    x_train = X_ts_all[train_row_idx].transpose(0, 2, 1).reshape(-1, c)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_all = X_ts_all.transpose(0, 2, 1).reshape(-1, c)
    x_all = scaler.transform(x_all)
    return x_all.reshape(n, t, c).transpose(0, 2, 1).astype(np.float32)


def fit_transform_context_from_saved_split(
    X_ctx_all: np.ndarray,
    meta_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> np.ndarray:
    """Use the original training rows from saved joint_splits.csv to fit train-only scaling."""
    train_row_idx = get_train_row_idx(meta_df, splits_df)

    if len(train_row_idx) == 0:
        raise RuntimeError("No training rows found in saved splits file. Cannot fit context scaler.")

    scaler = StandardScaler()
    scaler.fit(X_ctx_all[train_row_idx])

    return scaler.transform(X_ctx_all).astype(np.float32)


def attach_model_inputs(
    manifest_df: pd.DataFrame,
    pipeline,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Attach row_idx, normalized X_ts, and normalized X_ctx for videos whose BDD_ID exists
    in the kinematics/context tables.
    """
    require_file(pipeline.KINE_META_PATH, "kinematics metadata CSV")
    require_file(pipeline.KINE_X_PATH, "kinematics time-series array")
    require_file(pipeline.KINE_CTX_PATH, "context CSV")
    require_file(SPLITS_PATH, "saved split CSV")

    meta_df = pd.read_csv(pipeline.KINE_META_PATH).copy()
    ctx_df = pd.read_csv(pipeline.KINE_CTX_PATH).copy()
    splits_df = pd.read_csv(SPLITS_PATH).copy()
    X_ts_all = np.load(pipeline.KINE_X_PATH).astype(np.float32)

    if "BDD_ID" not in meta_df.columns:
        raise ValueError(f"{pipeline.KINE_META_PATH} must contain BDD_ID.")

    if len(meta_df) != len(ctx_df) or len(meta_df) != len(X_ts_all):
        raise ValueError(
            "Mismatch among meta.csv, X_ctx.csv, and X_ts.npy lengths. "
            f"meta={len(meta_df)}, ctx={len(ctx_df)}, X_ts={len(X_ts_all)}"
        )

    meta_df["BDD_ID"] = meta_df["BDD_ID"].astype(str)
    bdd_to_row = dict(zip(meta_df["BDD_ID"], np.arange(len(meta_df))))

    X_ctx_all = make_context_matrix(ctx_df)
    X_ts_norm_all = fit_transform_timeseries_from_saved_split(X_ts_all, meta_df, splits_df)
    X_ctx_norm_all = fit_transform_context_from_saved_split(X_ctx_all, meta_df, splits_df)

    df = manifest_df.copy()
    df["row_idx"] = df["BDD_ID"].map(bdd_to_row)
    df["VIDEO_EXISTS"] = df["VIDEO_PATH"].map(lambda x: Path(str(x)).exists())
    df["HAS_KINEMATICS_CONTEXT"] = df["row_idx"].notna()

    usable_mask = df["VIDEO_EXISTS"] & df["HAS_KINEMATICS_CONTEXT"]
    usable_df = df[usable_mask].copy().reset_index(drop=True)

    missing_video = int((~df["VIDEO_EXISTS"]).sum())
    missing_inputs = int((~df["HAS_KINEMATICS_CONTEXT"]).sum())

    if missing_video:
        log(f"[WARN] {missing_video} manifest rows have missing video files.")
    if missing_inputs:
        log(f"[WARN] {missing_inputs} rows have no matching kinematics/context row in data/meta.csv.")

    if len(usable_df) == 0:
        return (
            df,
            np.zeros((0,) + X_ts_norm_all.shape[1:], dtype=np.float32),
            np.zeros((0, X_ctx_norm_all.shape[1]), dtype=np.float32),
        )

    row_idx = usable_df["row_idx"].astype(int).to_numpy()
    X_ts_sel = X_ts_norm_all[row_idx]
    X_ctx_sel = X_ctx_norm_all[row_idx]

    # Preserve the usable ordering for downstream dataset creation.
    usable_df["_usable_order"] = np.arange(len(usable_df))
    df = df.merge(
        usable_df[["BDD_ID", "_usable_order"]],
        on="BDD_ID",
        how="left",
    )

    return df, X_ts_sel, X_ctx_sel


# ============================================================
# DATASET / DATALOADER
# ============================================================
class UnseenInferenceDataset(Dataset):
    def __init__(self, usable_df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray, video_loader) -> None:
        self.df = usable_df.reset_index(drop=True)
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.video_loader = video_loader

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = str(row["VIDEO_PATH"])
        bdd_start = safe_float(
            row.get("BDD_START_FOR_SAMPLING", DEFAULT_BDD_START_FOR_SAMPLING),
            DEFAULT_BDD_START_FOR_SAMPLING,
        )

        return (
            self.video_loader(video_path, bdd_start),
            self.X_ts[idx],
            self.X_ctx[idx],
            str(row["BDD_ID"]),
            video_path,
            float(bdd_start),
        )


def unseen_collate_fn(batch):
    videos, xs_ts, xs_ctx, ids, paths, starts = zip(*batch)
    return (
        torch.stack(videos, dim=0),
        torch.stack(xs_ts, dim=0),
        torch.stack(xs_ctx, dim=0),
        list(ids),
        list(paths),
        torch.tensor(starts, dtype=torch.float32),
    )


# ============================================================
# MODEL LOADING / INFERENCE
# ============================================================
def build_model(pipeline, in_chans_ts: int, in_chans_ctx: int):
    require_file(CHECKPOINT_PATH, "trained four-head checkpoint")
    require_file(pipeline.MMACTION_CONFIG_PATH, "MMAction2 SlowFast config")
    require_file(pipeline.MMACTION_CHECKPOINT_PATH, "Kinetics-400 SlowFast checkpoint")

    video_encoder = pipeline.MMAction2SlowFastFeatureExtractor(
        config_path=pipeline.MMACTION_CONFIG_PATH,
        checkpoint_path=pipeline.MMACTION_CHECKPOINT_PATH,
        freeze_backbone=False,
    )

    model = pipeline.JointMultiTaskFourHeadModel(
        video_encoder=video_encoder,
        in_chans_ts=in_chans_ts,
        in_chans_ctx=in_chans_ctx,
    ).to(pipeline.DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=pipeline.DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # This should be strict. If it fails, the checkpoint does not match the uploaded final pipeline.
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def decide_audit_flag(
    event_label: str,
    event_conf: float,
    conflict_group_conf: float,
    conflict17_conf: float,
) -> Tuple[bool, str]:
    reasons = []

    if event_conf < EVENT_CONFIDENCE_AUDIT_THRESHOLD:
        reasons.append(f"event_conf<{EVENT_CONFIDENCE_AUDIT_THRESHOLD:.2f}")

    if event_label == "Conflict":
        if conflict_group_conf < CONFLICT_HEAD_AUDIT_THRESHOLD:
            reasons.append(f"conflict_group_conf<{CONFLICT_HEAD_AUDIT_THRESHOLD:.2f}")
        if conflict17_conf < CONFLICT_HEAD_AUDIT_THRESHOLD:
            reasons.append(f"conflict17_conf<{CONFLICT_HEAD_AUDIT_THRESHOLD:.2f}")

    if reasons:
        return True, "; ".join(reasons)

    return False, ""


def run_inference(pipeline, usable_df: pd.DataFrame, X_ts_sel: np.ndarray, X_ctx_sel: np.ndarray) -> pd.DataFrame:
    if len(usable_df) == 0:
        return pd.DataFrame()

    video_loader = pipeline.SimpleSlowFastVideoLoader()

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=unseen_collate_fn,
        persistent_workers=INFERENCE_NUM_WORKERS > 0,
    )
    if INFERENCE_NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2

    dataset = UnseenInferenceDataset(usable_df, X_ts_sel, X_ctx_sel, video_loader)
    loader = DataLoader(dataset, **loader_kwargs)

    model = build_model(
        pipeline=pipeline,
        in_chans_ts=X_ts_sel.shape[1],
        in_chans_ctx=X_ctx_sel.shape[1],
    )

    rows: List[Dict[str, object]] = []
    start_time = time.perf_counter()

    log("\n" + "=" * 72)
    log("Running four-head model inference")
    log(f"Device: {pipeline.DEVICE}")
    log(f"Videos to score: {len(dataset)}")
    log(f"Batch size: {BATCH_SIZE}")
    log(f"DataLoader workers: {INFERENCE_NUM_WORKERS}")
    log("=" * 72 + "\n")

    with torch.no_grad():
        for batch_v, batch_ts, batch_ctx, ids, paths, starts in tqdm(loader, desc="Inference"):
            batch_v = batch_v.to(pipeline.DEVICE, non_blocking=PIN_MEMORY)
            batch_ts = batch_ts.to(pipeline.DEVICE, non_blocking=PIN_MEMORY)
            batch_ctx = batch_ctx.to(pipeline.DEVICE, non_blocking=PIN_MEMORY)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                outputs = model(batch_v, batch_ts, batch_ctx)
                event_probs = torch.softmax(outputs["event_logits"], dim=1)
                group_probs = torch.softmax(outputs["conflict_group_logits"], dim=1)
                c17_probs = torch.softmax(outputs["conflict17_logits"], dim=1)

            event_probs_np = event_probs.detach().cpu().numpy()
            group_probs_np = group_probs.detach().cpu().numpy()
            c17_probs_np = c17_probs.detach().cpu().numpy()
            start_pred_np = outputs["start_pred"].detach().cpu().numpy()

            event_pred_idx = np.argmax(event_probs_np, axis=1)
            group_pred_idx = np.argmax(group_probs_np, axis=1)
            c17_pred_idx = np.argmax(c17_probs_np, axis=1)

            for i, bdd_id in enumerate(ids):
                event_idx = int(event_pred_idx[i])
                group_idx = int(group_pred_idx[i])
                c17_idx = int(c17_pred_idx[i])

                event_label = pipeline.CLASS_MAP_STR[event_idx]
                event_conf = float(event_probs_np[i, event_idx])
                group_conf = float(group_probs_np[i, group_idx])
                c17_conf = float(c17_probs_np[i, c17_idx])

                pred_group = pipeline.IDX_TO_GROUP[group_idx] if event_label == "Conflict" else ""
                pred_c17 = pipeline.IDX_TO_CONFLICT17[c17_idx] if event_label == "Conflict" else ""

                audit_required, audit_reason = decide_audit_flag(
                    event_label=event_label,
                    event_conf=event_conf,
                    conflict_group_conf=group_conf,
                    conflict17_conf=c17_conf,
                )

                row = {
                    "BATCH": BATCH_NUMBER,
                    "BDD_ID": bdd_id,
                    "VIDEO_PATH": paths[i],
                    "INFERENCE_STATUS": "MODEL_SCORED",
                    "BDD_START_FOR_SAMPLING": float(starts[i].item()),
                    "MODEL_PRED_EVENT_TYPE": event_label,
                    "MODEL_PRED_EVENT_IDX": event_idx,
                    "MODEL_CONFIDENCE": event_conf,
                    "MODEL_PRED_CONFLICT_GROUP": pred_group,
                    "MODEL_PRED_CONFLICT_GROUP_CONFIDENCE": group_conf if event_label == "Conflict" else "",
                    "MODEL_PRED_CONFLICT_17WAY": pred_c17,
                    "MODEL_PRED_CONFLICT_17WAY_CONFIDENCE": c17_conf if event_label == "Conflict" else "",
                    "MODEL_PRED_START_TIME_SEC": float(start_pred_np[i]),
                    "HUMAN_AUDIT_REQUIRED": bool(audit_required),
                    "HUMAN_AUDIT_REASON": audit_reason,
                    "PROB_CONFLICT": float(event_probs_np[i, 0]),
                    "PROB_BUMP": float(event_probs_np[i, 1]),
                    "PROB_HARD_BRAKE": float(event_probs_np[i, 2]),
                    "PROB_NOT_SCE": float(event_probs_np[i, 3]),
                }

                for j, group_name in enumerate(pipeline.GROUP_NAMES):
                    row[f"PROB_CONFLICT_GROUP_{group_name}"] = float(group_probs_np[i, j])

                for j, c17_name in enumerate(pipeline.CONFLICT_TYPES_17):
                    row[f"PROB_CONFLICT_17WAY_{c17_name}"] = float(c17_probs_np[i, j])

                rows.append(row)

    elapsed = time.perf_counter() - start_time
    log(f"\nInference complete in {elapsed:.2f}s.")
    return pd.DataFrame(rows)


def build_missing_rows(full_manifest_df: pd.DataFrame, scored_ids: set[str]) -> pd.DataFrame:
    missing = full_manifest_df[~full_manifest_df["BDD_ID"].isin(scored_ids)].copy()
    if len(missing) == 0:
        return pd.DataFrame()

    rows = []
    for _, row in missing.iterrows():
        if not bool(row.get("VIDEO_EXISTS", False)):
            status = "MISSING_VIDEO_FILE"
            reason = "Video path does not exist."
        elif not bool(row.get("HAS_KINEMATICS_CONTEXT", False)):
            status = "MISSING_KINEMATICS_CONTEXT"
            reason = "BDD_ID not found in data/meta.csv, so X_ts/X_ctx cannot be aligned."
        else:
            status = "NOT_SCORED"
            reason = "Unknown skip reason."

        rows.append({
            "BATCH": BATCH_NUMBER,
            "BDD_ID": str(row["BDD_ID"]),
            "VIDEO_PATH": str(row.get("VIDEO_PATH", "")),
            "INFERENCE_STATUS": status,
            "BDD_START_FOR_SAMPLING": safe_float(row.get("BDD_START_FOR_SAMPLING", DEFAULT_BDD_START_FOR_SAMPLING)),
            "MODEL_PRED_EVENT_TYPE": "",
            "MODEL_PRED_EVENT_IDX": "",
            "MODEL_CONFIDENCE": "",
            "MODEL_PRED_CONFLICT_GROUP": "",
            "MODEL_PRED_CONFLICT_GROUP_CONFIDENCE": "",
            "MODEL_PRED_CONFLICT_17WAY": "",
            "MODEL_PRED_CONFLICT_17WAY_CONFIDENCE": "",
            "MODEL_PRED_START_TIME_SEC": "",
            "HUMAN_AUDIT_REQUIRED": True,
            "HUMAN_AUDIT_REASON": reason,
            "PROB_CONFLICT": "",
            "PROB_BUMP": "",
            "PROB_HARD_BRAKE": "",
            "PROB_NOT_SCE": "",
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep CPU threading conservative because video decoding already uses workers.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    pipeline = load_pipeline_module()
    pipeline.register_all_modules()
    pipeline.set_seed(pipeline.SEED)

    log("\n" + "=" * 72)
    log("Unseen batch deployment configuration")
    log(f"Batch number:       {BATCH_NUMBER:03d}")
    log(f"Pipeline file:      {PIPELINE_FILE}")
    log(f"Saved model dir:    {SAVED_MODEL_DIR}")
    log(f"Checkpoint:         {CHECKPOINT_PATH}")
    log(f"Splits:             {SPLITS_PATH}")
    log(f"Unseen video dir:   {UNSEEN_VIDEO_DIR}")
    log(f"Results dir:        {RESULTS_DIR}")
    log(f"Manifest:           {MANIFEST_PATH}")
    log(f"Output predictions: {OUTPUT_PATH}")
    log("=" * 72 + "\n")

    manifest_df = load_or_create_manifest()
    full_manifest_df, X_ts_sel, X_ctx_sel = attach_model_inputs(manifest_df, pipeline)

    usable_df = full_manifest_df[
        full_manifest_df["VIDEO_EXISTS"] & full_manifest_df["HAS_KINEMATICS_CONTEXT"]
    ].copy().reset_index(drop=True)

    if "_usable_order" in usable_df.columns:
        usable_df = usable_df.sort_values("_usable_order").reset_index(drop=True)

    if len(usable_df) == 0:
        log("[WARN] No rows are fully usable for inference.")
        scored_df = pd.DataFrame()
    else:
        scored_df = run_inference(pipeline, usable_df, X_ts_sel, X_ctx_sel)

    scored_ids = set(scored_df["BDD_ID"].astype(str)) if len(scored_df) else set()
    missing_df = build_missing_rows(full_manifest_df, scored_ids)

    final_df = pd.concat([scored_df, missing_df], ignore_index=True, sort=False)
    final_df = final_df.sort_values(["INFERENCE_STATUS", "BDD_ID"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    # Also refresh the manifest in the results folder so it records the actual source list.
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    log("\n" + "=" * 72)
    log("Done")
    log(f"Scored rows:        {len(scored_df)}")
    log(f"Skipped/missing:    {len(missing_df)}")
    log(f"Saved predictions:  {OUTPUT_PATH}")
    log(f"Manifest location:  {MANIFEST_PATH}")
    log("=" * 72)


if __name__ == "__main__":
    main()
