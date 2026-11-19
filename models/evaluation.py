"""Final evaluation of a model on the val pairs, and its comparison table: the metrics of
models/metrics.py for every val pair (fixed order), the E8 chunk baselines on the same
pairs, and the table next to the scripts/baselines.py results measured with the same
dataset, quaternion convention and keep ranges. Shared by training (best weights) and
--eval-only. Moved here from scripts/train.py on 2026-11-19 (refactor step 4c), unchanged.
"""
import json
from pathlib import Path

import numpy as np

from models import data, metrics
from models.pipeline import make_dataset
from models.references import copy_state_mse, state_group_metrics
from models.run_config import OUTPUTS


def baselines_json(canonical, trimmed=False):
    """Baselines measured with the same quaternion convention (and keep ranges) as the run:
    normalised MSE is not comparable across conventions (different label std)."""
    name = "baselines" + ("_canonical" if canonical else "") + ("_trimmed" if trimmed else "")
    return OUTPUTS / name / "baselines.json"

def evaluate(model, val, stats, batch_size, root, factor=None, keep_ranges=None, glitch_thr=None):
    """Predict every val pair (fixed order) and measure it with models/metrics.py.
    glitch_thr (glitch_thresholds of the train split): adds the per-group state report."""
    pred_action, true_action, pred_state, image = [], [], [], metrics.ImageMSE()
    for inputs, targets in make_dataset(val, batch_size, shuffle=False):
        a, s, img = model(inputs, training=False)
        pred_action.append(a.numpy())
        true_action.append(targets["action_output"].numpy())
        pred_state.append(s.numpy())
        image.update(img.numpy(), targets["next_image_output"].numpy())
    t = val["pair_index"]
    physical = metrics.has_tissue_lift_layout(root)   # else core metrics only, no "extra/"
    state_names, action_names = data.feature_names(root)
    if "chunk_factor" in val:   # E8: offsets in option-C units, per step, in mm / deg
        m = metrics.chunk_metrics(np.concatenate(pred_action), np.concatenate(true_action),
                                  data.chunk_anchors(val, t)[0], stats, action_names)
    else:
        m = metrics.action_metrics(np.concatenate(pred_action), val["actions"][t], stats,
                                   physical, action_names)
    m.update(metrics.next_state_metrics(np.concatenate(pred_state), val["states"][t + 1], stats,
                                        physical, state_names))
    m["next_state_mse_copy"] = copy_state_mse(val)   # 1a: reference for next_state_mse, same pairs
    if glitch_thr is not None:   # per group: model / copy / constant velocity, with and without glitches
        m.update(state_group_metrics(np.concatenate(pred_state), val, state_names, keep_ranges,
                                     glitch_thr))
    if factor is not None:   # E7: the loss on the change, model vs f = 0
        m.update(metrics.state_delta_metrics(np.concatenate(pred_state), val["states"][t],
                                             val["states"][t + 1], factor))
    m.update(metrics.next_image_metrics(image))
    return m

def chunk_baseline_metrics(val, stats, root):
    """E8: stand-still and constant velocity on the same val pairs, through the same
    metrics.chunk_metrics as the model (labels only)."""
    t = val["pair_index"]
    true = data.chunk_targets(val, t)   # the get_pairs target, without loading frames
    anchor_n, prev_n = data.chunk_anchors(val, t)
    names = data.feature_names(root)[1]
    return {who: metrics.chunk_metrics(pred, true, anchor_n, stats, names)
            for who, pred in metrics.chunk_baselines(anchor_n, prev_n, stats).items()}

def comparison_table(model_name, model_metrics, root, canonical=False, chunk_baselines=None,
                     keep_hash=None):
    """The model's metrics next to the baselines (if scripts/baselines.py has been run
    on the same dataset, quaternion convention and keep ranges; baselines.json files
    without "dataset_root" are tissue lift, without "keep_ranges_sha256" untrimmed).
    E8 chunk runs: next to their own chunk baselines, because the one-step baselines
    measure a different target."""
    columns = {model_name: model_metrics}
    if chunk_baselines is not None:
        columns.update(chunk_baselines)
        return metrics.format_table(columns)
    path = baselines_json(canonical, trimmed=keep_hash is not None)
    if path.exists():
        baselines = json.loads(path.read_text())
        if (Path(baselines.get("dataset_root", data.ROOT)) == Path(root)
                and baselines.get("keep_ranges_sha256") == keep_hash):
            columns.update(baselines["results"])
    return metrics.format_table(columns)
