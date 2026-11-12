# Keep ranges

A keep-ranges file says which frames of each episode are used for training and evaluation.
It is a CSV with one row per episode:

```
episode,first_visible,last_visible
```

Both indices are inclusive frame indices. An empty `first_visible`/`last_visible` means
the whole episode is dropped. `models/data.py` reads it with `load_keep_ranges`, and
`scripts/train.py --keep-ranges <file>` applies it:
- pairs are kept only where their whole window (t, t+1; or t-1 … t+k-1 with `--chunk k`)
  lies inside [first, last];
- all train statistics (normalisation, quaternion reference, delta and chunk scales) use
  only the kept rows;
- frames are never deleted, so the frame cache stays valid.

An episode of the split that is missing from the file raises an error. The file never
contains test-split episodes.

## `star_il_tool_visible.csv`

**Dataset:** STAR-IL (`data/open_h/Surgical/jhu/imerse/star_il/star_il`), train + val
(episodes 0–459, 460 rows).

**Rule:** keep [first visible frame, last visible frame] of each episode, where *visible*
means that the commanded tool position (`action` `kuka_pose.x/y/z`) projected into the left
endoscope image lies inside the image:
- calibration from `camera_info.yaml` and `transformation_side_camera_wrt_robot_base.yaml`;
- pinhole model, no distortion;
- **lag 0** (frame t paired with the pose at t);
- **no margin**.

This is conservative: frames where only a few pixels of the tool remain count as
invisible. Invisible frames in the middle of an episode are kept (only the start and end
are trimmed).

**Validation:**
- the projected point sits on (slightly above) the tool tip;
- the projected entry/exit frames match hand annotations of the video within a few frames
  (`scripts/look_tool_projection.py --strips` and `--summarise`).

**Command (regenerates this file exactly):**

```bash
python -m scripts.look_tool_projection --all --export-keep-ranges configs/keep_ranges/star_il_tool_visible.csv
```

**Created:** 2026-11-12 (code at commit `b04d90e` + the uncommitted `--all` / `--export-keep-ranges` changes).

**Result:**

| | train (409 ep.) | val (51 ep.) |
|---|---|---|
| invisible frames | 12.2% | 8.3% |
| … at start / end / middle | 1.8 / 10.2 / 0.2% | 0.6 / 7.6 / 0.1% |
| fully invisible episodes | 0 | 0 |
| t → t+1 pairs | 173,626 → 152,744 (−12.0%) | 20,111 → 18,457 (−8.2%) |
| chunk-16 pairs | 167,491 → 146,609 (−12.5%) | 19,346 → 17,692 (−8.5%) |

**Effect on normalisation:** trimming removes the tool-withdrawal tail, so the train std of
action z drops from 29.6 mm to 20.6 mm. Normalised losses of trimmed and untrimmed runs are
therefore not comparable. Compare them in physical units, or against baselines computed with
the same keep ranges.
