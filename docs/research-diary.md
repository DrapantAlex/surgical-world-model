# Research Diary

---

## 2026-10-08

Today I talked with Davide and Michael about the goal of the project and the next steps.

I also understood how the simple world model works. The network has two branches:

- **The actor branch** learns to move the robot autonomously by copying human movements (imitation learning).
- **The world model branch** learns to predict the next state and the next camera frame, given the current image, state and an action.

After training:

- The actor can move the robot autonomously.
- The world model works as a simulator: given an action, it predicts what the next scene should look like.

Training both together is useful so that the actor does not only copy human movements, but also benefits from a representation that captures how the scene evolves.

**Goal.** Develop a model that lets the robot move and perform a surgery autonomously.

**Next steps and ideas.**

- Run the simple world model on the Open-H dataset.
- Run several experiments to improve the model's performance (ablation studies).
- Secondary goal: make the robot able to complete a brain surgery.

---

## 2026-10-09

Today I explored the Cosmos-H-Surgical-Simulator repository to understand how it handles data and evaluation. I also set up my own repository, `surgical-world-model`, where I began examining how the Open-H dataset is organized and how I can use it in my work.

I learned these concepts:

- **Episode:** one recorded attempt to complete the task, from start to end.
- **Video:** the sequence of frames collected by one camera during an episode. An episode has one video per camera.

---

## 2026-10-12

**Data.** I studied the structure of an Open-H episode (LeRobot v2.1, `srt_tissue_lift`). The JSON files in `meta/` describe the whole sub-dataset (225 episodes): file paths, features, episode lengths, tasks, statistics for normalisation, and (`modality.json`, NVIDIA-specific) the meaning of each slice of the state and action vectors. Each episode has one parquet file (one row per frame: 16D state, 16D action, text, bookkeeping columns) and one MP4 video per camera. Row *t* of the table corresponds to frame *t* of each video.

**Findings.**

1. The right endoscope video is an exact copy of the left one (verified on episode 0, consistent with the per-episode statistics), so the dataset has no real stereo; I use only the left view.
2. The state is the measured pose, the action is the commanded setpoint (`_sp`).

**Pipeline.** I wrote a loader that builds training triplets (frame *t*, state *t*, action *t* → frame *t+1*, state *t+1*), with images resized to 224 × 224 and state/action normalised with `stats.json`.

**Model.** I adapted the professor's actor-world model to Open-H (16D state and action). I added a 512-unit bottleneck before the image decoder: the original first decoder layer had about 2.5 billion parameters (about 40 GB to train); the adapted model has 72 M in total.

**Overfitting test** (episode 0, 300 epochs, DGX A100). Next-state MSE ≈ 0.0001 (normalised). Image MSE 0.000372 against 0.000825 for the "copy the current frame" baseline (ratio 0.45). The pipeline works end to end, but the image branch is not fully memorised.

**Infrastructure.** Set up the DGX (SSH key for GitHub, virtual environment, TensorFlow on one GPU with memory growth) and W&B logging.

---

## 2026-10-13

Today was a productive day: I now understand how training works and how to extract the model inputs from the raw data.

### Input pipeline

- **Frames:** the videos are decoded into individual frames, which are then resized into 224 × 224 × 3 image arrays that the visual encoder can take as input.
- **States and actions:** read directly from the Parquet files.

I grouped frames, states and actions into aligned arrays and used them as model inputs.

### Overfitting test (sanity check)

I trained the model on a single episode, without the latent *z* layer. The model overfits both actions and states well, but it beats the copy baseline on only about half of the [time steps / dimensions]. Since this is a single episode, the test only checks that the pipeline and the model work end to end; it says nothing about generalization. As a sanity check, the result is acceptable.

### Architecture: what I understood

The actor and the world model are two separate heads that share the visual encoder, which maps each image to an embedding. The actor predicts the action, while the world model predicts the next [frame / embedding]. Because the encoder is trained by both losses, the world model loss acts as an auxiliary signal that pushes the encoder towards representations that capture the dynamics of the scene, which can also help the actor.

For my goal, making the robot choose the right action, only the encoder and the actor are needed at inference time. The world model is still useful during training, because it shapes the shared encoder.

### Meeting with Michael

- Train the model on more episodes and compare different visual encoders from Keras Applications.
- Apply Michael's model to Open-H. This dataset is harder because some episodes involve robots with several arms.

**Open question:** how should I handle action prediction when the number of arms varies across episodes?

---

## 2026-10-14

Today I read a paper that tries to answer the question: *"Can large visual encoders, pre-trained on out-of-domain data, be useful for control tasks, instead of training the encoder from scratch?"*

I understood that frozen pre-trained visual encoders can perform better than encoders trained from scratch, especially when little training data is available. This supports Michael's suggestion to use pre-trained encoders. Since our task needs precise positions, I will also extract features from the middle of the network, not only from the last layer, and compare which works better for predicting the actions.

I also understood that an action is a single vector that describes the movement of all the arms at once. Each arm is described by 8 numbers:

- the position of the instrument tip (x, y, z: 3 numbers);
- the orientation of the tip (a quaternion: 4 numbers);
- the opening of the gripper (jaw: 1 number).

The state has the same structure:

- **State** = where the tip is now (measured).
- **Action** = where the tip is told to go (commanded).

I downloaded the full dataset (225 episodes) to the DGX, but only the left endoscope videos, because the right view seems to be a duplicate of the left one. I still need to check this for all episodes.

Finally, I extended `data.py` so that it loads the whole dataset with the official train/validation/test split, and I verified the result on the DGX.

### Questions for tomorrow

1. **Goal and evaluation:** "Is the final goal a model that predicts correct actions to control a real robot? If so, how can I evaluate it without access to the robot?"
2. **Pipeline:** "Could we outline the main steps from now to the end of the thesis? Michael suggested training on more episodes first, then comparing different encoders. Is that the right order?"
3. **Model:** "Should I build on the model Michael gave me, or also consider other world-model architectures?"
4. **Robots:** "Open-H contains robots with different numbers of arms. Should my model handle all of them, or should I focus on one type of robot?"
5. **PoliTo:** "How should I coordinate with my supervisor at Politecnico di Torino? Should they be involved in these discussions?"
6. **Research question:** "What is the main research question of the thesis? What should be its contribution?" For example: a comparison of pre-trained encoders for surgical imitation learning? Showing whether a world model helps the actor? The whole plan depends on this answer.
7. **Contact:** "Who is my day-to-day contact for technical questions: Michael?"

---

## 2026-10-15

### 1. Left and right cameras

I checked whether the left and right endoscope views contain the same images. In all 225 episodes of `srt_tissue_lift`, the right endoscope video is a byte-identical copy of the left one (identical statistics and identical file hashes). The right view therefore adds no information, and I use only `observation.images.endoscope.left`.

I also clarified that "left/right" means two different things in this dataset: for the endoscope, it refers to the two lenses of one stereo camera; for the wrist cameras, it refers to the left and right arms.

### 2. Paper: OmniRAS (sent by Davide)

OmniRAS is a family of large video encoders (1B and 2B parameters) based on Meta's V-JEPA 2.1. The authors continued its self-supervised pretraining on about 2,650 hours of surgical video, then evaluated it on several surgical tasks: phase and step recognition, action-triplet recognition, action segmentation and action detection.

**Main findings:** once partially fine-tuned, OmniRAS gives the best results on all benchmarks; when frozen, its advantage over the original encoder depends on the task.

### 3. JEPA

JEPA (Joint-Embedding Predictive Architecture) is a self-supervised method for training an encoder: the model predicts the embeddings of hidden parts of the input, instead of their pixels. Applied to my project, this suggests a possible future change: a world model that predicts the embedding of the next frame, so that its predictions live in the latent space and no image decoder is needed.

### 4. Baselines

I computed four trivial predictors on the validation split, to know what the trained model has to beat:

| Baseline | Rule | Result |
|---|---|---|
| `mean_action` | the action is always the mean action of the training episodes | Huber 0.41 (4.8 mm) |
| `action_eq_state` | the action is always equal to the current state | Huber 0.0011 (0.18 mm) |
| `copy_state` | the next state is always equal to the current state | MSE 0.0018 (0.15 mm) |
| `copy_frame` | the next frame is always equal to the current frame | MSE 0.0007 |

For each model output, the trained model has to beat the strongest baseline: `action_eq_state` for the action, `copy_state` for the next state, and `copy_frame` for the next image.

### 5. Metrics

- **Huber:** the Huber loss on normalised values, the same formula as the training loss for the action.
- **MSE:** the mean squared error on normalised values, the same as the training losses for the next state and the next image.
- **pos_mm:** the Euclidean distance in 3D between the predicted and the true instrument-tip position, in millimetres.
- **rot_deg:** the angle, in degrees, between the predicted and the true orientation.
- **jaw_raw:** the absolute error of the gripper opening.

The physical metrics are computed for each arm separately and for both arms together.

---

## 2026-10-16

### Goals

Understand the model architecture in detail. Switch to a single-arm Open-H subset. Launch a first training run.

### Architecture

I studied the actor–world model block by block:

- **Shared visual encoder:** 4 strided convolutions → 14 × 14 × 256 → 50,176 features
- **Actor:** concat → Dense 256 → action
- **World model:** next-state head, plus an image decoder (512 bottleneck *z* → Dense + Reshape 14 × 14 × 256 → 4 transposed convolutions → 224 × 224 × 3)

**Key point:** the actor and the world model share only the encoder. The world model therefore acts as an auxiliary task that shapes the encoder's visual features.

### Dataset: STAR-IL

- `Surgical/jhu/imerse/star_il/star_il`: KUKA LBR iiwa with Endo360, ex vivo suturing, 512 episodes (409/51/52), ~3.8 GB with the left endoscope only.
- State = 8 joint positions, action = 7D Cartesian pose (xyz + quaternion, no jaw). A single instruction ("suture throw"), so the language input is constant.
- Episodes 504–511 (test split) have no endoscope video on the Hub, although `info.json` reports 1024 videos. They're skipped automatically, so the test set has 44 episodes.

### Code (with Claude Code)

The pipeline is now dataset-agnostic:

- `--dataset-root`, with dimensions read from `info.json`
- missing videos skipped and logged
- train-only normalisation statistics
- the same core metrics for every dataset (`action_mse`, `action_huber`, `next_state_mse`, `next_image_mse`), with the tissue-lift physical metrics under `extra/`

### Checks

- Shape check ✓ (71.9 M parameters).
- STAR-IL val baselines: mean-action 1.008, copy-state 0.0002, copy-frame 0.0006.
- Tissue-lift val baselines: action = state 0.0022, i.e. the actor can shortcut by copying the state. STAR-IL has no such shortcut.
- Overfit on episode 0 ✓ (loss 1.26 → 7.2e-4), but the image head stays at 1.1e-4 vs 2.0e-5 for copying.
- Smoke test ✓.

### Insights

1. At 30 fps, consecutive steps are almost identical, so the copy baselines are very strong for the world-model heads. Candidates: residual targets, or a k-step horizon.
2. With `latent = None`, the checkpoint is ~10 GB, so use latent 512.

### Running

`star-il-simplecnn-latent512-seed0` on the DGX (GPU 3, tmux `swm`), with the same hyperparameters as tissue lift.

---

## 2026-10-19

### Setup

I connected VS Code to my repository on the DGX via the **Remote-SSH** extension, so I can work on the cluster files directly from VS Code.
I installed **Claude Code** in the repository and configured its settings files and `CLAUDE.md`, so that it can work on the repository following my conventions.

### Results of the first STAR-IL run

Run: `star-il-simplecnn-latent512-seed0` — **now my reference baseline**.

| Val metric     | Model | Baseline                |
|----------------|-------|-------------------------|
| action MSE     | [x]   | 1.008 (mean action)     |
| next-state MSE | [x]   | 0.0002 (copy state)     |
| next-image MSE | [x]   | 0.0006 (copy frame)     |

1. **The actor learns, but generalises poorly.**
   It is far better than the mean-action baseline. However, there is a large generalisation gap: after epoch 4, train action loss improves ~5× while val improves only ~10%, so the model mostly memorises the training episodes. The next-state head overfits for real (val loss rises after epoch 2).
   Likely causes:
   - only 409 demonstrations with nearly identical consecutive frames;
   - a 72M-parameter model;
   - no regularisation (no dropout, weight decay or augmentation).

2. **The world-model heads are worse than the copy baselines.**
   At 30 fps, consecutive frames and states are almost identical, so copying the input is already very accurate, while the model has to reconstruct the whole state and image.
   A possible fix is to predict the change (Δstate, Δimage) instead of the absolute values. This addresses problem 2, not the overfitting.

### Next steps

- [ ] **Diagnostics:** train/val overlay plot, per-dimension action error, state-only baseline.
- [ ] **First regularisation experiment:** weight decay (AdamW), compared against the reference run with everything else unchanged.
- [ ] **Later:** image augmentation, and residual targets for the world model (to discuss with Michael).

---

## 2026-10-20

### Reading

Today I read the paper "Generalized Recognition of Basic Surgical Actions…". It introduces **Basic Surgical Actions (BSA)** and shows that a video transformer, built on a pretrained ViT-B/16, can classify them very accurately in surgical videos.
This is a perception task, not control, so it relates to my **encoder** rather than my actor. Two ideas for future extensions:

1. break the evaluation down per action, to see which ones are hardest;
2. try an encoder that takes several frames as input.

### Diagnostics of the first STAR-IL run

I added per-dimension errors (action and next state) to the metrics and re-evaluated `star-il-simplecnn-latent512-seed0` with its saved weights (no retraining).

| | Position (x, y, z) | Orientation (quaternion) |
|---|---|---|
| Val MSE per dimension (normalised, ≈ 1 − R²) | 0.011–0.034 | 0.12–0.20 |
| Val / train | ×27–49 | ×100–335 |

1. **The model learns the tool position well, but not its orientation.**
   The quaternion is ~90% of the val action error. Rotation error on val: median 3.6°, mean 6.6° (train: 0.87° mean).
2. **Orientation is almost constant within an episode but varies a lot between episodes** (up to 122° from the mean orientation).
   So there are only about 409 distinct orientations, and the model has enough capacity to partly memorise them.
3. **The quaternion labels have an inconsistent sign.**
   The same rotation is stored as q in some episodes and as −q in others (259 vs 150 train episodes on the two sides), and MSE treats them as different targets. So on unseen episodes the model sometimes predicts the wrong sign (4.9% of val frames) or an average of the two (|q| < 0.7, 10.5% of val frames).
   The effect is mainly between episodes; within episodes it is rare (71 sign flips in 174k frames).

Overall, three causes:

- the sign inconsistency (measured: about **57%** of the val orientation MSE);
- few distinct orientations;
- general overfitting (position overfits too).

### The fix: canonical quaternion sign

I canonicalise the quaternion sign at load time (`--canonical-quaternion`, off by default). Every quaternion is flipped to the side of a reference rotation, the mean rotation of the train labels, which is saved with each run as `quat_reference.json`. The Open-H files stay unchanged.

**E0** confirmed that it works: 0 sign flips and 0 opposite-sign neighbours on train and val. Every orientation is at most 123.5° (train) / 120.8° (val) from the reference, well below 180°, where the sign would become ambiguous again.

Options, decision and success criteria: `docs/decisions/2026-10-20_next-direction.md`.

### Next steps

- [ ] **E1:** retrain with canonicalised labels, 3 seeds (running overnight), plus the old reference with seeds 1 and 2 to measure the seed-to-seed variation. Compare in degrees, not in normalised MSE.
- [ ] **E2:** a state-only MLP (joint angles → action), to check whether the image encoder helps.
- [ ] Then decide between regularisation (E3) and other choices (relative actions, 6D rotation).

---

## 2026-10-21

### E1: canonicalisation runs (3 seeds)

I checked the canonicalisation runs (E1, 3 seeds). **Just making the quaternion sign consistent reduced the val rotation error from 6.6° to 0.92° and the position error from 5.7 to 2.9 mm.**

The tissue-lift dataset doesn't have STAR-IL's sign problem, so the problem comes from STAR-IL's logging.
I also confirmed the units: positions in metres, joint angles in radians, orientation as quaternions.

### E2: state-only MLP

I ran E2, a state-only MLP with no image. **It performs better than the full image model** (0.41° / 1.2 mm vs 0.92° / 2.9 mm, better on 49 of 51 val episodes).
For an absolute pose, the target is almost forward kinematics from the joint angles, so the image isn't needed, and the CNN branch mostly memorises episodes (train/val gap ×7 vs ×2).

### Next steps

- [ ] **World-model-off ablation** (running, 3 seeds): does the world-model auxiliary task help the actor?
- [ ] **Relative action chunks:** the actor predicts the next k movements relative to the current pose, so it has to use the scene. To read: ACT (Sec. IV.A, VI.A) and Open-H (GR00T-H methods).
- [ ] **Delta state** for the world model, instead of the full next state.
- [ ] **Regularisation**, once the architecture is fixed.
- [ ] **Encoder comparison** and more Open-H datasets.

---

## 2026-10-22

### E6: world model off

E1 (WM on) vs E6 (WM loss weights = 0), 3 seeds, 51 val episodes:

| | WM on | WM off |
|---|---|---|
| Rotation | 0.92° | 0.89° |
| Position | 2.90 mm | 2.92 mm |

**Verdict: no detectable effect** (seed ranges overlap; E1 better on 26/51 episodes).
This was expected: the image isn't needed for this task (E2), so a loss that shapes the image encoder can't help.
To check: the nowm runs were trained from uncommitted code; some runs hit the 100-epoch limit.

### The shortcut

At 30 Hz the tool moves ~0.1 mm per frame, so action_t ≈ state_t+1 ≈ state_t.

- **Actor:** in STAR-IL the state is 8 joint angles, and the next pose ≈ forward kinematics(angles). The network learns this formula and ignores the image.
- **World model:** it gets action_t and predicts state_t+1 ≈ action_t, so it can copy.

### Papers

- **ACT:** predict a chunk of k actions from one observation, which reduces compounding errors (success 1% at k = 1 → 44% at k = 100). Temporal ensembling adds only ~3%. A plain CNN + MLP also benefits from chunking.
- **GR00T-H:** GR00T-N1.6 post-trained on 601 h of surgical data. Chunks of 50, relative EEF + 6D rotation, normalisation per chunk step. It beats ACT, especially in unseen scenes (54% vs 5%), so small models generalise poorly.
- **Related work:** Surgical WAM, FLARE, Cosmos-H-Surgical. None tests a WM auxiliary loss on/off in a small model on real data.

### Proposed changes (asked Davide and Michael)

- **Actor:** predict k relative actions (anchor a_t−1) → the target becomes the movement, which needs the image.
- **WM:** predict the change of state (open issue: Δstate ≈ Δaction).
- **Baselines:** stand still, state-only; error per chunk step; repeat E6.
- Chunks first, then the encoder comparison.

### Next steps

- [ ] Relative-action statistics and stand-still error per horizon (no training).
- [ ] Update the decision record; add Surgical WAM and FLARE to the Library.

---

## 2026-10-23

### Stand-still baseline per horizon

How wrong is the commanded pose a(t) as a prediction of a(t+h), for h = 1..60 frames? No training, labels only, loaded like train.py (canonical quaternions). Time = frame_index / 30, because STAR-IL timestamps are constant within an episode. Script: `scripts/look_standstill_baseline.py`.

Val split (51 episodes, mean / median):

| h | time | Position | Rotation |
|---|---|---|---|
| 1 | 0.03 s | 0.42 / 0.29 mm | 0.09 / 0.03° |
| 5 | 0.17 s | 1.92 / 1.35 mm | 0.36 / 0.12° |
| 10 | 0.33 s | 3.53 / 2.42 mm | 0.67 / 0.22° |
| 30 | 1.0 s | 8.78 / 5.88 mm | 1.69 / 0.56° |
| 60 | 2.0 s | 15.71 / 10.75 mm | 2.80 / 1.18° |

Train gives almost the same curve. Correction to yesterday: the tool moves ~0.3 mm per frame (median), not ~0.1 mm.

**Static frames** (a(t) → a(t+1) below 0.1 mm and 0.1°): 25% on val, 24% on train. Per episode from 5% to 54%.

### E1 and E2 in "frames of delay"

At h = 1 the stand-still baseline uses the same 20,111 val pairs as E1 and E2, so the errors can be compared directly:

| | Error (mean) | ≈ stand still for |
|---|---|---|
| E1 image model | 2.90 mm, 0.92° | 8 frames (pos), 14 frames (rot) |
| E2 state-only | 1.24 mm, 0.41° | 3 frames (pos), 6 frames (rot) |

**No model beats standing still for one frame** (0.42 mm, 0.09°). This is not a fair contest, because the actor never sees a(t). But it gives a physical scale: E2 is as wrong as a command 0.1–0.2 s old, E1 as one 0.3–0.5 s old.

### What this means for relative action chunks

- A quarter of the targets would be ≈ 0, so a model that always predicts "no motion" already scores well. Report chunk errors next to the stand-still curve, and separately for static and moving frames.
- The error keeps growing with h (~0.35 mm per frame up to h = 10, ~0.23 mm per frame from h = 30 to 60), so later chunk steps will dominate an unweighted loss. Normalisation per chunk step (as in GR00T-H) handles this.
- Rotation: mean ≈ 2.4–3.6× median, so most rotation change happens in a few large reorientations.

### Next steps

- [x] Stand-still error per horizon.
- [ ] Relative-action statistics (distribution of Δa per chunk step).
- [ ] Add Surgical WAM and FLARE to the Library.

---

## 2026-10-27 (World Model Day)

### Roadmap step 0: errors by phase and speed (no retraining)

Val errors of E1 (image + WM), E6 (image, WM off) and E2 (state-only), split by phase and speed, 3 seeds each. Script: `scripts/look_static_vs_moving.py`.

- **Static vs moving** (25% static frames, same threshold as the stand-still script): no model does better on static frames.
- **Speed bins** (moving frames only, 3 equal groups by Δpos per frame: slow 0.03–0.29, medium 0.29–0.61, fast 0.61–5.89 mm/frame): the error is flat across speed. A second speed measure (max-rank of position/rotation change) gives the same result.
- **State-only wins in every group**: ~1.2 mm / 0.4° vs ~2.9 mm / 0.9° (2.3–2.7× better on mean, median and p90). E1 ≈ E6 in every group.

**Interpretation:** the current actors predict where the tool should be now, which is almost where it already is (action ≈ FK(state)). They read the pose and don't predict motion. This is the "before" picture for the chunked actor, whose error should grow with speed.

### Davide (reply to my roadmap)

- Roadmap approved.
- WM targets from a frozen encoder: "ALPS e DINO fanno meglio di JEPA".
- Suggested Long-WAM (step-by-step with teacher forcing). My plan: start with parallel jumps (t+4, t+8, t+12, t+16), and use Long-WAM only if the WM doesn't help. Still to confirm with him.

### Meeting with Michael

- Distribution analysis of actions and states in Open-H.
- Penalise orientation errors more (exact meaning to clarify).
- Check the normalisation. Checked in the code: per-dimension z-score with train-split mean/std (`compute_norm_stats` / `normalise` in `models/data.py`, std floored at 1e-6, quaternions canonicalised before the stats).
- WM with two targets: Δstate (actions given as input) + future-image features from a frozen encoder.
- Action chunking. The action should be a "velocity".

### Open questions for Michael

- "Velocity": the change from the previous step, or offsets from the current pose? (Per-step changes pile up errors; ACMT found chained deltas worse.)
- Orientation penalty: a bigger loss weight, or an angle-based loss? Still needed with 6D?
- Δstate horizon (one frame or the whole chunk) and units (joint angles or tool pose).

### Planned WM ablation

All with the same chunked actor and 3 seeds:

| | WM |
|---|---|
| A | off |
| B | full (Δstate + image features) |
| C (optional) | image features only |

### Next steps

- [ ] Distribution analysis of actions and states (train vs val, absolute values and changes over 1 and 16 frames).
- [ ] Send the open questions to Michael.
- [ ] Chunked actor (k = 16) + 6D, with stand-still, constant-velocity and state-only baselines. Rerun the phase/speed table on it.

---

## 2026-10-28

### Today in five lines

- I studied the STAR-IL distributions to choose the normalisation from data (Michael said normalisation is the main problem).
- Positions are well-behaved, but changes (one frame or 16 frames) are tiny, smooth, heavy-tailed and often exactly zero.
- From this I chose: keep the z-score for positions; use a median-based scale + Huber for change targets; use option C for action chunks later.
- I prepared E7: the world model now predicts the change of the joints instead of the absolute next state.
- E7 is coded (one line written by me), the smoke test passed, and the 3 training runs are running.

### What the data showed → what I chose

| # | The data shows | So I chose |
|---|---|---|
| 1 | Positions look like a bell curve (kurtosis −1.3 to 1.4, max \|z\| 4.5), but scales differ ~40× (tool position std 0.025, joint 6 std 0.95); train and val means differ ≤ 0.25 std | train-only z-score for the state input; val is a fair test set |
| 2 | One frame moves ~1% of the pose spread (0.3 mm vs 24 mm); "copy the last action" already gets 0.42 mm / 0.09° on val | report every result next to the stand-still baseline; move to targets that are the motion itself |
| 3 | The 16-step change is ~13× the 1-step change (0.31 → 3.96 mm), close to 16 (steady motion), far from 4 (random noise) | 16-step action chunks + a constant-velocity baseline |
| 4 | Changes have heavy tails (kurtosis up to 339) and many exact zeros (endoscope still 84% of frames); joint 6 IQR/std ≈ 0.002 | scale changes by the median, with a floor, and train with Huber |
| 5 | Chunk offsets grow ~13× from step 1 to 16, while the ratio stays ~11.8 mm per degree at every step | option C: one median scale per group (position, rotation) per step, rotation as one angle (6D), Huber; fallback = option A (GR00T-H, Open-H Table S2 p. 52) |

**Key ideas to remember:**

- **Position vs change:** the z-score is good for positions, bad for changes (the std of changes is set by rare jumps).
- **Normalisation = hidden loss weights.** Per-step scaling makes step 1 and step 16 count equally (with one ruler and MSE, step 16 would count ~170× more).
- **Huber** stops rare jumps from dominating; it does not fix the step imbalance (that is the per-step scale's job).

### E7: what it is

E7 = E1 + one design change: the WM state branch predicts the change f (Δstate), not the absolute next state.

**Why:** with the absolute next state, "copy the current state" is almost perfect, so the WM had nothing to learn. This is a likely reason why E6 (WM off) = E1 (WM on).

### E7: what was implemented (by Claude Code, except one line)

| File | Change in plain words |
|---|---|
| `swm.py` | The state branch outputs f (the change). A fixed skip connection adds it back: next state = s(t) + k × f, with k = scale / std (normalised units). No new trainable weights. |
| `data.py` | `compute_delta_scale`: from raw train states, Δ = s(t+1) − s(t); one scale per joint, saved in `norm_stats.json` (`state_delta`). The WM target = Δ / scale. The scale line is mine. |
| `train.py` | New flag `--state-delta`; `DELTA_FLOOR_C = 0.25`; WM state loss = Huber on the scaled change; logs the "predict f = 0" baseline (`val_next_state_output_loss_zero`). Old runs still load. |
| `metrics.py` | `state_delta_metrics`: model's Huber on f vs Huber of f = 0. |

**How the loss is applied:** it sits on `next_state_output` with the usual s(t+1) target, but both are divided by k first. Since ŝ(t+1) − s(t+1) = k · (f − Δ/scale), this is exactly Huber(f, Δ/scale). So there are still 3 outputs and `get_pairs`, the pipeline and the plots stay unchanged.

**The scale rule** (8 numbers, one per joint): scale = max(median |Δ|, 0.25 × mean |Δ|), raising an error if a joint never moves.

- Median = typical movement, when the joint usually moves (e.g. kuka 1: 1.5e-4 rad).
- 0.25 × mean = safety floor for "lazy" joints whose median ≈ 0 (kuka 6 → 2.5e-5, endoscope → 2.5e-4).
- Why scale at all: raw changes (~0.00005 rad) are too small for the loss to notice, and joints differ ~25×.
- Why 0.25: it caps every joint's average target at 4; it is a judgement call (0.5 would also work).

Script for the per-joint numbers: `scripts/look_state_delta.py`.

### E7: status

- Smoke test `smoke-e7-delta` passed: `state_delta` in `norm_stats.json`, the f = 0 column in `history.csv`, both Huber values in the final table.
- Code committed as `6dd76b3`; 3 seeds launched in parallel (tmux `e7`, GPUs 2–4), same settings as E1 (state weight 1.0, image weight 0.1, Adam 3e-4, batch 32, patience 10).
- Off arm = the 3 existing E6 runs (older commit, same actor). If on vs off differ, rerun "off" on the same commit.
- **Compare the WM with the exact numbers in `val_metrics.json`** (`next_state_delta_huber` vs `next_state_delta_huber_zero`). The Keras val loss averages batch means equally, so the last, smaller batch is over-weighted: 3.912 vs 3.954 in the smoke run (25 batches), negligible on the full val split (629 batches).
- The val frame cache was rebuilt at launch: same 51 episodes, 20,111 pairs as E1/E6.

### Next steps

- [x] Distribution analysis of actions and states.
- [ ] E7 results: action error vs E6, and WM Huber vs f = 0.
- [ ] Send the open questions to Michael.
- [ ] Chunked actor (k = 16) + 6D, option C normalisation.

---

## 2026-10-29

### Today in five lines

- E7 (WM predicts Δstate) **failed**: the actor got ~12× worse, and the WM never beat "no motion" on val.
- Two label-only diagnostics (D1, D2) explain why: one-frame Δ can't be predicted from the WM inputs, so the WM memorised the train episodes.
- E7 also made the state loss ~1000× the action loss, so that memorisation took over the shared encoder.
- Michael's design (predict the absolute state, then subtract) is E1: the change in E7 was the loss scale, not the architecture.
- Decision: keep the E1 world model, and make the **actor** predict motion (chunk of 16 offsets) → E8 / E9.

### E7 results (3 seeds, 51 val episodes)

| | E1 (abs. state) | E6 (WM off) | E7 (Δ state) |
|---|---|---|---|
| val action Huber (best) | 0.0027–0.0031 | 0.0026–0.0035 | 0.032–0.041 |
| best epoch / epochs run | 45–100 | 74–100 | 1 / 11 |
| WM state Huber vs "f = 0" (val) | – | – | 3.07–3.11 vs 2.955 |

Every seed: val action rises from epoch 1 (0.032 → 0.063), train action fits ~20× worse than E1, and the state loss is ~200× (epoch 1) to ~900× (epoch 11) the action loss (in E1 they were ~1:1).

### D1: "no motion" on each split

Script: `scripts/look_delta_reference.py` (labels only). It reproduces E7's logged 2.955, so the target is the same as in training.

| | f = 0 | E7 head, epoch 1 | E7 head, epoch 11 |
|---|---|---|---|
| train | 3.59 | 3.49 | 2.10 |
| val | 2.955 | 3.1 | 4.0 |

**Memorisation, not a train/val shift**: below "no motion" on train, above it on val. A constant (the train mean motion) is worse than 0 on both splits.

### D2: can the one-frame Δ be predicted without the image?

Script: `scripts/look_delta_predictability.py`. Probes on train (ridge, MLP 2×256 Huber), val Huber relative to f = 0:

| Inputs | Best probe | val / (f = 0) |
|---|---|---|
| s, a (the WM inputs minus the image) | ridge | 1.13 (MLP: 1.23, same pattern as E7) |
| s, a, a(t) − a(t−1) | MLP | 0.63 |
| constant velocity f(t−1), no learning | – | 0.50 |
| s, a, f(t−1) | MLP | 0.42 |

- From absolute values, the network can't resolve a change of ~1/1000 of their spread. The same information given as a difference is learnable.
- The motion is smooth: history predicts it well. The WM inputs have no history.

**Conclusion on E7:** the target was not identifiable from the WM inputs, and the amplified loss let the memorisation damage the actor. E1 asked for the same information (the motion), but so quietly that the WM just copied the state.

### Decision: E8 = chunked actor on the E1 architecture

- WM unchanged (absolute s(t+1), MSE, same inputs). Only the actor target changes.
- k = 16; target = offsets from the last command, a(t+i) − a(t−1), i = 0..15.
- Quaternions as now, component difference; reconstruct with normalise(q(t−1) + Δq).
- Option C: one scale per group (position, rotation) per step = max(median, 0.25 × mean) of the group norm, train only; Huber.
- Baselines on the same pairs: stand-still (a(t−1)) and constant velocity.
- E9 = E8 with the WM off, same commit.
- Expect step 1 ≈ stand-still (D2). The test is steps 4–16.

### Next steps

- [ ] E8 code: groups + chunk scale, chunk pairs/targets, per-step metrics + baselines, chunk head, `--chunk` flag.
- [ ] E8 / E9, 3 seeds each.
- [ ] Send the open questions to Michael.

---

## 2026-11-12

### Today in five lines

- In many STAR-IL episodes the tool is out of view, mostly at the end. I checked whether the camera calibration can detect those frames: it can.
- The commanded tool position, projected into the endoscope image, lands on (slightly above) the tool tip, and its entry/exit frames match my hand annotations within a few frames.
- Over train + val, 11.8% of frames have the tool out of view (train 12.2%, val 8.3%), about 85% of them at the end of an episode.
- I implemented trimming: each episode keeps only [first visible, last visible] frame, for the pairs and for every train statistic (`--keep-ranges`).
- Re-evaluating E8 on the trimmed val pairs shows that E8 is at stand-still level (ratio 1.00 untrimmed, 1.016 trimmed). Trimming removes easy frames; it doesn't fix the actor.

### Step 1: does the projection find the tool?

Script: `scripts/look_tool_projection.py` (inspection only, no change to the model).

**Maths** (all numbers read from the dataset folder):
- the transform file gives the camera pose in the world (p_world = R p_cam + t), so world → camera is **p_cam = Rᵀ (p_world − t)**;
- pinhole at the calibration size (848×480), u = fx X/Z + cx, v = fy Y/Z + cy, then rescaled per axis to the video (960×540);
- visible ⇔ Z > 0 and the point lies inside the image;
- distortion only moves the drawn dot. The plumb_bob polynomial (k3 < 0) can fold points far outside the image back inside, so visibility is decided on the undistorted point.

**What I checked:**

| Check | Result |
|---|---|
| Ranges (10 train episodes) | Z > 0 in 100% of frames, depth 0.05–0.17 m: same frame, right direction, right units |
| Dot test (8 frames per episode) | dot on, slightly above, the tool tip; "OUT" with v < 0 (above the image) when the tool is withdrawn |
| Timing strips (`--strips`, 8 episodes) | projected entry/exit = first frame of the new state; matches the real tip entry/exit within a few frames (`timing_validation.csv`) |

**Lag:** `--lag-frames k` is a pure shift (projected frame + k), so annotating once is enough and the mean error is itself an estimate of the lag. **Decision:** lag 0, no margin. This is conservative: frames where only a few pixels of the tool remain count as invisible.

### Step 2: how many frames are invisible? (`--all`, tables only)

| | train (409 ep.) | val (51 ep.) |
|---|---|---|
| invisible frames | 12.2% | 8.3% |
| … start / end / middle | 1.8 / 10.2 / 0.2% | 0.6 / 7.6 / 0.1% |
| trail (end), seconds: mean / median / max | 1.44 / 1.10 / 13.2 | 1.00 / 0.00 / 6.2 |
| episodes: fully visible / middle absence / fully invisible | 175 / 14 / 0 | 29 / 3 / 0 |
| chunk-16 pairs, before → after trimming | 167,491 → 146,609 (−12.5%) | 19,346 → 17,692 (−8.5%) |

- **The problem is the end of the episode.** More than half of train episodes end with the tool out of view.
- **Middle absences are negligible** (0.2%), and each affected episode has a single absence. They are kept.
- **Outliers:** episode 317 has a 13.2 s tail (half the episode); episodes 393, 343, 369 and 434 have 6–7 s.
- **Val has fewer invisible frames than train**, so trimming changes train more than val.

### Step 3: trimming (implemented by Claude Code)

| File | Change in plain words |
|---|---|
| `configs/keep_ranges/star_il_tool_visible.csv` | tracked file `episode, first_visible, last_visible` (460 rows, train + val); regenerated exactly by `look_tool_projection --all --export-keep-ranges`; rule, command and result in its `README.md` |
| `data.py` | section B1: `load_keep_ranges`, `kept_window`, `load_kept_table`, `pair_positions`, `check_keep_ranges`, `file_sha256`. Pairs only where the whole window (t, t+1 or t−1 … t+15) lies in the kept range. Norm stats, quaternion reference, delta and chunk scales read only kept rows. Frames are not deleted, so the cache stays valid. `data.py` knows nothing about the projection. |
| `train.py` | `--keep-ranges`: the file is copied into the run folder with its hash; `--eval-only` reads the copy and checks the hash; `--eval-keep-ranges` evaluates any run on other val pairs (separate JSON); `keep_ranges` column in `results.csv`; baselines are only compared on the same hash |
| `baselines.py` | `--keep-ranges`, output in `outputs/baselines[_canonical]_trimmed/` |

**Safety checks:**
- a missing episode raises an error;
- a file whose episodes ≠ train + val of the dataset is rejected before the run folder exists (this caught a tissue-lift run with the STAR-IL file);
- old runs load as before: recomputing E8's stats without keep ranges reproduces its `norm_stats.json` exactly.

**Trimming changes the normalisation:** the train std of action z drops from 29.6 mm to 20.6 mm, because the tail is the tool being lifted out. **Normalised losses of trimmed and untrimmed runs are not comparable.** Compare them in mm / degrees, or as a ratio to the stand-still baseline on the same pairs.

### E8 on the trimmed val pairs (same weights, `--eval-keep-ranges`)

3 seeds, mean of the 16 steps, model / stand-still on the same pairs:

| | pairs | position mm | ratio | rotation deg | ratio |
|---|---|---|---|---|---|
| untrimmed val | 19,346 | 2.981 ± 0.034 (stand-still 2.986, const. vel. 2.573) | 0.998 ± 0.011 | 0.578 ± 0.003 | 1.005 ± 0.005 |
| trimmed val | 17,692 | 2.807 ± 0.028 (stand-still 2.764, const. vel. 2.452) | 1.016 ± 0.010 | 0.561 ± 0.003 | 1.008 ± 0.006 |

- **The error drops 6% only because the pairs are easier.** Stand-still drops by the same amount.
- **E8 is at stand-still level**, at every step (1, 8, 16). On the visible frames it is slightly worse than doing nothing.
- **Constant velocity beats both** in position up to about step 12. A model that uses the recent motion should at least match it.

**Conclusion:** trimming makes the data cleaner, but it does not fix the actor. The bottleneck is how much the inputs (frame, state) tell the model about future motion. This is consistent with D2: from absolute values the motion is not identifiable, but from history it is.

### Next steps

- [ ] Trimmed E8, 3 seeds (`star-il-simplecnn-latent512-canonq-chunk16-trim-seed{0,1,2}`): target ratio < 1.016 / 1.008.
- [ ] If they also stay at ≈ 1.0: give the actor the recent motion (a(t−1) − a(t−2), or a short history), and compare with constant velocity.
- [ ] Commit the projection script, the keep-ranges file and the trimming code.
- [ ] Send the open questions to Michael.

---

## 2026-11-13

### Today in five lines

- Trimmed E8 (tool-visible frames only, 3 seeds) is **exactly stand-still** (1.001 / 1.006), with a 10× smaller seed spread than E8: every seed = "offset ≈ 0". Trimming stays, but it doesn't fix the actor.
- All chunk runs keep their epoch-1 weights; after that train falls and val rises (memorisation).
- Step 1a (WM state = s(t) + zero-initialised Δ): the state head is now **at copy_state (1.03×)** instead of 23× worse, but never below it on val, while train goes below (memorisation).
- 1a made the actor **15% worse** than stand-still (1.152 / 1.141), like E9. Interpretation: the old head's large initial error only delayed the actor's memorisation; "WM = mild regulariser" was that delay.
- Every head is at or below its trivial baseline (actor vs stand-still, state vs copy, image 4–6× worse than copy_frame), while train loss falls in all of them: the bottleneck is input information, not capacity.

### Trimmed E8: setup

Same as E8 (simple_cnn, latent 512, canonical quaternions, chunk 16, WM on, 3 seeds) + `--keep-ranges configs/keep_ranges/star_il_tool_visible.csv`. Every train statistic (norm stats, quaternion reference, chunk scale) comes from the kept rows only.

**Which numbers compare:**

| Comparison | Same val pairs? | Same normalisation? | Valid metrics |
|---|---|---|---|
| Trimmed E8 vs E8 on trimmed val (`val_metrics_keep_star_il_tool_visible.json`) | yes, 17,692 | no (z std 29.6 → 20.6 mm) | mm, degrees, ratio to stand-still, `next_image_mse` |
| Trimmed E8 vs E8 on untrimmed val | no (17,692 vs 19,346) | no | only ratios to the baselines on each run's own pairs |

`action_mse` (8.35 vs 6.20) and `action_huber` (0.780 vs 0.639) differ only because of the normalisation: not used.

### Result: mean of the 16 steps (3 seeds, mean ± std)

| | pairs | position mm | ÷ stand-still | ÷ const. vel. | rotation ° | ÷ stand-still | ÷ const. vel. |
|---|---|---|---|---|---|---|---|
| E8, untrimmed val | 19,346 | 2.981 ± 0.034 | 0.998 ± 0.011 | 1.158 | 0.578 ± 0.003 | 1.005 ± 0.005 | 0.753 |
| E8, trimmed val | 17,692 | 2.807 ± 0.028 | 1.016 ± 0.010 | 1.145 | 0.561 ± 0.003 | 1.008 ± 0.006 | 0.737 |
| **Trimmed E8** | 17,692 | **2.768 ± 0.003** | **1.001 ± 0.001** | 1.129 | **0.560 ± 0.001** | **1.006 ± 0.002** | 0.735 |
| stand-still (trimmed val) | 17,692 | 2.764 | 1 | – | 0.557 | 1 | – |
| const. velocity (trimmed val) | 17,692 | 2.452 | – | 1 | 0.761 | – | 1 |

**Per step (trimmed E8 ÷ stand-still):** position 1.001 at every step 1–16; rotation 1.004 (step 1) → 1.006 (step 16). Flat across the horizon = the "no change" predictor.

**Per step (trimmed E8 ÷ const. velocity), position:** 2.06 (step 1), 1.59 (step 4), 1.22 (step 8), 1.06 (step 12), 0.97 (step 16). Constant velocity wins up to ~step 14. In rotation the model "wins" at every step (0.91 → 0.68), but only because constant velocity is bad at rotation (noisy angular velocity).

### What it means

**a) The gain over E8 is a move towards zero offsets, not learned motion.**
- Same pairs: 2.768 vs 2.807 mm (−1.4%). With 3 seeds, Welch t ≈ 2.4, df ≈ 2: **not significant**.
- The seed spread drops 10× (±0.003 vs ±0.028 mm). Three independent trainings within 0.1% of stand-still = the same trivial predictor.

**b) Why E8 was 0.998 on untrimmed val but 1.016 on trimmed val (hypothesis, not checked yet).**
- The removed pairs (1,654 on val) are not only the withdrawal tail but also the approach at the start: the tool is off-screen while the robot moves fast (val frames removed: 0.6% at the start, 7.6% at the end).
- They are fast pairs: stand-still ≈ 5.36 mm on them (vs 2.76 mm on the kept pairs), and E8 ≈ 4.84 mm = **0.90× stand-still** there. E8's small edge on untrimmed val comes entirely from these pairs.
- Two explanations: (1) a **constant drift** learned from these phases, which helps on them and hurts on visible pairs; (2) a **"no tool in image" cue**: E8 predicts motion only when the tool is not visible.
- Check: mean predicted offset (per axis, step 16) of E8 **separately on visible and removed pairs**. ≠ 0 on both = constant drift; ≠ 0 only on removed = image cue. Trimmed E8 should be ≈ 0 on both.

**c) The checkpoint is the barely trained model.** After epoch 1 it has seen every pair once and learned only the mean offset ≈ 0 (≈ the median, under Huber).

| | best epoch | val action loss: best → last | train action loss: first → last |
|---|---|---|---|
| E8, seeds 0 / 1 / 2 | 1 / 1 / 2 | 0.667 → 0.95 (+43%) | 0.66 → 0.25 |
| Trimmed E8, seeds 0 / 1 / 2 | 1 / 1 / 1 | 0.780 → 1.15 (+47%) | 0.76 → 0.28 |

- Every epoch that learns something makes val worse. Early stopping on `val_action_output_loss` therefore keeps the epoch-1 weights, which are ≈ the Huber-optimal constant (offset ≈ 0) = stand-still.
- So "E8 ≈ stand-still" means "the selected model has learned nothing beyond the mean offset", not "the model learned to stand still".
- Trimming doesn't change this pattern (same shape, same relative size). It matches D2 and the chunk probe: without history, (frame, state) doesn't identify the future motion.

**d) World model (auxiliary).**

| | next_image_mse (val) | next_state_mse (normalised) |
|---|---|---|
| E8, untrimmed val | 0.00302 ± 0.00021 | 0.0040 ± 0.0012 |
| E8, trimmed val | 0.00306 ± 0.00021 | 0.0041 ± 0.0013 |
| Trimmed E8 | 0.00314 ± 0.00003 | 0.0036 ± 0.0015 |

- Images: same on the same pairs. State: only roughly comparable (trimming moved the state std by up to 5%).
- Both WMs are evaluated at epoch 1, because the checkpoint is chosen on the action loss. Their val image loss keeps improving to ~0.0023 by epoch 10. The val state loss gets worse (0.003 → 0.017–0.027): the state head overfits too.

### Step 1a: Δstate with a skip connection

**What:** next_state = s(t) + Dense(...), with the last Dense zero-initialised, so the untrained model is exactly copy_state (checked before training: max difference 0.0). The loss stays MSE on s(t+1), as in E1 / trimmed E8. Same parameter count (71,975,675). Flag `--state-skip`, commit `2bde5ef`. New reference line in every run: `val_next_state_output_loss_copy` (history.csv) and `next_state_mse_copy` (val_metrics*.json).

**My five lines:**

| | |
|---|---|
| Question | Can the WM state head beat copy_state on val, with a stable curve? |
| Hypothesis | Predicting the small change Δ is easier than the full state. |
| Prediction | 1a gives a smooth curve, close to copy_state but not clearly better. |
| Surprise | Still worse than copy_state → the inputs lack motion info. |
| Change | WM branch only; actor and trimmed data as in C (trimmed E8). |

**Results** (3 seeds, trimmed val, 17,692 pairs; same keep ranges, so identical statistics and comparable normalised losses):

| | Trimmed E8 (absolute head) | **1a (skip)** | Trivial baseline |
|---|---|---|---|
| `next_state_mse` | 0.0036 ± 0.0015 (23× copy) | **0.000160 ± 0.000002 (1.03× copy)** | copy_state 0.000156 |
| actor position ÷ stand-still | 1.001 ± 0.001 | **1.152 ± 0.008** | 1 |
| actor rotation ÷ stand-still | 1.006 ± 0.002 | **1.141 ± 0.028** | 1 |
| actor Huber | 0.780 ± 0.001 | 0.878 ± 0.019 | stand-still 0.779 |
| `next_image_mse` (checkpoint) | 0.00314 ± 0.00003 | 0.00351 ± 0.00019 | copy_frame ≈ 0.0006¹ |
| best epoch | 1 / 1 / 1 | 1 / 1 / 1 | |

¹ From `baselines_canonical_trimmed` (18,457 one-step trimmed pairs; almost the same set as the chunk pairs, not identical).

**What it means:**

1. **The state head stays at copy in every epoch (1.02–1.10×), never below.** Train goes below its own copy level (0.000157): 0.000156 → 0.000130 (−17%) = memorisation. The val rise is small (+2–10% over copy) because the head only outputs the small Δ; it is about half the train drop or less.
2. **~94% of the copy error is `endo360_joint_pos`** (the needle driver). The kuka joints barely move per frame.
3. **The actor got 15% worse** (1.152 / 1.141). With 1a its train loss falls already in epoch 1 (0.72 vs 0.76) and val rises at once (0.86–0.90 vs 0.78), like E9 (WM off). *Interpretation:* the old absolute head's large initial error dominated the gradients reaching the shared encoder in epoch 1, so the actor memorised more slowly. 1a removes that, like E9. So the old "WM = mild regulariser" was a **delay of memorisation**, not useful structure.
4. **Every head is at or below its trivial baseline:** actor vs stand-still, state vs copy_state, image 4–6× worse than copy_frame (≈ 0.0006). Train loss falls in every head, so capacity isn't the limit: **the bottleneck is input information.**

**Note:** 1a and E7 differ in loss (MSE vs scaled Huber), initialisation, actor (chunk) and data (trim), so 1a doesn't prove that the loss scale alone caused E7's failure.

**Verdict:** 1a met its success criterion (stable curve, no rise), and my prediction was right.

### Decision

- **Keep trimming as the default for STAR-IL** (cleaner data, a lower seed spread, no bias from the removed fast phases).
- **Keep the 1a state head** (starts at copy_state, honest reference line, no initial gradient shock).
- I follow Michael's roadmap order: step 1 = the world model (1a done), history = step 4.

### Next steps

- [x] Trimmed E8, 3 seeds: 1.001 / 1.006, below the 1.016 / 1.008 target, but only because it reached stand-still.
- [x] Commit the projection script, the keep-ranges file and the trimming code (`29c0fb8`).
- [x] Step 1a: Δstate with a skip connection, 3 seeds (`2bde5ef`).
- [ ] Check hypothesis b: mean predicted offset of E8 (and trimmed E8) separately on visible and removed val pairs.
- [ ] Send the open questions to Michael.

---

## 2026-11-16

### Today in five lines

- New tool: `--val-every N` (commit `58b7a93`) validates every N batches instead of once per epoch. Epoch 1 is proven bitwise identical to a normal run.
- Run 1a-val500 (trim-skip, seed 0, `--val-every 500`): the best check is at **batch 500 (0.11 epochs), ≈ stand-still**.
- At that point the actor equals an input-free rule, "always predict the mean training offset" (0.7787 vs 0.7785).
- From batch ~1,000, train goes below stand-still and val goes above it at the same moment: memorisation.
- Early stopping can't fix the actor: the bottleneck is the input information.

### The tool: `--val-every N`

**Why:** every run so far had "best = epoch 1", which is the earliest point measured. What happens inside epoch 1 was invisible.

**How (in simple words):**
- The training data becomes an endless reshuffled stream (`.repeat()` after `.batch()`, so each pass keeps its own smaller last batch). Keras treats N batches as one "epoch" (`steps_per_epoch=N`), so checkpointing, early stopping and logging run every N batches. Validation is always the full val split.
- `--epochs`, `--patience` and `--vis-every` stay in real epochs and are converted to checks with ceil(x × batches per epoch / N). At N = 500 (4,582 batches per epoch): 100 epochs = 917 checks, patience 10 = 92 checks.
- New in every run: `batches_seen` / `epochs_seen` columns (a real x-axis), the stand-still reference line `val_action_output_loss_standstill` (0.7795 on trimmed val, equal to `chunk_baselines.json` to 8e-9), and `best_batch` / `best_epoch_equiv` in `config.json`. `best_epoch` keeps its meaning (index of the best check), so old code still works.

**The shuffle finding:**

| | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| normal Keras epochs (`shuffle_order: keras_epochs`) | shuffle #1 | #3 | #5 |
| `--val-every` (`shuffle_order: repeat`) | #1 | #2 | #3 |

- Normal Keras (3.15, `epoch_iterator.py`) creates an extra iterator between epochs, and creating an iterator already draws the next reshuffle. So every second shuffle is silently skipped.
- Test (`scripts/check_val_every_weights.py`, deterministic smokes, `--val-every 11` vs none): **all 86 weight arrays are bitwise identical after epoch 1**, despite the extra mid-epoch validation. After epoch 2 they differ, because of the data order, not the validation. Later epochs therefore use a different, equally random order. This is recorded in `config.json` as `shuffle_order` (old runs: `keras_epochs`).
- **Lesson:** `scripts/check_repeat_batches.py` passed, but it compared against plain tf.data passes, not against Keras. Only the test against the real training found the difference.

**Cost:** at N = 500, 16 s training + 16 s validation per check (≈ 2× the run time).

### Run 1a-val500

Same as 1a seed 0 (simple_cnn, latent 512, canonical quaternions, chunk 16, trimmed, `--state-skip`, WM on) + `--val-every 500 --seed 0`. Same `norm_stats.json` as 1a seed 0.

**My five lines:**

| | | Outcome |
|---|---|---|
| Question | Inside epoch 1, is there a moment where the actor beats stand-still? | Formally at batch 500 (−0.1%), but not a real win (see below). |
| Hypothesis | The general part is learned in the first few hundred batches, then memorisation starts. | **Confirmed, refined:** the general part is the **mean training offset**, not 0. Memorisation starts by batch ~1,000. |
| Prediction | Best around batch 500–1,500, still ≈ stand-still. | **Confirmed** (best at 500). |
| Surprise | Clearly below stand-still at some check. | **Not observed.** |
| Change | Evaluation schedule only. | **Holds** (epoch 1 bitwise identical). |

**The curve** (action loss, Huber in target units; val stand-still 0.7795, train stand-still 0.7629):

| batch | epochs | train action loss² | val action loss | val − stand-still |
|---|---|---|---|---|
| 0¹ | 0 | – | 0.7799 | +0.0005 |
| **500** | **0.11** | 0.7755 | **0.7785** | **−0.0010** |
| 1,000 | 0.22 | 0.7559 | 0.7806 | +0.0011 |
| 1,500 | 0.33 | 0.7444 | 0.7826 | +0.0031 |
| 2,000 | 0.44 | 0.7372 | 0.7925 | +0.013 |
| 3,000 | 0.65 | 0.7106 | 0.8210 | +0.042 |
| 4,500 | 0.98 | 0.6599 | 0.8986 | +0.119 |
| 9,500 | 2.07 | 0.4849 | 1.0164 | +0.237 |
| 46,500 | 10.15 | 0.2646 | 1.1031 | +0.324 |

¹ Not a logged check: an untrained model built with seed 0, evaluated afterwards on the same val pairs (its outputs are ≈ 0, |pred| 0.031).
² Mean over the last 500 batches (Keras resets the training metrics at every check).

**Run facts:** 93 checks (1 + 92 patience), stopped at 46,500 batches (10.15 epochs), 49.7 min.

- **Val is lowest at the first check and rises almost monotonically.** There is no plateau. By the end of epoch 1 it is 12% above stand-still.
- **This explains the old "best = epoch 1".** The 1a seeds had val 0.858–0.896 at epoch 1. This run has 0.8986 at batch 4,500, inside that band (not bitwise comparable: the normal runs were not `--deterministic`). The real minimum was more than 4,000 batches earlier.

### Is the batch-500 advantage real?

Best weights vs stand-still, per pair, on all 17,692 trimmed val pairs (51 episodes):

| Test | Result |
|---|---|
| mean difference (model − stand-still) | −0.0008 (−0.1%) |
| pairs where the model is better | 48.1% |
| episodes where the model is better | 34 / 51 (sign test p = 0.024) |
| 95% bootstrap CI over episodes | [−0.0025, +0.0009]: **includes 0** |
| position error, mean of 16 steps | 2.766 mm vs 2.764 mm stand-still |
| rotation error, mean of 16 steps | 0.557° vs 0.557° |

The sign is consistent per episode, but the size is tiny: not significant once the differences are weighted by size, and gone in physical units. Two caveats:
- **Selection bias:** the best of 93 checks, chosen on the same val set it is measured on. This makes the result optimistic.
- **The logged val loss is batch-weighted, not exact:** the logged val loss of the best check is 0.77849, and re-evaluating the same weights per pair gives 0.77867 (1a seed 0: 0.8792 vs 0.8794). This ≈ 2e-4 gap is not noise. Keras logs the val loss as the unweighted mean of the batch means (our batch size is `None` in the tf.data graph), so the last, smaller batch (28 of 17,692 pairs) is over-weighted. The tests above use the exact per-pair values; the checkpoint was chosen on the logged one. Since step 1b every run also logs the exact value as `val_action_output_huber` (checked: 0.778667 = per-pair to 7e-8). The conclusions don't change.
  *(Corrected 2026-11-16: first written as "numerical noise of the same order". Found when the new per-dimension state metrics did not average to the logged state loss.)*

### What the actor learned

- **At batch 500 it ignores its inputs.** The prediction spread across pairs is 0.059 (target 2.25), and the correlation with the target is 0.026. The input-free rule "always predict the mean training offset" scores the same (0.7785 vs 0.7787 for the model).
- **From batch ~1,000 its predictions depend on the input.** Train loss goes below the best-constant level (≈ 0.76: stand-still 0.7629, mean offset 0.7628, median offset 0.7624 on train). A model that ignores its inputs can't do that. *This is a deduction, not measured directly.* The moment is approximate, because the train loss is averaged over the last 500 batches.
- **At the same moment val gets worse**, so what it learns from the inputs doesn't generalise.
- **Careful wording for conclusions:** "In this setup no generalising use of the inputs is visible." Not "the model never uses its inputs": a small generalising part could be hidden under larger memorisation.

### World model

- **State:** stays at copy level (≈ 0.00016, ratio ≈ 1.0), as in 1a.
- **Image:** val falls from 0.0189 to 0.0023, but copy_frame on trimmed val is ≈ 0.0006³. So it is ≈ 4× worse than copying the frame: it learns to **redraw the scene**, not the dynamics.
- The WM can't provide motion information that isn't in a single snapshot.

³ From `baselines_canonical_trimmed` (0.000607, one-step trimmed pairs; almost the same set as the chunk pairs, not identical).

### What it means

- **The earlier trim-skip runs looked 10–15% worse than stand-still only because they were measured late** (epoch 1). Measured early, the actor ties stand-still.
- **Early stopping can't fix this:** no point on the curve beats the mean-offset rule.
- **The best-checkpoint row of this run in `results.csv` (action Huber 0.7787) is "≈ train-mean offset", not a learned policy.**
- **For 1b and 1c the actor metric won't be informative.** Judge the WM state curve against copy_state, and the image against copy_frame.

### Next steps

- [x] `--val-every N`, commit `58b7a93`, epoch-1 proof.
- [x] Run 1a-val500.
- [ ] Decide: add a "train-mean offset" reference line (labels only) now or with 1b.
- [ ] Step 1b: deeper state head (1–2 more Dense layers), with `--val-every 1000`.
- [ ] Step 1c: depthwise convs, WM branch only.
- [ ] Nice to have: save weights at a few checks and measure prediction spread and correlation on train and val (direct proof of when the inputs start to be used).
- [ ] Skipped: `--val-every 50` diagnostic (batch 0 and 1,000 are both ≈ stand-still, so it can't change the conclusion).

---

## 2026-11-17

Tags: **[M]** = measured on my files, **[H]** = hypothesis (not tested).

### Today in five lines

- Step 1b (deeper state head, K = 2 and 3, seed 0, `--val-every 1000`): **0 of 47 checks below copy_state**; more depth only adds a little memorisation. [M]
- The per-joint metrics show that **endo360 (the needle driver) is 94% of the state copy error** (93.5% in radians): the WM state metric mostly measures one joint. [M]
- endo360 is not commanded by the action, changes on 16% of frames in encoder steps of 0.0006 rad, and contains one-frame glitches (4 events in val = 26% of its copy error). [M]
- The actor controls only the KUKA arm; the arm is smooth and predictable with history (constant velocity 7–14× better than copy). [M]
- Next: evaluate KUKA and endo360 separately; ask Michael whether the WM should keep, balance or exclude endo360.

### Step 1b: code (implemented 2026-11-16)

**Commit 1 (`4f9fdac`): measurements only, no model change**

| What was modified | Why | What it does |
|---|---|---|
| `val_action_output_loss_trainmean` reference line | 11-16: the actor at its best equals "always predict the mean training offset" | The input-free bar "average training offset" (0.7785 on trimmed val), logged next to stand-still |
| Per-joint state metrics, each with its own `_copy` line | The aggregate state loss can hide which joint drives it | One val MSE column per state dimension, and the copy_state value of the same dimension |

**Commit 2 (`c578b50`): `--state-layers K`**

| What was modified | Why | What it does |
|---|---|---|
| K − 1 extra `Dense(128, relu)` layers before `state_change` | Test Michael's hypothesis "one hidden layer is too simple" | Deeper state head; `state_change` stays zero-initialised and the skip stays, so the untrained model is still copy_state (checked: difference 0.0) |
| Own seeds for the new layers, `GlorotUniform(seed = seed + 1000 + i)` | Keras draws initial weights from one global sequence in creation order: new layers would shift the decoder's initialisation | Every other layer starts bitwise identical to K = 1 (checked: 0.0 on all 28 shared weight arrays) |
| `cfg.get("state_layers", 1)`, `cfg.get("seed", 0)` | Old runs have neither key | Old runs stay loadable; +16,512 parameters per extra layer (128 × 128 + 128) |

**Commit 3 (`7eaf2b8`): exact actor metric**

| What was modified | Why | What it does |
|---|---|---|
| `val_action_output_huber` | Keras logs the val loss as the mean of batch means, because `tf.numpy_function` hides the batch size: the last batch of 28 pairs counts 32/28 too much (≈ 0.03%) | Exact per-pair Huber, logged at every check. `MONITOR` unchanged (`val_action_output_loss`) |

### Step 1b: my five lines and outcomes

| | | Outcome |
|---|---|---|
| Question | Does a deeper state head let the WM beat copy_state on val? | **No.** 0 / 47 checks below copy, K = 2 and 3, overall and on endo360. [M] |
| Hypothesis | (Michael) one hidden layer is too simple for s(t) → s(t+1). | **Not supported.** [M] |
| Prediction | Val stays at copy; train goes lower than in 1a. | **Confirmed.** The train ordering is small and from one seed: suggestive, not established. [M] |
| Surprise | Val below copy at some check by more than the noise. | **Not observed.** Best checks 1.003 (K = 3) and 1.004 (K = 2), both at batch 1,000. [M] |
| Change | State head only. | **Holds:** encoder, actor and decoder start bitwise identical to K = 1. [M] |

### Step 1b: results

Seed 0, trimmed val, `--val-every 1000`; 47 checks per run (≈ 43 min). [M]

| batch | val state ÷ copy, K = 1 / 2 / 3 | train state ÷ train copy, K = 1 / 2 / 3 |
|---|---|---|
| 1,000 | 1.005 / 1.004 / **1.003** | 0.87 / 0.97 / 0.97 |
| 5,000 | 1.016 / 1.015 / 1.011 | 0.99 / 0.98 / 0.98 |
| 10,000 | 1.026 / 1.017 / 1.036 | 0.83 / 0.94 / 0.94 |
| 30,000 | 1.045 / 1.053 / 1.069 | 0.77 / 0.83 / 0.81 |
| 46,000 | 1.052 / 1.046 / 1.072 | 0.71 / 0.75 / 0.73 |
| **last ~5 epochs, mean** | **1.051 / 1.060 / 1.072** | **0.84 / 0.82 / 0.80** |

K = 1 is run 1a-val500 (checks every 500 batches, the same batch counts read off). Its values come from the batch-weighted logged loss (≈ 1e-3 relative offset); K = 2 and 3 from the exact per-dimension metrics.

- **The untrained head is exactly copy (1.000); every check after that is worse.** Training only moves the head away from copy, at every depth. [M]
- **More depth = a bit more memorisation:** train goes further below copy (0.84 → 0.80), val rises a bit more (1.051 → 1.072). The differences are about the size of the check-to-check jitter (±0.01–0.02), one seed each. [M; the trend itself is H]
- **Per joint:** `kuka_joint_pos[1]` dips below its copy line in 19% of the K = 2 checks (min 0.963), but its median is 1.025 and it carries 1.5% of the error: noise, not a signal. [M]
- **Actor:** best at the first check (batch 1,000). Exact Huber 0.7813 (K = 2) / 0.7805 (K = 3), just above stand-still (0.7795) and train-mean (0.7785). Then it memorises as in 1a. The state-head depth doesn't change the actor. [M]
- **Image:** 0.0076 → 0.0023, still ≈ 4× worse than copy_frame (0.0006). [M]

### endo360: three data checks

`scripts/look_endo360.py` (tables only, train + val, 460 episodes, never test; ≈ 6 s). Every check is reported on all frames and on the kept frames (`star_il_tool_visible.csv`); the two agree to within a few %. Check 3 uses the exact val pairs of the runs (17,692; the normalised copy MSE per joint matches `config.json` exactly). Outputs: `outputs/analysis/endo360_checks/`.

**Check 1: what are the large jumps?**
- *Question:* the top 1% of val pairs carry 69% of the endo360 error. Real fast movements, or an artefact?
- *Decision rule (written before looking):* |Δ| ≈ 2π or ±π flips → wrap-around; a jump followed within 1–3 frames by an opposite jump of similar size → glitch; neither, and the value stays → real movement.

| | all | kept | run pairs (val) |
|---|---|---|---|
| \|Δ\| p50 / p99 / p99.9 / max (rad) | 0 / 0.020 / 0.052 / 0.126 | 0 / 0.020 / 0.056 / 0.126 | 0 / 0.021 / 0.048 / 0.126 |
| count > 0.1 / > 0.5 / > 1 / > 3 rad (train + val; val) | 66 / 0 / 0 / 0 | 66 / 0 / 0 / 0 | 8 / 0 / 0 / 0 |
| top 1% share of Σ Δ² | 0.70 | 0.71 | **0.69** |

- *Result:* **glitch branch.** [M]
  - No wrap-around: the largest |Δ| is 0.126 rad (0.04 π), and all values lie within ±0.396 rad.
  - The 66 deltas > 0.1 rad are **30 one-frame out-and-back spikes (26 train, 4 val) plus 6 unpaired deltas, in 36 episodes**. Each spike jumps 0.11–0.126 rad toward zero for one frame and comes back (e.g. episode 437: 0.116 → −0.010 → 0.108).
  - Their size is 173–208 encoder ticks (not one fixed count). All are inside the kept windows.
  - They are ≈ 0.2% of all endo360 changes, but the **8 val spike deltas (4 events) carry 25.7% of the val endo360 copy error**.
- "Recording error" is **[H]**: the frame strips (`check1d_*.png`) show no visible tool change, but only at thumbnail size, so they're not conclusive.
- Per episode (median): range 0.17 rad, net change ≈ 0, path ÷ range 2.4. The joint goes back and forth. [M]

**Check 2: is the staircase a slower sampling rate (sample-and-hold)?**
- *Decision rule:* one dominant regular gap between change frames while KUKA changes on almost every frame → sample-and-hold; irregular gaps → the joint really stops and starts.

| gap between change frames (kept) | 1 | 2 | 3 | 4 | 5 | 6–15 | > 15 |
|---|---|---|---|---|---|---|---|
| count | **15,715** | 3,923 | 1,808 | 907 | 622 | 1,978 | 2,435 |

| share of pairs that change (kept) | endo360 | KUKA joints | action dims |
|---|---|---|---|
| | **16%** | 87–99.7% | 99.8–99.9% |

- *Result:* **sample-and-hold not supported.** [M]
  - The most common gap is 1 frame (63% of gaps ≤ 15), and the counts fall off steadily: the sensor reports changes on consecutive frames at 30 Hz.
  - 100% of the non-zero |Δ| are integer multiples of **0.0006059 rad** (0.035°, encoder ticks). The most common step is 1 tick, then 8, 7 and 2.
- It fits neither branch cleanly: the gaps are irregular, but mostly short. **[H]:** slow motion below one tick per frame (< 0.018 rad/s ≈ 1°/s) shows up as irregular single ticks.

**Check 3: is "endo360 = 94% of the error" real, or a normalisation effect?**
- *Decision rule:* endo360 still ≳ 80% of the copy error in radians → real; much smaller → normalisation effect.

| val run pairs (n = 17,692) | share, rad² | share, normalised | RMSE (°/frame) | train std (rad) |
|---|---|---|---|---|
| endo360 | **93.5%** | **94.0%** | 0.27 | 0.140 |
| each KUKA joint | 0.05–1.7% | 0.02–2.7% | 0.006–0.037 | 0.110–0.962 |
| endo360, top 1% pairs removed | 81.9% | 82.8% | 0.15 | |

- *Result:* **real.** [M] endo360's std is in the same range as the KUKA stds (only joint 6, 0.96 rad, is an outlier), so normalisation barely changes the shares. Each endo360 step (0.004–0.05 rad) is 10–40× larger than a KUKA joint's per-frame change.

### What it means

- **The state is two problems.** [M for the properties; H that endo360 is unpredictable from the inputs]
  - The KUKA arm: smooth, commanded (the action is its pose), predictable with history.
  - endo360: rare tick steps, not commanded, back and forth, plus glitches.
- **The WM's state loss and metric are 82–94% endo360,** a joint the actor doesn't control. [M] This hides what the WM does on the arm, and probably pulls its state effort away from it. [H]
- **1a and 1b weren't failing "on the state" in general,** but mostly on one joint that may be unpredictable by construction. [H]
- **Whether the WM can beat copy on the KUKA joints is still open:** KUKA ratio over the checks 1.01–1.34. [M] Two hypotheses [H]: endo360 dominates the loss, or the per-frame KUKA change is tiny compared with the absolute values the network sees.
- **This doesn't change what the actor must learn, but it changes how the WM must be trained and evaluated.**

### Corrections

- "The needle driver moves in sparse, unpredictable bursts": this was **my interpretation of measured statistics** (83% still, top 1% = 69%, lag-1 correlation −0.04), not from a paper. Measured today: rare tick steps going back and forth, plus one-frame glitches that make up a large share of the "bursts".
- "Sample-and-hold staircase" [H, 11-17] → **not supported** (check 2).
- "The large jumps are wrap-arounds" [H] → **no:** one-frame glitches (check 1).
- "94% is a normalisation artefact" [H] → **no:** 93.5% in radians (check 3).
- `instruction.text` is "suture throw" in every frame [M, 8,635 pairs, train episodes 0–19]: no usable subtask labels, although the README mentions subtask-specific instructions.

### Next steps

- [x] 1b, K = 2 and 3, seed 0.
- [x] endo360 checks 1–3.
- [x] Are all glitches the same number of encoder ticks? No: 173–208 ticks. [M]
- [ ] Split the state evaluation: KUKA (vs copy and constant velocity) and endo360 (vs copy), always reported separately.
- [ ] Report endo360 with and without the glitch pairs.
- [ ] Look at the glitch frame strips at full size.
- [ ] Ask Michael: keep, balance or exclude endo360 in the WM state target? Repair the glitches? 1c as planned, or history earlier?
- [ ] Decide whether to commit `scripts/look_endo360.py` as the evidence for the checks.

---

## 2026-11-19

- Analysis scripts moved to `scripts/analysis/`; commands in earlier entries refer to the old paths.
- `check_wm_off_gradient` had been failing since E7 (`build_model` gained a `stats` argument in `6dd76b3`); fixed by passing `stats=None`. Re-run with today's code: same result as for E6 (weights 1, 0, 0: largest gradient difference 0.0 over 28 variables; control 1, 1, 0.1: 24 of 28 differ).
- `dataset_root` bug, since 2026-10-16: the four oldest STAR-IL runs (`star-il-simplecnn-latent512-seed0` = E1 and `…-canonq-seed0/1/2`) saved an absolute `dataset_root`, every other run and `baselines.json` a relative one, and `Path` equality compares the spelling. Effects: the E1 `--eval-only` table never showed the baseline columns, and reloading one of those runs rebuilt the val frame cache (and the next relative run rebuilt it back). No metric was wrong. Fix: `run_config.dataset_root` returns the repo-relative path for an absolute path inside the repo; E1 now shows its baselines, its 37 re-evaluated metrics are identical, and E1 → 1c → E1 rebuilds nothing.
- `check_orientation_error` and `look_static_vs_moving` had also been failing since E7 (same `build_model` signature change, `6dd76b3`); fixed by passing the run's own statistics (`build_model(cfg, stats)`; E7 runs need them). Quick runs on E1 / the E1-E2-E6 runs: `look_static_vs_moving` identical to its 10-27 quick-test CSVs; `check_orientation_error` within about 1e-4 relative of its 10-20 quick-test file (older script version and code, not the fix). E1's full `check_orientation_error.json` (10-21) predates `pos_err_raw`, so `compare_runs` needs it re-run.

---

## 2026-11-20

Tags: **[M]** = measured on my files, **[H]** = hypothesis (not tested).

### Today in five lines

- **History inputs h(t) = [Δs(t), Δa(t−1)] move the actor from ≈ 1.00× to 0.556× stand-still** [0.548, 0.566], 3 seeds. Without history, five step-1 runs (1a / 1b / 1c, 11-16 to 11-18) all had their best check at the first check, at stand-still. [M]
- **The actor uses almost only Δs(t), the last state change.** Zeroing Δs → 0.916×; zeroing Δa → 0.611×; zeroing both → 1.019× (= stand-still). [M]
- It beats constant velocity on the commands (1.185× stand-still) by far: the history is used as an input, not copied as a rule. [M]
- **The WM state head beats copy_state for the first time:** 159 of 160 checks below copy (without history: 0 of 281). KUKA 0.14–0.19× copy, but still worse than constant velocity (0.107×). [M]
- The actor still memorises from ≈ 1.5 epochs on (best check 7–8 of 53; train/val gap 0.73 at the end). Next: does the image add anything once the history is there? → experiment B (11-23).

### Code (three commits + one refactor)

| Commit | What was modified | Why | What it does |
|---|---|---|---|
| `8e07a35` | `CheckMonitor`, `timeline_callback` → `models/callbacks.py` | Refactor (train.py keeps no callbacks) | Unchanged; `CheckMonitor` gets the monitored name as an argument. History columns identical to the earlier smoke run |
| `82c1b15` | `val_action_output_loss_constvel` reference line | The bar for "uses the last command change": the rule offset_i = (i+1)·(a(t−1) − a(t−2)) | Logged next to stand-still and train-mean, through the same `chunk_baselines` functions as `chunk_baselines.json` (so they always agree). **0.9239 = 1.185× stand-still:** extrapolating the last command change is *worse* than standing still |
| `fda6107` | `--history` (data.py D3, `swm.py`, pipeline, train) | 1a–1c: without history, neither branch got past "nothing changes" | h(t) = [s_n(t) − s_n(t−1), a_n(t−1) − a_n(t−2)], normalised units, each dim ÷ its train std (`history_scale` in `norm_stats.json`), **no mean subtracted**, so 0 still means "no change" |
| `5e58cba` | `scripts/analysis/look_history_ablation.py` | Which half of h(t) do the trained models use? | Eval-only: the best weights with Δs, Δa or both zeroed, through the same `evaluate()`; "full" reproduces `val_metrics.json` within 1e-4 relative (GPU) |

Design choices in `--history` [M: from the code]:
- **Both branches** get h(t), appended *last* to the actor's and the WM's concat. No new layer: the first actor Dense, the first state-head Dense and the latent Dense get 15 more input rows. Since Glorot's limit depends on fan-in, these three kernels start different from 1a; every other weight starts identical.
- **The fallback is 0 ("change unknown"), with each change following the rule of its own reference line:**
  - Δs uses `prev_in_window`, the rule of constant-velocity state.
  - Δa uses `chunk_anchors`, the rule of constant-velocity Huber.
  - Fallbacks: Δs 0 / 0 pairs (train / val), Δa 305 / 45 (the t = 1 pairs, where a(t−2) is not in the episode).
- The scale is the std over the *real* changes only (fallback zeros excluded), ddof = 0 like `compute_norm_stats`. Parameters: 71,989,115 (1a: 71,975,675; +13,440 = 15 rows × (512 + 256 + 128)).

### My five lines and outcomes

My exact pre-run wording was not recorded. These lines are **reconstructed on 11-23 from the evidence that existed before the runs**: the probes in `docs/experiment-comparison.md` §5 (D2 and the chunk probe, fitted on train, scored on val) and the design in `fda6107`. Read the "Prediction" row with that in mind.

| | | Outcome |
|---|---|---|
| Question | Does one step of history (the last state change and the last command change) let the network get past "nothing changes", in the actor and in the WM? | **Yes, in both branches.** [M] |
| Hypothesis | (from the probes) Without history the targets are not learnable from one frame + one state; with one step of history they are, and they generalise. Chunk probe: every no-history probe ≈ stand-still or worse; s(t) + Δa → −33%. D2: the last state change makes Δstate predictable (0.42× f = 0) | **Supported.** Without history, 5 of 5 runs are at stand-still at their best check; with history 0.556×. [M] |
| Prediction | Actor clearly below stand-still, about the probe's level (0.68× stand-still on its val pairs), **mostly through Δa** (Δa alone was the strongest single probe input, −25%). WM state below copy_state for the first time | **Level: better than predicted** (0.556×). **WM: confirmed** (159 / 160 checks below copy). **Mechanism: wrong.** The actor relies on **Δs**, not Δa (zeroing Δs → 0.916×, zeroing Δa → 0.611×). The chunk probe never tested Δs as an actor input, so it could not have shown this. [M] |
| Surprise | The actor stays at stand-still even with history (the network can't use what a linear probe can) | **Not observed.** [M] |
| Change | h(t) to both branches, appended last; everything else as 1a (`--state-skip`, trimmed, chunk 16, `--val-every 1000`) | **Holds,** except that the three Dense kernels with 15 more input rows start differently (Glorot fan-in); every other weight starts as in 1a. [M: from the code] |

### Results

Trimmed val (17,692 pairs), `--val-every 1000`, best weights chosen by the actor loss. History: 3 seeds (`star-il-simplecnn-latent512-canonq-chunk16-trim-skip-hist-val1000-seed{0,1,2}`, ≈ 50 min each). Without history: the step-1 runs, seed 0 each (1a `skip-val500`, 1b `sl2` / `sl3`, 1c `mix1` / `mix3`). [M]

| | without history (5 runs) | **with history** (3 seeds) |
|---|---|---|
| **best val ÷ stand-still** | 0.999–1.002 | **0.556** [0.548, 0.566] |
| best check | 1 (batch 500 / 1,000), every run | 7–8 of 53 (1.5–1.7 epochs) |
| step 1: mm / ° | 0.39 / 0.089 (= stand-still) | **0.112 / 0.032** |
| step 8: mm / ° | 2.72 / 0.54 | **1.49 / 0.40** |
| step 16: mm / ° | 4.86 / 0.99 | **3.05 / 0.79** |
| WM state ÷ copy, checks below copy | 1.000–1.008, **0 / 281** | 0.75–0.80, **159 / 160** |

The no-history arm is five different variants with one seed each, not three seeds of one configuration. But all five are at stand-still at their best check, so the gap of 0.44× stand-still is far beyond any seed spread. [M]

### Which half of h(t) does the actor use? (`look_history_ablation`)

Best weights of the 3 history seeds, nothing retrained; Huber ÷ the stand-still value of the same step; mean of 3 seeds (all three seeds within ±0.01). [M]

| val history input | total | step 1 | step 16 | mm (mean) | ° (mean) |
|---|---|---|---|---|---|
| full | **0.557** | 0.258 | 0.670 | 1.57 | 0.42 |
| Δs = 0 | 0.916 | 0.930 | 0.911 | 2.62 | 0.53 |
| Δa = 0 | 0.611 | 0.284 | 0.727 | 1.67 | 0.44 |
| both = 0 | 1.019 | 1.015 | 1.020 | 2.83 | 0.58 |
| *stand-still* | *1* | *1* | *1* | *2.76* | *0.56* |

- **Δs carries most of it:** without it the gain drops from 0.44 to 0.08. Without Δa it drops only to 0.39. [M]
- **Step 1 is almost entirely Δs** (0.26 → 0.93). The current joint velocity tells where the commanded pose goes next. [M; the reason is H]
- **Both zeroed = stand-still exactly.** The image + s(t) path adds nothing the actor can use on val. [M] This was the reason for experiment B (11-23).
- Caveat: zeroing is not removing. Δs = 0 also means "the robot stands still", which the model has seen in training. So a drop measures how much the model *relies* on the input, an upper bound on what it gets from it.

### World model with history

- **State, KUKA arm:** 0.14 / 0.18 / 0.19× copy (seeds 0 / 1 / 2), while constant velocity is 0.107×. The WM now uses the history, but it is still 1.3–1.8× worse than the plain rule "repeat the last change". [M]
- **State, endo360:** 0.79–0.84× copy, the first time below copy. endo360 is 82–94% of the state error (11-17), so it drives the overall 0.75–0.80. **[H]:** endo360 moves in runs of single ticks with gap 1 most often (11-17, check 2), so its last change partly predicts the next one.
- **Image:** 0.0032–0.0035, still ≈ 5.5× worse than copy_frame (0.00059). [M]
- These WM values are at the *actor's* best check (MONITOR), not at the WM's own best.

### What it means

- **1a–1c were not failing on the architecture, but on missing information.** With one frame and one state, neither branch could know the current velocity. With h(t) both branches use it, from the first checks on. [M for the curves; H for the general reading]
- **The actor's information is almost all in s(t) + Δs(t).** Neither the image nor Δa adds much on val. This is the E2 pattern again (10-21), now with relative chunks: the scene is not used. [M]
- **The actor learns more than the rule.** Constant velocity on the commands is *worse* than stand-still (1.185×), while the model reaches 0.556×. It uses Δs as an input, not as an extrapolation. [M]
- **The memorisation is unchanged.** Train goes on to 0.22× stand-still while val climbs back to 0.96×. Is it the image path that memorises? → experiment B. [H, tested 11-23]

### Next steps

- [x] Ablation of the two history halves (`look_history_ablation`, `5e58cba`).
- [ ] Experiment B: state-only actor + history (no image, no WM). Does the image add anything? (done 11-23)
- [ ] WM off with history, 3 seeds (done 11-23).
- [ ] KUKA WM vs constant velocity: why is the WM still worse than "repeat the last change", although it gets that change as an input?

---

## 2026-11-23

Tags: **[M]** = measured on my files, **[H]** = hypothesis (not tested).

Reference ("history runs"): `star-il-simplecnn-latent512-canonq-chunk16-trim-skip-hist-val1000-seed{0,1,2}` (11-20, `fda6107`): full model, chunk 16, trimmed, `--state-skip`, `--history`, `--val-every 1000`. Best val actor loss **0.556× stand-still**.

### Today in five lines

- **B (state-only actor + history): 0.501× stand-still** [0.5007, 0.5015] vs 0.556× [0.548, 0.566] for the full model: **−10%, the seed ranges don't overlap**, and B is better at every chunk step in mm and in degrees. [M]
- B barely memorises: train/val gap 0.05 at the end (full model: 0.73); best check after 26.7 epochs (full model: 1.6). [M]
- **WM off: 0.5564× vs 0.5564× with the WM on.** Same best check, same curves, same memorisation: the WM has no measurable effect on the actor. [M]
- Why: the WM losses are **≈ 0.1% of the total loss** (image 0.00023, state 0.00015, actor ≈ 0.46). In this setup "WM on" was effectively "WM off". [M for the losses; H that the encoder gradients are just as small]
- Both results **replicate E2 and E6 (10-21/22) in the new setup** (relative chunks, history, trimmed). The 10 % gap comes from the image path, not the WM. Information or capacity is still open.

### Code (two commits)

**Commit 1 (`c759cec`): refactor, no number changed**

| What was modified | Why | What it does |
|---|---|---|
| `prepare_stats`, `check_schedule`, `checks`, `DELTA_FLOOR_C`, `CHUNK_FLOOR_C` moved from `scripts/train.py` to `models/run_config.py` (section E) | B must use exactly the same statistics and `--val-every` schedule as train.py, without a copy | One source for the run statistics (keep ranges, quaternion reference, norm stats, Δ / chunk / history scales) and for the schedule. Checked: `check_eval_unchanged compare pre-refactor` UNCHANGED; smoke `norm_stats.json` byte-identical to an older smoke run |

**Commit 2 (`e6ee6b5`): state-only actor for B**

| What was modified | Why | What it does |
|---|---|---|
| `build_state_only_actor(..., action_chunk, history_dim)` | The history runs' actor, minus the image and the WM | concat(s(t), h(t)) → Dense(256, relu) → Dense(16·7) → Reshape((16, 7)), the same order and names as `swm.py`. **34,928 parameters** (full model: 71,989,115). Defaults = the old E2 graph (4,103 parameters; old E2 weights reload with difference 0.0) |
| `train_state_only.py`: `--chunk`, `--history`, `--keep-ranges`, `--val-every` | Same setup as train.py | Same statistics, pairs, targets, reference lines, exact per-pair Huber and final chunk metrics as train.py; frames never loaded |
| `AliasLogs` (new mode only) | A one-output Keras model logs `loss` / `huber`, without the `action_output_` prefix | Also logs `(val_)action_output_loss` / `_huber`, so the W&B panels are the same as train.py's. Old E2 columns unchanged |

Checks for B (GPU) [M]:
- `norm_stats.json` (including `history_scale` and `action_chunk`) is **byte-identical** to the history run's.
- The reference lines are bit-identical: stand-still 0.7795, train-mean 0.7785, constant velocity 0.9239.
- Pairs 146,609 / 17,692, Δa fallbacks 305 / 45.
- On CPU, stand-still and train-mean differ by 1 float32 ULP (6·10⁻⁸): the device changes the summation order of Keras' Huber. Reference lines must be compared on the same device.

`check_orientation_error` is skipped for chunk runs (not chunk-aware yet). So there is no episode bootstrap or paired test for today's comparisons yet: 3 seeds per arm, and the ranges do not overlap.

### B: my five lines and outcomes

| | | Outcome |
|---|---|---|
| Question | Does the image add anything to the actor once it has the history? | **No, it costs ≈ 10%.** [M] |
| Hypothesis | (from the history ablation) the image adds no usable information yet; in the full model it mostly adds memorisation capacity | **Supported, and stronger than expected:** without the image the actor is better, and the memorisation is almost gone. [M] Whether the cause is information or capacity is still open. |
| Prediction | ≈ the same level (≈ 0.556× stand-still), memorises later or less | **Level: wrong** (0.501×). **Memorisation: confirmed**, much later and much less. [M] |
| Surprise | Clearly worse (the image adds information, e.g. the task phase), or clearly better (the image hurts) | **"Clearly better" branch.** [M] |
| Change | No image, no WM; the actor head on s(t) + h(t) only; everything else as in the history runs | **Holds:** same statistics, pairs and reference lines (checked bitwise). [M] |

### B and WM off: results

3 seeds each, trimmed val (17,692 pairs), `--val-every 1000`; mean [min, max]. WM off = the history runs with `--w-next-state 0 --w-next-image 0` (`e6ee6b5`, only these two config keys differ). [M]

| | full, WM on | full, WM off | **B: state-only** |
|---|---|---|---|
| parameters | 72.0 M | 72.0 M | 34,928 |
| **best val ÷ stand-still** | 0.5564 [0.548, 0.566] | 0.5564 [0.550, 0.561] | **0.5011** [0.5007, 0.5015] |
| best val ÷ constant velocity | 0.470 | 0.470 | 0.423 |
| best check (epochs) | 7.3 (1.6) | 7.3 (1.6) | 122 (26.7) |
| train ÷ stand-still at the best check | 0.518 | 0.518 | 0.464 |
| train / val ÷ stand-still at the end | 0.218 / 0.960 | 0.214 / 0.952 | 0.460 / 0.505 |
| training time | ≈ 50 min | ≈ 48 min | ≈ 5 min |

Per chunk step (best weights):

| step | full WM on: mm / ° | full WM off: mm / ° | B: mm / ° | stand-still: mm / ° |
|---|---|---|---|---|
| 1 | 0.112 / 0.0323 | 0.113 / 0.0323 | **0.092 / 0.0171** | 0.390 / 0.089 |
| 8 | 1.490 / 0.396 | 1.491 / 0.397 | **1.394 / 0.366** | 2.711 / 0.539 |
| 16 | 3.052 / 0.793 | 3.045 / 0.793 | **2.924 / 0.741** | 4.837 / 0.982 |

Train vs val over training (seed means, ÷ stand-still):

| epochs seen | full, WM on: val / train | B: val / train |
|---|---|---|
| 0.44 | 0.611 / 0.608 | 0.589 / 0.589 |
| 1.53 | **0.561** / 0.522 | 0.541 / 0.526 |
| 4.58 | 0.745 / 0.386 | 0.519 / 0.496 |
| 10.0 | 0.949 / 0.244 | 0.508 / 0.484 |

- **B is ahead from the 2nd check on:** better at the same number of batches, not only because it trains longer. [M]
- **The largest gain is step-1 rotation (−47%).** Near t the action is mostly forward kinematics of s(t) plus the last command change. The image path adds noise there. [M; the reason is H]
- **At its best check the full model hasn't even fitted train as well as B** (0.518 vs 0.464). Val turns up before the full model has learned what s(t) + h(t) already give. This looks like a shortcut (memorising episodes by their look), not like late overfitting of a converged model. [H]
- **WM on = WM off at every check:** check 1 agrees to 4 decimals in the seed mean, and the per-seed curves differ only in the 3rd–4th decimal. So the WM gradient does reach the encoder (no `stop_gradient`, `swm.py` C), but its effect is negligible. [M]

### Why the WM does nothing: the loss magnitudes

| loss term (val, at the best check) | value × weight | share of the total |
|---|---|---|
| actor Huber | ≈ 0.46 × 1.0 | ≈ 99.9% |
| next image MSE | ≈ 0.0023 × 0.1 | ≈ 0.05% |
| next state MSE | ≈ 0.00015 × 1.0 | ≈ 0.03% |

- The weights 1.0 / 1.0 / 0.1 come from the Open-X reference. Here the loss *magnitudes* differ by three orders, so the weights don't balance anything. [M]
- What the WM learns: image 0.011 → 0.0023, still ≈ 4× worse than copy_frame (0.00059). State 0.00014–0.00016 ≈ copy_state (0.000156). [M]
- With the WM off, its heads stay at their start values: image 0.102; state exactly copy_state, since the state skip is zero-initialised. [M]
- Loss share ≠ gradient share on the encoder: the gradient also depends on how steep each loss is. "The WM gradient on the encoder is negligible" is still **[H]** (consistent with the identical curves, not measured).

### What it means

- **Replication.** E2 (10-21: state-only better, train/val gap ×7 vs ×2) and E6 (10-22: WM off = WM on) hold in the new setup too. That setup has relative chunks, history and trimming, plus a check every 1,000 batches. [M]
- **E2's explanation no longer covers it.** In October: "the target is an absolute pose ≈ FK(angles), so the image isn't needed". The relative chunks were meant to force the actor to use the scene (10-21, next steps). With relative chunks + history, **the image still adds no usable information**; it costs ≈ 10% and drives the memorisation. [M]
- **Three explanations for B's lead:**
  1. Information: the image enables a shortcut, memorising episodes by their look. [H]
  2. The WM shapes the encoder against the actor: **ruled out** by WM off. [M]
  3. Capacity and optimisation: 72 M vs 35 k parameters with the same Adam 3e-4. [H]
- **For the thesis.** The product is encoder + actor, and right now the encoder makes it worse than no encoder. The premise "the WM as an auxiliary task shapes the encoder" **has not been tested yet**: at these loss scales the WM barely acts. E6 = E1 was not "the WM doesn't help", but "the WM wasn't really on". [M for the scales; the conclusion follows from them]

### Corrections

- 10-22, E6: "expected, because the image isn't needed (E2), so a loss that shapes the image encoder can't help". This is incomplete. A WM loss ≈ 0.1% of the total can't shape anything, whatever the task. [M]
- 10-22, E7 motivation: "copy is almost perfect, so the WM had nothing to learn" was half of it. The other half is that the absolute loss is tiny next to the actor loss, so even a WM that learns has almost no weight. [M]

### Next steps

- [x] B: state-only + history, 3 seeds.
- [x] WM off with history, 3 seeds.
- [ ] Gradient norms on the encoder: ‖∇L_actor‖ vs ‖∇L_image‖ vs ‖∇L_state‖, on the saved best weights, about 20 val batches (`look_*`). This turns the [H] above into [M].
- [ ] History model with a zero / constant image: the same 72 M capacity without image information. Separates explanation 1 from 3.
- [ ] A WM that actually acts: balance the WM losses to the actor (e.g. normalise by copy_frame / copy_state, or larger weights). Does it shrink the 0.73 memorisation gap? Design options to be written up before any code.
- [ ] Step (1), paused: `check_orientation_error` for chunk runs, so that `compare_runs` gives the episode bootstrap CI and the paired test for B vs history.
- [ ] Ask Michael: is "state-only + history = 0.50× stand-still" the bar the encoder must beat from now on?
