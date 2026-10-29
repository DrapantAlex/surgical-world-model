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
