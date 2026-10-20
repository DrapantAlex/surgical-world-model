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