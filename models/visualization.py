"""Training visualisation: GT/PRED timeline for TensorBoard and W&B."""
import cv2
import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras

class WorldModelTimelineCallback(keras.callbacks.Callback):
    """Professor's FastWorldModelVisCallback, adapted to one Open-H episode in NumPy.
    GT:   [t_0]   [t_1] ... [t_k]
    PRED: [empty] [t_1] ... [t_k]
    """

    def __init__(self, log_dir, inputs, targets, instruction_text, wandb_run=None,   # NEW argument
                 num_frames=8, start=0, stride=1, every=25):
        super().__init__()
        self.wandb_run = wandb_run 
        self.file_writer = tf.summary.create_file_writer(str(log_dir / "model_visualizations"))
        self.instruction_text = instruction_text
        self.every = every
        # CHANGED: frames chosen by index, so the timeline is a real sequence
        self.idx = start + stride * np.arange(num_frames)
        assert self.idx[-1] < len(targets["next_image_output"]), "timeline goes past the episode"
        self.ref_inputs = {k: v[self.idx] for k, v in inputs.items()}
        self.ref_targets = {k: v[self.idx] for k, v in targets.items()}

    @staticmethod
    def _label(img, text, pos, color, scale=0.4):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def create_timeline_image(self, pred_next, pred_actions):
        to_u8 = lambda a: np.ascontiguousarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        true_actions = self.ref_targets["action_output"]

        # first column: the starting frame, and an empty slot in the PRED row
        img_t0 = to_u8(self.ref_inputs["image_input"][0])
        self._label(img_t0, f"GT [t_{self.idx[0]}]", (10, 20), (0, 255, 0))
        blank = np.zeros_like(img_t0)
        self._label(blank, "[empty]", (10, 20), (150, 150, 150))
        gt_row, pred_row = [img_t0], [blank]

        # one column per chosen moment t: true and predicted frame t+1
        for i, t in enumerate(self.idx):
            gt = to_u8(self.ref_targets["next_image_output"][i])
            pr = to_u8(pred_next[i])
            self._label(gt, f"GT [t_{t + 1}]", (10, 20), (0, 255, 0))
            self._label(gt, f"Act: {true_actions[i][0]:.2f}, {true_actions[i][1]:.2f}",
                        (10, 210), (255, 255, 255))
            self._label(pr, f"PRED [t_{t + 1}]", (10, 20), (255, 100, 100))
            self._label(pr, f"Act: {pred_actions[i][0]:.2f}, {pred_actions[i][1]:.2f}",
                        (10, 210), (255, 255, 255))
            gt_row.append(gt)
            pred_row.append(pr)

        grid = np.concatenate([np.concatenate(gt_row, axis=1),
                               np.concatenate(pred_row, axis=1)], axis=0)
        header = np.zeros((60, grid.shape[1], 3), np.uint8)
        self._label(header, "Task Instruction:", (10, 20), (200, 200, 200), 0.5)
        self._label(header, self.instruction_text, (10, 45), (0, 255, 255), 0.6)
        return np.concatenate([header, grid], axis=0)

    def on_epoch_end(self, epoch, logs=None):
        last = epoch == self.params["epochs"] - 1
        if epoch % self.every != 0 and not last:      # CHANGED: every 25 epochs, not every epoch
            return
        pred_actions, _, pred_next = self.model.predict(self.ref_inputs, verbose=0)
        img = self.create_timeline_image(pred_next, pred_actions)
        with self.file_writer.as_default():
            tf.summary.image("Episode_Timeline_Rollout", img[None], step=epoch)
            self.file_writer.flush()
        if self.wandb_run is not None:                                               # CHANGED
            self.wandb_run.log({"timeline": wandb.Image(img, caption=f"epoch {epoch}")},
                               step=epoch)