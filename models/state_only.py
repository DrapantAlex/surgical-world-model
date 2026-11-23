"""State-only actor (E2): the actor branch of swm.py without the image (and the zero language).

Question it answers: does the image encoder add information over the robot state?
On STAR-IL the state is the joint angles and the action the end-effector pose, so
state -> action is close to forward kinematics and needs no image at all.

NEW (B): the same actor head as the history runs, minus the image: optional action chunk
(E8) and history input [Δs(t), Δa(t-1)] / scale (data.py D3). Defaults give the old E2 model.
"""
from tensorflow import keras
from tensorflow.keras import layers

HIDDEN = 256   # same width as the actor head in swm.py (section C)


def build_state_only_actor(state_dim, action_dim, hidden=HIDDEN, action_chunk=None,
                           history_dim=0):
    # NEW (B): action_chunk = k -> k offsets from a(t-1), shape (k, action_dim), as swm.py (E8);
    #   history_dim = len(h(t)) = state_dim + action_dim -> one more Input, appended LAST to
    #   the actor's concat (the order of swm.py). 0 / None: the old E2 graph, unchanged.
    if history_dim < 0:
        raise ValueError(f"history_dim must be >= 0, got {history_dim}")
    state_input = keras.Input(shape=(state_dim,), name="state_input")
    history = ([keras.Input(shape=(history_dim,), name="history_input")]   # NEW (B)
               if history_dim > 0 else [])

    # the actor head of swm.py section C, fed with the state only (+ h(t))
    x = layers.Concatenate()([state_input] + history) if history else state_input
    a_x = layers.Dense(hidden, activation="relu")(x)
    if action_chunk is None:
        action_output = layers.Dense(action_dim, name="action_output")(a_x)
    else:
        # NEW (B): one Dense for the whole chunk, then (k, action_dim); same names as swm.py
        a_x = layers.Dense(action_chunk * action_dim, name="action_chunk")(a_x)
        action_output = layers.Reshape((action_chunk, action_dim), name="action_output")(a_x)

    inputs = [state_input] + history if history else state_input   # old E2: a single tensor
    return keras.Model(inputs=inputs, outputs=action_output, name="state_only_actor")
