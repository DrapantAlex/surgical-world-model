"""State-only actor (E2): the actor branch of swm.py without the image (and the zero language).

Question it answers: does the image encoder add information over the robot state?
On STAR-IL the state is the joint angles and the action the end-effector pose, so
state -> action is close to forward kinematics and needs no image at all.
"""
from tensorflow import keras
from tensorflow.keras import layers

HIDDEN = 256   # same width as the actor head in swm.py (section C)


def build_state_only_actor(state_dim, action_dim, hidden=HIDDEN):
    state_input = keras.Input(shape=(state_dim,), name="state_input")

    # the actor head of swm.py section C, fed with the state only
    a_x = layers.Dense(hidden, activation="relu")(state_input)
    action_output = layers.Dense(action_dim, name="action_output")(a_x)

    return keras.Model(inputs=state_input, outputs=action_output, name="state_only_actor")
