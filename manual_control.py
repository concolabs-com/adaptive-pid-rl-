"""
Manual MuJoCo viewer for the 2-wheeled car.
Keyboard controls:
  W / S  — forward / backward (both motors)
  A / D  — turn left / right (differential)
  Space  — stop
  Q/Esc  — quit

Camera: mouse drag to orbit, scroll to zoom (MuJoCo viewer built-in).
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = os.path.join(os.path.dirname(__file__), "envs", "assets", "car_model.xml")
SPEED = 0.8  # motor magnitude for forward/back
TURN = 0.6  # motor magnitude for turning

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Print actuator indices so we know what we're controlling
print("Actuators:")
for i in range(model.nu):
    print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

# Shared state for key presses (viewer callback runs in another thread)
keys = {"w": False, "s": False, "a": False, "d": False, "quit": False}


def key_callback(keycode):
    # MuJoCo viewer passes GLFW key codes
    # GLFW key codes: W=87, S=83, A=65, D=68, Space=32, Q=81, Esc=256
    if keycode == 87:
        keys["w"] = not keys["w"]  # W toggle
    elif keycode == 83:
        keys["s"] = not keys["s"]  # S toggle
    elif keycode == 65:
        keys["a"] = not keys["a"]  # A toggle
    elif keycode == 68:
        keys["d"] = not keys["d"]  # D toggle
    elif keycode == 32:  # Space — full stop
        for k in ("w", "s", "a", "d"):
            keys[k] = False
    elif keycode in (81, 256):  # Q or Esc
        keys["quit"] = True


def compute_ctrl():
    """Differential drive: left=ctrl[0], right=ctrl[1]."""
    left = right = 0.0

    if keys["w"]:
        left += SPEED
        right += SPEED
    if keys["s"]:
        left -= SPEED
        right -= SPEED
    if keys["a"]:
        left -= TURN
        right += TURN
    if keys["d"]:
        left += TURN
        right -= TURN

    return np.clip([left, right], -1.0, 1.0)


print("\nControls: W=fwd  S=back  A=left  D=right  Space=stop  Q=quit")
print("Keys TOGGLE on press. Press same key again to release.\n")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # Start with a nice camera angle
    viewer.cam.distance = 0.8
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 135

    mujoco.mj_resetData(model, data)

    while viewer.is_running() and not keys["quit"]:
        step_start = time.time()

        data.ctrl[:] = compute_ctrl()
        mujoco.mj_step(model, data)

        viewer.sync()

        # Maintain ~60 Hz
        elapsed = time.time() - step_start
        time.sleep(max(0.0, 1 / 60 - elapsed))

print("Viewer closed.")
