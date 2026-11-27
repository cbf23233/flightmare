import io
import os
import time
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1

# ============================================================
# 1. Experiment configuration
# ============================================================

ROOT = Path(os.environ["FLIGHTMARE_PATH"])

FPS = 50
DT = 1.0 / FPS
DURATION = 4.0
STEPS = round(DURATION * FPS)

BASE_POSITION = np.array(
    [0.0, 0.0, 5.0], dtype=np.float32
)

PHASES = [
    ("X translation", 0, 3.0),
    ("Y translation", 1, 3.0),
    ("Z translation", 2, 3.0),
    ("Roll rotation", 3, 30.0),
    ("Pitch rotation", 4, 30.0),
    ("Yaw rotation", 5, 90.0),
]

# ============================================================
# 2. Quaternion conversion
# ============================================================

def euler_to_quaternion(roll, pitch, yaw):
    """Convert ZYX Euler angles to quaternion (wxyz)."""

    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)

    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)

    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)

# ============================================================
# 3. Create environment
# ============================================================

yaml = YAML()

with open(ROOT / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)

cfg["env"]["seed"] = 1
cfg["env"]["scene_id"] = 2
cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = True

stream = io.StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)

obs = np.zeros(
    (1, env.getObsDim()), dtype=np.float32
)

state = np.zeros((1, 13), dtype=np.float32)

frame_id = 0
connected = False
render_times = []

max_position_error = 0.0
max_attitude_error = 0.0

# ============================================================
# 4. Set pose and render
# ============================================================

def render_pose(position, euler_deg):
    global frame_id
    global max_position_error
    global max_attitude_error

    state.fill(0)

    state[0, 0:3] = position

    state[0, 3:7] = euler_to_quaternion(
        *np.deg2rad(euler_deg)
    )

    if not env.resetState(state, obs):
        raise RuntimeError("resetState failed")

    if not np.all(np.isfinite(obs)):
        raise RuntimeError("Non-finite observation")

    position_error = np.max(
        np.abs(obs[0, 0:3] - position)
    )

    # Observation order: yaw, pitch, roll
    actual_euler = np.rad2deg(obs[0, 3:6][::-1])

    attitude_error = np.max(
        np.abs(actual_euler - euler_deg)
    )

    max_position_error = max(
        max_position_error, float(position_error)
    )

    max_attitude_error = max(
        max_attitude_error, float(attitude_error)
    )

    frame_id += 1

    start = time.perf_counter()

    if not env.render(frame_id):
        raise RuntimeError(
            f"Render failed at frame {frame_id}"
        )

    render_times.append(
        time.perf_counter() - start
    )

# ============================================================
# 5. Run experiment
# ============================================================

try:
    state[0, 0:3] = BASE_POSITION
    state[0, 3] = 1.0

    if not env.resetState(state, obs):
        raise RuntimeError("Initial reset failed")

    print("Connecting Unity...")

    connected = env.connectUnity()

    if not connected:
        raise RuntimeError("Unity connection failed")

    print("Unity connected")
    print(f"Target render frequency: {FPS} Hz")

    render_pose(BASE_POSITION, np.zeros(3))
    time.sleep(2)

    wall_start = time.perf_counter()

    for name, axis, amplitude in PHASES:

        print(f"\n========== {name} ==========")

        for k in range(1, STEPS + 1):

            loop_start = time.perf_counter()

            t = k * DT

            scale = 0.5 * (
                1.0 - np.cos(
                    2.0 * np.pi * t / DURATION
                )
            )

            position = BASE_POSITION.copy()
            euler_deg = np.zeros(3)

            if axis < 3:
                position[axis] += amplitude * scale
            else:
                euler_deg[axis - 3] = amplitude * scale

            render_pose(position, euler_deg)

            if k % FPS == 0:
                print(
                    f"t={t:.1f} s | "
                    f"position={np.round(obs[0, :3], 3)} | "
                    f"RPY(deg)="
                    f"{np.round(np.rad2deg(obs[0, 3:6][::-1]), 3)}"
                )

            elapsed = time.perf_counter() - loop_start

            if elapsed < DT:
                time.sleep(DT - elapsed)

        print(f"{name}: completed")

        time.sleep(1)

    wall_time = time.perf_counter() - wall_start

    # ========================================================
    # 6. Results
    # ========================================================

    print("\n========== Final results ==========")

    print(f"Total rendered frames: {frame_id}")
    print(f"Motion duration: {len(PHASES) * DURATION:.3f} s")
    print(f"Wall time including pauses: {wall_time:.3f} s")

    print(
        f"Average render call time: "
        f"{np.mean(render_times) * 1000:.3f} ms"
    )

    print(
        f"P95 render call time: "
        f"{np.percentile(render_times, 95) * 1000:.3f} ms"
    )

    print(
        f"Maximum render call time: "
        f"{np.max(render_times) * 1000:.3f} ms"
    )

    print(
        f"Maximum position error: "
        f"{max_position_error:.6e} m"
    )

    print(
        f"Maximum attitude error: "
        f"{max_attitude_error:.6e} deg"
    )

    print("Final observation:", obs[0])

    assert frame_id == 1 + len(PHASES) * STEPS
    assert max_position_error < 1e-5
    assert max_attitude_error < 1e-3

    print("\nPASS: Independent pose rendering calls")

    time.sleep(3)

finally:
    if connected:
        env.disconnectUnity()
        print("Unity disconnected")

    env.close()
    print("Environment closed")
