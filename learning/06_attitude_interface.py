import os
from pathlib import Path
from io import StringIO

import numpy as np
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1

root = Path(__file__).resolve().parents[1]
os.environ["FLIGHTMARE_PATH"] = str(root)

yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)

cfg["env"]["num_envs"] = 4
cfg["env"]["num_threads"] = 2
cfg["env"]["render"] = "no"

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)


# ZYX Euler angles to quaternion [w, x, y, z]
def euler_to_quat(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw]) / 2

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])


n = env.getNumOfEnvs()

states = np.zeros((n, 13), dtype=np.float32)
obs = np.zeros((n, 12), dtype=np.float32)

# Roll, Pitch, Yaw in degrees
angles = np.array([
    [10, 0, 0],
    [0, -20, 0],
    [0, 0, -30],
    [15, -25, 45]
], dtype=np.float64)

positions = np.array([
    [0, 0, 5],
    [1, 0, 6],
    [0, 1, 7],
    [-1, -1, 8]
], dtype=np.float32)

try:
    print("========== Test 1: Multi-env reset ==========")

    for i in range(n):
        states[i, 0:3] = positions[i]
        states[i, 3:7] = euler_to_quat(*angles[i])

    assert env.resetState(states, obs)

    np.testing.assert_allclose(
        obs[:, 0:3], positions, atol=1e-6
    )

    print("Independent positions: PASS")

    print("\n========== Test 2: Attitude ==========")

    # Observation order: Yaw, Pitch, Roll
    actual = np.rad2deg(obs[:, 3:6][:, ::-1])

    print("Expected [Roll, Pitch, Yaw]:")
    print(angles)

    print("Actual [Roll, Pitch, Yaw]:")
    print(actual)

    np.testing.assert_allclose(
        actual, angles, atol=1e-4
    )

    print("Non-zero attitude: PASS")

    print("\n========== Test 3: Quaternion sign ==========")

    obs_before = obs.copy()

    states[:, 3:7] *= -1

    assert env.resetState(states, obs)

    np.testing.assert_allclose(
        obs, obs_before, atol=1e-6
    )

    print("q and -q equivalence: PASS")

    print("\n========== Test 4: Pitch boundaries ==========")

    pitch_angles = [89.9, -89.9, 90, -90]

    for i, pitch in enumerate(pitch_angles):
        states[i, 3:7] = euler_to_quat(0, pitch, 0)

    assert env.resetState(states, obs)

    print("Expected pitch:", pitch_angles)
    print("Actual pitch:", np.rad2deg(obs[:, 4]))

    assert np.isfinite(obs).all()

    np.testing.assert_allclose(
        np.rad2deg(obs[:, 4]),
        pitch_angles,
        atol=0.05
    )

    print("Pitch boundaries: PASS")

    print("\nAll attitude interface tests passed.")

finally:
    env.close()
