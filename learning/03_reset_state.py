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

cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = "no"

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)

try:
    obs = np.zeros((1, 12), dtype=np.float32)
    action = np.zeros((1, 4), dtype=np.float32)
    reward = np.zeros(1, dtype=np.float32)
    done = np.zeros(1, dtype=np.bool_)
    extra_info = np.zeros(
        (1, len(env.getExtraInfoNames())),
        dtype=np.float32
    )

    # Initial state: p, q(wxyz), v, w
    state = np.array([[
        0.0, 0.0, 5.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]], dtype=np.float32)

    expected = np.zeros((1, 12), dtype=np.float32)
    expected[0, 2] = 5.0

    np.set_printoptions(precision=8, suppress=True)

    # Test 1: deterministic reset
    print("========== Test 1: Reset ==========")

    assert env.resetState(state, obs)
    print("Observation:", obs)

    np.testing.assert_allclose(
        obs, expected, atol=1e-6, rtol=0
    )
    print("Deterministic reset: PASS")

    # Test 2: quaternion normalization
    print("\n========== Test 2: Quaternion ==========")

    scaled_state = state.copy()
    scaled_state[0, 3] = 2.0

    assert env.resetState(scaled_state, obs)

    np.testing.assert_allclose(
        obs, expected, atol=1e-6, rtol=0
    )
    print("Quaternion normalization: PASS")

    # Test 3: invalid input
    print("\n========== Test 3: Invalid input ==========")

    invalid = state.copy()
    invalid[0, 3:7] = 0.0
    assert not env.resetState(invalid, obs)
    print("Zero quaternion rejected: PASS")

    invalid = state.copy()
    invalid[0, 0] = np.nan
    assert not env.resetState(invalid, obs)
    print("NaN rejected: PASS")

    invalid = np.zeros((1, 12), dtype=np.float32)
    assert not env.resetState(invalid, obs)
    print("Wrong dimension rejected: PASS")

    # Test 4: hover simulation
    print("\n========== Test 4: Hover ==========")

    assert env.resetState(state, obs)

    obs_before = obs.copy()

    assert env.step(
        action, obs, reward, done, extra_info
    )

    print("Before:", obs_before)
    print("After: ", obs)
    print("Change:", obs - obs_before)
    print("Reward:", reward)
    print("Done:", done)

    assert np.isfinite(obs).all()
    assert not done[0]

    print("Hover step execution: PASS")

    # Test 5: reset after simulation
    print("\n========== Test 5: Reset again ==========")

    assert env.resetState(state, obs)

    np.testing.assert_allclose(
        obs, expected, atol=1e-6, rtol=0
    )

    print("Observation:", obs)
    print("Repeated reset: PASS")

    print("\nAll interface tests passed.")

finally:
    env.close()
