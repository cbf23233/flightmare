import io
import os
import time
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1


# ============================================================
# 1. Environment configuration
# ============================================================

root = Path(os.environ["FLIGHTMARE_PATH"])
yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)

cfg["env"]["seed"] = 1
cfg["env"]["scene_id"] = 2
cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = True

stream = io.StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)

num_envs = env.getNumOfEnvs()
obs_dim = env.getObsDim()
act_dim = env.getActDim()
extra_dim = len(env.getExtraInfoNames())

print("\n========== Environment ==========")
print("Number of environments:", num_envs)
print("Observation dimension:", obs_dim)
print("Action dimension:", act_dim)
print("Extra info dimension:", extra_dim)

obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
act = np.zeros((num_envs, act_dim), dtype=np.float32)
reward = np.zeros(num_envs, dtype=np.float32)
done = np.zeros(num_envs, dtype=np.bool_)
extra = np.zeros((num_envs, extra_dim), dtype=np.float32)

# Position, quaternion (wxyz), velocity, angular velocity
state = np.array(
    [[0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.float32,
)

dt = 0.02

phases = [
    ("Hover", 0.00, 1.0),
    ("Accelerate upward", 0.05, 1.0),
    ("Decelerate upward", -0.05, 1.0),
    ("Hover at new altitude", 0.00, 1.0),
]

connected = False

try:
    # ========================================================
    # 2. Reset and connect Unity
    # ========================================================

    if not env.resetState(state, obs):
        raise RuntimeError("Failed to reset initial state")

    print("\n========== Initial state ==========")
    print(obs[0])

    print("\n========== Connecting Unity ==========")

    connected = env.connectUnity()

    if not connected:
        raise RuntimeError("Unity connection failed")

    print("Unity connection successful")

    # ========================================================
    # 3. Dynamics and rendering
    # ========================================================

    sim_time = 0.0
    wall_start = time.perf_counter()
    total_steps = 0
    render_step_times = []

    for name, command, duration in phases:
        print(f"\n========== {name} ==========")
        print(f"Action: {command:+.3f}")

        act.fill(command)
        steps = round(duration / dt)

        for k in range(steps):
            step_start = time.perf_counter()

            ok = env.step(act, obs, reward, done, extra)

            elapsed = time.perf_counter() - step_start
            render_step_times.append(elapsed)

            if not ok:
                raise RuntimeError("Environment step failed")

            if not np.all(np.isfinite(obs)):
                raise RuntimeError("Non-finite observation")

            if np.any(done):
                raise RuntimeError("Unexpected terminal state")

            total_steps += 1
            sim_time = total_steps * dt

            # Print every 0.2 simulation seconds
            if (k + 1) % 10 == 0:
                print(
                    f"t={sim_time:5.2f} s | "
                    f"z={obs[0, 2]:8.5f} m | "
                    f"vz={obs[0, 8]:+8.5f} m/s | "
                    f"step={elapsed * 1000:7.2f} ms"
                )

            # Limit simulation speed to approximately real time
            remaining = dt - elapsed

            if remaining > 0:
                time.sleep(remaining)

    wall_time = time.perf_counter() - wall_start

    # ========================================================
    # 4. Results
    # ========================================================

    print("\n========== Final results ==========")
    print("Total steps:", total_steps)
    print(f"Simulation time: {sim_time:.3f} s")
    print(f"Wall time: {wall_time:.3f} s")
    print(f"Real-time factor: {sim_time / wall_time:.3f}")
    print(f"Average step time: "
          f"{np.mean(render_step_times) * 1000:.3f} ms")

    print(f"Final altitude: {obs[0, 2]:.6f} m")
    print(f"Final vertical velocity: {obs[0, 8]:.6f} m/s")

    expected_z = 5.981

    print(f"Expected altitude: {expected_z:.6f} m")
    print(f"Altitude error: "
          f"{abs(obs[0, 2] - expected_z):.6e} m")

    if abs(obs[0, 2] - expected_z) > 0.01:
        raise RuntimeError("Altitude verification failed")

    if abs(obs[0, 8]) > 0.01:
        raise RuntimeError("Velocity verification failed")

    print("\nPASS: Dynamics and Unity integration")

    # Keep the final scene visible
    time.sleep(3)

finally:
    if connected:
        env.disconnectUnity()
        print("Unity disconnected")

    env.close()
    print("Environment closed")
