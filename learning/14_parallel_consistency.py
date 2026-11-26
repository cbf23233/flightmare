import os
from pathlib import Path
from io import StringIO
from time import perf_counter

import numpy as np
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1


# 1. Environment configuration
root = Path(__file__).resolve().parents[1]
os.environ["FLIGHTMARE_PATH"] = str(root)
os.environ["OMP_DYNAMIC"] = "FALSE"

yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)


# 2. Experiment parameters
n = 100
thread_counts = [1, 2, 4, 8, 12]

seed = 42
dt = 0.02
duration = 10.0
steps = round(duration / dt)

tolerance = 1e-6

rng = np.random.default_rng(seed)

print("========== Experiment configuration ==========")
print("Number of environments:", n)
print("Thread counts:", thread_counts)
print("Random seed:", seed)
print("Simulation dt:", dt)
print("Simulation duration:", duration)
print("Simulation steps:", steps)
print("Transitions per configuration:", n * steps)


# 3. Generate identical initial states
state = np.zeros((n, 13), dtype=np.float32)

state[:, 0:2] = rng.uniform(
    -2.0, 2.0, size=(n, 2)
)

state[:, 2] = rng.uniform(
    4.0, 6.0, size=n
)

# Identity quaternion
state[:, 3] = 1.0

# Initial linear velocity
state[:, 7:10] = rng.uniform(
    -0.1, 0.1, size=(n, 3)
)

# Initial angular velocity
state[:, 10:13] = rng.uniform(
    -0.05, 0.05, size=(n, 3)
)

# Force some environments to terminate early
state[:10, 2] = 0.1
state[:10, 9] = -1.0


# 4. Pre-generate identical action sequences
actions = rng.uniform(
    -0.01, 0.01,
    size=(steps, n, 4)
).astype(np.float32)

# Force the first 10 environments to fall
actions[:10, :10, :] = -0.5

print("Initial state shape:", state.shape)
print("Action sequence shape:", actions.shape)


# 5. Run simulation with a specified thread count
def run_simulation(threads):
    cfg["env"]["num_envs"] = n
    cfg["env"]["num_threads"] = threads
    cfg["env"]["seed"] = seed
    cfg["env"]["render"] = "no"

    stream = StringIO()
    yaml.dump(cfg, stream)

    env = QuadrotorEnv_v1(
        stream.getvalue(), False
    )

    try:
        env.setSeed(seed)

        obs_dim = env.getObsDim()
        act_dim = env.getActDim()
        extra_dim = len(env.getExtraInfoNames())

        assert act_dim == actions.shape[2]

        obs = np.zeros(
            (n, obs_dim), dtype=np.float32
        )

        reward = np.zeros(
            n, dtype=np.float32
        )

        done = np.zeros(
            n, dtype=np.bool_
        )

        extra_info = np.zeros(
            (n, extra_dim), dtype=np.float32
        )

        obs_history = np.zeros(
            (steps + 1, n, obs_dim),
            dtype=np.float32
        )

        reward_history = np.zeros(
            (steps, n), dtype=np.float32
        )

        done_history = np.zeros(
            (steps, n), dtype=np.bool_
        )

        extra_history = np.zeros(
            (steps, n, extra_dim),
            dtype=np.float32
        )

        if not env.resetState(state.copy(), obs):
            raise RuntimeError("Reset failed")

        obs_history[0] = obs

        start = perf_counter()

        for k in range(steps):
            if not env.step(
                actions[k],
                obs,
                reward,
                done,
                extra_info
            ):
                raise RuntimeError(
                    f"Step failed: {k}"
                )

            if not np.isfinite(obs).all():
                raise RuntimeError(
                    f"Invalid observation: {k}"
                )

            if not np.isfinite(reward).all():
                raise RuntimeError(
                    f"Invalid reward: {k}"
                )

            if not np.isfinite(extra_info).all():
                raise RuntimeError(
                    f"Invalid extra info: {k}"
                )

            obs_history[k + 1] = obs
            reward_history[k] = reward
            done_history[k] = done
            extra_history[k] = extra_info

        elapsed = perf_counter() - start

        print(f"\n========== {threads} thread(s) ==========")
        print("Elapsed time:", f"{elapsed:.6f} s")
        print(
            "Transitions per second:",
            f"{n * steps / elapsed:.2f}"
        )
        print(
            "Total terminal events:",
            np.count_nonzero(done_history)
        )
        print(
            "Environments terminated:",
            np.count_nonzero(
                np.any(done_history, axis=0)
            )
        )

        return {
            "obs": obs_history,
            "reward": reward_history,
            "done": done_history,
            "extra": extra_history,
            "elapsed": elapsed
        }

    finally:
        env.close()


# 6. Single-thread reference
reference = run_simulation(1)

assert reference["done"].any(), (
    "Automatic reset was not exercised"
)


# 7. Compare numerical results
def max_error(a, b):
    if a.size == 0:
        return 0.0

    return float(np.max(np.abs(a - b)))


def compare(reference, result, threads):
    print(f"\n---------- Comparison: 1 vs {threads} ----------")

    groups = {
        "Position": slice(0, 3),
        "Attitude": slice(3, 6),
        "Linear velocity": slice(6, 9),
        "Angular velocity": slice(9, 12)
    }

    for name, indices in groups.items():
        error = max_error(
            reference["obs"][:, :, indices],
            result["obs"][:, :, indices]
        )

        print(f"{name:20s}: {error:.10e}")

    reward_error = max_error(
        reference["reward"],
        result["reward"]
    )

    extra_error = max_error(
        reference["extra"],
        result["extra"]
    )

    done_equal = np.array_equal(
        reference["done"],
        result["done"]
    )

    exact_equal = all(
        np.array_equal(
            reference[key], result[key]
        )
        for key in ["obs", "reward", "done", "extra"]
    )

    obs_error = max_error(
        reference["obs"],
        result["obs"]
    )

    numerical_equal = (
        obs_error <= tolerance
        and reward_error <= tolerance
        and extra_error <= tolerance
        and done_equal
    )

    print("Reward error:", f"{reward_error:.10e}")
    print("Extra info error:", f"{extra_error:.10e}")
    print("Done flags identical:", done_equal)
    print("Exact equality:", exact_equal)
    print("Numerical equality:", numerical_equal)

    # Locate the first differing observation
    different = np.argwhere(
        reference["obs"] != result["obs"]
    )

    if len(different):
        k, i, j = different[0]

        print("First observation difference:")
        print("  Step:", k)
        print("  Environment:", i)
        print("  Observation index:", j)
        print("  Reference:", reference["obs"][k, i, j])
        print("  Current:", result["obs"][k, i, j])

    speedup = (
        reference["elapsed"] / result["elapsed"]
    )

    print("Preliminary speedup:", f"{speedup:.4f}x")

    return exact_equal, numerical_equal, speedup


# 8. Multi-thread experiments
summary = [(1, True, True, 1.0)]

for threads in thread_counts[1:]:
    result = run_simulation(threads)

    exact, numerical, speedup = compare(
        reference, result, threads
    )

    summary.append(
        (threads, exact, numerical, speedup)
    )

    del result


# 9. Summary
print("\n========== Consistency summary ==========")

print(
    f"{'Threads':>8}"
    f"{'Exact':>12}"
    f"{'Numerical':>12}"
    f"{'Speedup':>12}"
)

for threads, exact, numerical, speedup in summary:
    print(
        f"{threads:8d}"
        f"{str(exact):>12}"
        f"{str(numerical):>12}"
        f"{speedup:12.4f}"
    )

if all(row[2] for row in summary):
    print("\nAll parallel consistency tests PASSED.")
else:
    raise RuntimeError(
        "Parallel consistency test FAILED"
    )
