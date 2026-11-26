import os
from pathlib import Path
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1


# 1. Environment configuration
root = Path(__file__).resolve().parents[1]
os.environ["FLIGHTMARE_PATH"] = str(root)

yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)

n = 4
threads = 4

cfg["env"]["num_envs"] = n
cfg["env"]["num_threads"] = threads
cfg["env"]["render"] = "no"

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)


# 2. Simulation parameters
g = 9.81
dt = 0.02
duration = 0.8
steps = round(duration / dt)

t = np.arange(steps + 1) * dt

obs = np.zeros(
    (n, env.getObsDim()), dtype=np.float32
)

action = np.zeros(
    (n, env.getActDim()), dtype=np.float32
)

reward = np.zeros(n, dtype=np.float32)
done = np.zeros(n, dtype=np.bool_)

extra_info = np.zeros(
    (n, len(env.getExtraInfoNames())),
    dtype=np.float32
)

state = np.zeros((n, 13), dtype=np.float32)

state[:, :3] = [
    [0, 0, 5],
    [1, 0, 5],
    [0, 1, 5],
    [1, 1, 5]
]

state[:, 3] = 1.0

print("========== Environment ==========")
print("Number of environments:", env.getNumOfEnvs())
print("Configured threads:", threads)
print("Observation shape:", obs.shape)
print("Action shape:", action.shape)
print("Reward shape:", reward.shape)
print("Done shape:", done.shape)


# 3. Auxiliary functions
def reset(initial_state):
    if not env.resetState(initial_state, obs):
        raise RuntimeError("Reset failed")

    done.fill(False)


def step():
    if not env.step(
        action, obs, reward, done, extra_info
    ):
        raise RuntimeError("Step failed")

    if not np.isfinite(obs).all():
        raise RuntimeError("Non-finite observation")


def run_vertical_test(inputs, name):
    reset(state)

    action[:] = inputs[:, None]

    history = np.zeros(
        (steps + 1, n, env.getObsDim())
    )

    history[0] = obs

    for k in range(steps):
        step()

        if done.any():
            raise RuntimeError(
                f"Unexpected termination at step {k}"
            )

        history[k + 1] = obs

    z_theory = (
        state[None, :, 2]
        + g * t[:, None]**2 * inputs[None, :]
    )

    vz_theory = (
        2 * g * t[:, None] * inputs[None, :]
    )

    z_error = np.max(
        np.abs(history[:, :, 2] - z_theory),
        axis=0
    )

    vz_error = np.max(
        np.abs(history[:, :, 8] - vz_theory),
        axis=0
    )

    xy_error = np.max(
        np.abs(
            history[:, :, :2]
            - state[None, :, :2]
        ),
        axis=(0, 2)
    )

    print(f"\n========== {name} ==========")

    for i in range(n):
        print(f"\nEnvironment {i}")
        print(f"Action: {inputs[i]:.2f}")
        print(
            f"Final height: "
            f"{history[-1, i, 2]:.8f} m"
        )
        print(
            f"Theoretical height: "
            f"{z_theory[-1, i]:.8f} m"
        )
        print(
            f"Final vertical velocity: "
            f"{history[-1, i, 8]:.8f} m/s"
        )
        print(f"Max height error: {z_error[i]:.3e}")
        print(f"Max velocity error: {vz_error[i]:.3e}")
        print(f"Max XY drift: {xy_error[i]:.3e}")

    np.testing.assert_allclose(
        history[:, :, 2],
        z_theory,
        atol=1e-4,
        rtol=0
    )

    np.testing.assert_allclose(
        history[:, :, 8],
        vz_theory,
        atol=1e-4,
        rtol=0
    )

    assert np.max(xy_error) < 1e-4

    print(f"\n{name}: PASSED")

    return history, z_theory, vz_theory


# 4. Independent environment test
try:
    inputs = np.array(
        [0.0, 0.1, -0.1, -0.5],
        dtype=np.float32
    )

    history, z_theory, vz_theory = (
        run_vertical_test(
            inputs,
            "Independent dynamics"
        )
    )


    # 5. Action permutation test
    permutation = np.array([3, 2, 1, 0])

    swapped, _, _ = run_vertical_test(
        inputs[permutation],
        "Action permutation"
    )

    np.testing.assert_allclose(
        swapped[:, :, 2],
        history[:, permutation, 2],
        atol=1e-4,
        rtol=0
    )

    np.testing.assert_allclose(
        swapped[:, :, 8],
        history[:, permutation, 8],
        atol=1e-4,
        rtol=0
    )

    print("\nAction permutation consistency: PASSED")


    # 6. Independent automatic reset test
    print("\n========== Automatic reset ==========")

    terminal_state = state.copy()

    # Environment 3 starts close to the ground
    terminal_state[3, 2] = 0.1
    terminal_state[3, 9] = -1.0

    reset(terminal_state)

    action.fill(0.0)
    action[3, :] = -0.5

    terminated = False

    for k in range(100):
        step()

        if done.any():
            print("Termination step:", k + 1)
            print("Done flags:", done)
            print("Observations after step:")
            print(obs[:, :3])
            print("Rewards:", reward)

            assert np.array_equal(
                done,
                [False, False, False, True]
            )

            np.testing.assert_allclose(
                obs[:3, :3],
                state[:3, :3],
                atol=1e-4,
                rtol=0
            )

            np.testing.assert_allclose(
                obs[:3, 6:9],
                0.0,
                atol=1e-4,
                rtol=0
            )

            # The terminal environment should have
            # returned its reset observation.
            assert obs[3, 2] > 1.0

            terminated = True
            break

    assert terminated, "No terminal state detected"

    print("Independent automatic reset: PASSED")

finally:
    env.close()


# 7. Plot simulation results
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5)
)

names = [
    "Hover",
    "Ascent",
    "Descent",
    "Free fall"
]

for i in range(n):
    line, = axes[0].plot(
        t, history[:, i, 2],
        label=f"Env {i}: {names[i]}"
    )

    axes[0].plot(
        t, z_theory[:, i],
        "--", color=line.get_color()
    )

    line, = axes[1].plot(
        t, history[:, i, 8],
        label=f"Env {i}: {names[i]}"
    )

    axes[1].plot(
        t, vz_theory[:, i],
        "--", color=line.get_color()
    )

axes[0].set_title("Vertical Position")
axes[0].set_ylabel("Height (m)")

axes[1].set_title("Vertical Velocity")
axes[1].set_ylabel("Velocity (m/s)")

for ax in axes:
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "Flightmare Parallel Environment Independence"
)

fig.tight_layout()

output = root / "learning/13_parallel_independence.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("\nAll parallel independence tests PASSED.")
print("Figure saved:", output)
