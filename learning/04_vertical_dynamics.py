import os
from pathlib import Path
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1


# 1. Project configuration
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


# 2. Simulation parameters
dt = 0.02
duration = 0.8
steps = round(duration / dt)
g = 9.81

obs = np.zeros((1, 12), dtype=np.float32)
action = np.zeros((1, 4), dtype=np.float32)
reward = np.zeros(1, dtype=np.float32)
done = np.zeros(1, dtype=np.bool_)
extra_info = np.zeros(
    (1, len(env.getExtraInfoNames())),
    dtype=np.float32
)

initial_state = np.array([[
    0, 0, 5,
    1, 0, 0, 0,
    0, 0, 0,
    0, 0, 0
]], dtype=np.float32)


# 3. Run one experiment
def simulate(u):
    assert env.resetState(initial_state, obs)

    action.fill(u)

    history = [obs[0].copy()]

    for k in range(steps):
        assert env.step(
            action, obs, reward, done, extra_info
        )

        if done[0]:
            raise RuntimeError(
                f"Unexpected termination at step {k + 1}"
            )

        if not np.isfinite(obs).all():
            raise RuntimeError("Invalid observation")

        history.append(obs[0].copy())

    return np.asarray(history)


# 4. Run all experiments
experiments = {
    "Free fall": -0.5,
    "Hover": 0.0,
    "Ascent": 0.1,
    "Descent": -0.1,
}

results = {}
time = np.arange(steps + 1) * dt

try:
    for name, u in experiments.items():
        history = simulate(u)

        acceleration = 2 * g * u

        z_theory = 5 + 0.5 * acceleration * time**2
        vz_theory = acceleration * time

        z_sim = history[:, 2]
        vz_sim = history[:, 8]

        results[name] = {
            "history": history,
            "z_theory": z_theory,
            "vz_theory": vz_theory,
        }

        print(f"\n========== {name} ==========")
        print(f"Action: {u:.2f}")
        print(f"Final z (sim):    {z_sim[-1]:.8f} m")
        print(f"Final z (theory): {z_theory[-1]:.8f} m")
        print(f"Final vz (sim):    {vz_sim[-1]:.8f} m/s")
        print(f"Final vz (theory): {vz_theory[-1]:.8f} m/s")
        print(f"Max z error:  {np.max(np.abs(z_sim-z_theory)):.8e} m")
        print(f"Max vz error: {np.max(np.abs(vz_sim-vz_theory)):.8e} m/s")
        print(f"Mean acceleration: {vz_sim[-1]/duration:.8f} m/s^2")

finally:
    env.close()


# 5. Plot simulation and theoretical results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for name, data in results.items():
    history = data["history"]

    line, = axes[0, 0].plot(
        time, history[:, 2], label=name
    )
    color = line.get_color()

    axes[0, 0].plot(
        time, data["z_theory"],
        "--", color=color, alpha=0.7
    )

    axes[0, 1].plot(
        time, history[:, 8],
        color=color, label=name
    )
    axes[0, 1].plot(
        time, data["vz_theory"],
        "--", color=color, alpha=0.7
    )

    axes[1, 0].plot(
        time,
        history[:, 2] - data["z_theory"],
        color=color, label=name
    )

    axes[1, 1].plot(
        time,
        history[:, 8] - data["vz_theory"],
        color=color, label=name
    )

titles = [
    "Vertical Position",
    "Vertical Velocity",
    "Position Error",
    "Velocity Error",
]

ylabels = [
    "z (m)",
    "vz (m/s)",
    "Position error (m)",
    "Velocity error (m/s)",
]

for ax, title, ylabel in zip(
    axes.flat, titles, ylabels
):
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "Flightmare Vertical Dynamics\n"
    "Solid: Simulation | Dashed: Theory"
)

fig.tight_layout()

output = root / "learning/04_vertical_dynamics.png"
fig.savefig(output, dpi=200)

print(f"\nFigure saved: {output}")

plt.show()
