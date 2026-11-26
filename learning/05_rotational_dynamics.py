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

with open(root / "flightlib/configs/quadrotor_env.yaml") as f:
    quad_cfg = yaml.load(f)

cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = "no"

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)


# 2. Physical parameters
params = quad_cfg["quadrotor_dynamics"]

m = float(params["mass"])
l = float(params["arm_l"])
kappa = float(params["kappa"])

g = 9.81
d = l / np.sqrt(2)

J = m * l**2 / 12 * np.diag([4.5, 4.5, 7.0])

B = np.array([
    [1, 1, 1, 1],
    [d, -d, -d, d],
    [-d, -d, d, d],
    [kappa, -kappa, kappa, -kappa]
])

print("========== Inertia matrix ==========")
print(J)

print("\n========== Allocation matrix ==========")
print(B)


# 3. Simulation parameters
dt = float(quad_cfg["quadrotor_env"]["sim_dt"])
duration = 0.4
steps = round(duration / dt)
time = np.arange(steps + 1) * dt

delta = 0.02

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


# 4. Experiment definitions
experiments = {
    "Roll": {
        "action": [delta, -delta, -delta, delta],
        "axis": 0,
        "angle_idx": 5,
        "rate_idx": 9,
    },
    "Pitch": {
        "action": [-delta, -delta, delta, delta],
        "axis": 1,
        "angle_idx": 4,
        "rate_idx": 10,
    },
    "Yaw": {
        "action": [delta, -delta, delta, -delta],
        "axis": 2,
        "angle_idx": 3,
        "rate_idx": 11,
    },
}


# 5. Run one experiment
def simulate(u):
    assert env.resetState(initial_state, obs)

    action[0, :] = u

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


# 6. Run all experiments
results = {}

try:
    for name, exp in experiments.items():

        u = np.asarray(exp["action"], dtype=np.float64)
        axis = exp["axis"]

        rotor_thrusts = m * g / 4 * (1 + 2 * u)
        force_torque = B @ rotor_thrusts

        torque = force_torque[1:]
        alpha = np.linalg.solve(J, torque)

        rate_theory = alpha[axis] * time
        angle_theory = 0.5 * alpha[axis] * time**2

        history = simulate(u)

        rate_sim = history[:, exp["rate_idx"]]
        angle_sim = history[:, exp["angle_idx"]]

        other_axes = [i for i in range(3) if i != axis]

        results[name] = {
            "history": history,
            "rate_sim": rate_sim,
            "rate_theory": rate_theory,
            "angle_sim": angle_sim,
            "angle_theory": angle_theory,
        }

        print(f"\n========== {name} ==========")
        print("Action:", u)
        print("Rotor thrusts:", rotor_thrusts)
        print("Total thrust:", force_torque[0], "N")
        print("Body torque:", torque, "N*m")
        print("Angular acceleration:", alpha, "rad/s^2")

        print(
            f"Final rate (sim): "
            f"{rate_sim[-1]:.8f} rad/s"
        )
        print(
            f"Final rate (theory): "
            f"{rate_theory[-1]:.8f} rad/s"
        )
        print(
            f"Final angle (sim): "
            f"{np.rad2deg(angle_sim[-1]):.8f} deg"
        )
        print(
            f"Final angle (theory): "
            f"{np.rad2deg(angle_theory[-1]):.8f} deg"
        )

        print(
            f"Max rate error: "
            f"{np.max(np.abs(rate_sim-rate_theory)):.8e} rad/s"
        )
        print(
            f"Max angle error: "
            f"{np.max(np.abs(angle_sim-angle_theory)):.8e} rad"
        )

        print(
            "Max cross-axis angular rate:",
            np.max(np.abs(history[:, 9:12][:, other_axes])),
            "rad/s"
        )

        print("Final position:", history[-1, :3])

finally:
    env.close()


# 7. Plot results
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for row, (name, data) in enumerate(results.items()):

    axes[row, 0].plot(
        time,
        np.rad2deg(data["angle_sim"]),
        label="Simulation"
    )

    axes[row, 0].plot(
        time,
        np.rad2deg(data["angle_theory"]),
        "--",
        label="Theory"
    )

    axes[row, 1].plot(
        time,
        data["rate_sim"],
        label="Simulation"
    )

    axes[row, 1].plot(
        time,
        data["rate_theory"],
        "--",
        label="Theory"
    )

    axes[row, 2].plot(
        time,
        data["rate_sim"] - data["rate_theory"],
        label="Rate error"
    )

    axes[row, 0].set_title(f"{name}: Angle")
    axes[row, 1].set_title(f"{name}: Angular Rate")
    axes[row, 2].set_title(f"{name}: Rate Error")

    axes[row, 0].set_ylabel("Angle (deg)")
    axes[row, 1].set_ylabel("Rate (rad/s)")
    axes[row, 2].set_ylabel("Error (rad/s)")

    for ax in axes[row]:
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()

fig.suptitle("Flightmare Rotational Dynamics")
fig.tight_layout()

output = root / "learning/05_rotational_dynamics.png"

fig.savefig(output, dpi=200)

print(f"\nFigure saved: {output}")

if "agg" == plt.get_backend().lower():
    plt.close(fig)
else:
    plt.show()
