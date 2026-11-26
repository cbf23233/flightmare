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

cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = "no"

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)


# 2. Simulation parameters
g = 9.81
dt = 0.02
duration = 10.0
steps = round(duration / dt)

z_initial = 5.0
z_target = 6.0

wn = 2.0

cases = [
    ("Underdamped", 0.3),
    ("Critically damped", 1.0),
    ("Overdamped", 1.5)
]

state = np.zeros((1, 13), dtype=np.float32)
state[0, 2] = z_initial
state[0, 3] = 1.0

obs = np.zeros((1, env.getObsDim()), dtype=np.float32)
action = np.zeros((1, env.getActDim()), dtype=np.float32)
reward = np.zeros(1, dtype=np.float32)
done = np.zeros(1, dtype=np.bool_)

extra_info = np.zeros(
    (1, len(env.getExtraInfoNames())),
    dtype=np.float32
)

results = {}


# 3. Closed-loop simulation
try:
    for name, zeta in cases:

        kp = wn ** 2
        kd = 2 * zeta * wn

        if not env.resetState(state, obs):
            raise RuntimeError("Reset failed")

        done.fill(False)

        time = [0.0]
        height = [float(obs[0, 2])]
        velocity = [float(obs[0, 8])]
        actions = []
        attitude = [obs[0, 3:6].copy()]

        for k in range(steps):

            z = float(obs[0, 2])
            vz = float(obs[0, 8])

            # PD acceleration controller
            az_des = kp * (z_target - z) - kd * vz

            # Acceleration to normalized action
            u = az_des / (2 * g)

            # Prevent negative commanded rotor thrust
            u = np.clip(u, -0.5, 1.0)

            action.fill(u)

            if not env.step(
                action, obs, reward, done, extra_info
            ):
                raise RuntimeError("Step failed")

            if done[0]:
                raise RuntimeError(
                    f"{name}: episode terminated at "
                    f"t={(k + 1) * dt:.2f} s"
                )

            if not np.isfinite(obs).all():
                raise RuntimeError("Invalid observation")

            time.append((k + 1) * dt)
            height.append(float(obs[0, 2]))
            velocity.append(float(obs[0, 8]))
            actions.append(float(u))
            attitude.append(obs[0, 3:6].copy())

        t = np.asarray(time)
        z = np.asarray(height)
        vz = np.asarray(velocity)
        u = np.asarray(actions)
        angles = np.asarray(attitude)

        # Performance metrics
        amplitude = z_target - z_initial

        overshoot = max(
            0.0,
            (np.max(z) - z_target) / amplitude * 100
        )

        outside = np.flatnonzero(
            np.abs(z - z_target) > 0.02 * amplitude
        )

        if len(outside) == 0:
            settling_time = 0.0
        elif outside[-1] == len(t) - 1:
            settling_time = np.nan
        else:
            settling_time = t[outside[-1] + 1]

        results[name] = {
            "t": t,
            "z": z,
            "vz": vz,
            "u": u
        }

        print(f"\n========== {name} ==========")
        print(f"Natural frequency: {wn:.2f} rad/s")
        print(f"Damping ratio: {zeta:.2f}")
        print(f"Kp: {kp:.4f}")
        print(f"Kd: {kd:.4f}")
        print(f"Final height: {z[-1]:.8f} m")
        print(f"Final velocity: {vz[-1]:.8f} m/s")
        print(f"Final error: {z_target-z[-1]:.8f} m")
        print(f"Overshoot: {overshoot:.4f} %")
        print(f"Settling time: {settling_time:.4f} s")
        print(f"Action range: [{u.min():.6f}, {u.max():.6f}]")
        print(
            "Max attitude:",
            f"{np.max(np.abs(np.rad2deg(angles))):.8f} deg"
        )

finally:
    env.close()


# 4. Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for name, data in results.items():

    t = data["t"]
    z = data["z"]
    vz = data["vz"]
    u = data["u"]

    axes[0, 0].plot(t, z, label=name)
    axes[0, 1].plot(t, vz, label=name)
    axes[1, 0].step(t[:-1], u, where="post", label=name)
    axes[1, 1].plot(t, z_target-z, label=name)

axes[0, 0].axhline(
    z_target, color="black",
    linestyle="--", label="Reference"
)

axes[0, 0].axhspan(
    z_target-0.02, z_target+0.02,
    color="gray", alpha=0.15
)

axes[0, 0].set_title("Height Response")
axes[0, 0].set_ylabel("Height (m)")

axes[0, 1].set_title("Vertical Velocity")
axes[0, 1].set_ylabel("Velocity (m/s)")

axes[1, 0].set_title("Control Action")
axes[1, 0].set_ylabel("Normalized Action")

axes[1, 1].set_title("Height Error")
axes[1, 1].set_ylabel("Error (m)")

for ax in axes.flat:
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Flightmare Height PD Control")
fig.tight_layout()

output = root / "learning/07_height_control.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("\nFigure saved:", output)
