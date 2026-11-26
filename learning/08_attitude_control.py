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
dyn = quad_cfg["quadrotor_dynamics"]

m = float(dyn["mass"])
l = float(dyn["arm_l"])
kappa = float(dyn["kappa"])

g = 9.81

dt = float(quad_cfg["quadrotor_env"]["sim_dt"])
duration = 3.0
steps = round(duration / dt)

d = l / np.sqrt(2)

# Inertia matrix
J = m * l**2 / 12 * np.diag([4.5, 4.5, 7.0])

# Control allocation matrix
B = np.array([
    [1,      1,      1,      1],
    [d,     -d,     -d,      d],
    [-d,    -d,      d,      d],
    [kappa, -kappa, kappa, -kappa]
], dtype=np.float64)

thrust_mean = m * g / 4
thrust_std = m * g / 2

print("========== Physical parameters ==========")
print("Mass:", m, "kg")
print("Arm length:", l, "m")
print("Inertia matrix:\n", J)
print("Allocation matrix:\n", B)
print("Hover rotor thrust:", thrust_mean, "N")
print("Control frequency:", 1 / dt, "Hz")


# 3. Controller parameters
wn = 4.0
zeta = 1.0

kp = wn**2
kd = 2 * zeta * wn

target_deg = 10.0

cases = [
    ("Roll", 0),
    ("Pitch", 1),
    ("Yaw", 2)
]

state = np.zeros((1, 13), dtype=np.float32)
state[0, 2] = 5.0
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


# 4. Attitude controller
def attitude_controller(target, angles, omega):

    error = target - angles

    # Wrap angle errors to [-pi, pi]
    error = (error + np.pi) % (2 * np.pi) - np.pi

    alpha_des = kp * error - kd * omega

    torque_des = J @ alpha_des

    return torque_des


# 5. Control allocation
def allocate(total_thrust, torque):

    wrench = np.concatenate(
        ([total_thrust], torque)
    )

    rotor_thrusts = np.linalg.solve(B, wrench)

    u_raw = (
        rotor_thrusts - thrust_mean
    ) / thrust_std

    u = np.clip(u_raw, -0.5, 1.0)

    saturated = np.any(
        np.abs(u_raw - u) > 1e-10
    )

    return u, rotor_thrusts, saturated


# 6. Closed-loop simulation
try:
    for name, axis in cases:

        target = np.zeros(3)
        target[axis] = np.deg2rad(target_deg)

        if not env.resetState(state, obs):
            raise RuntimeError("Reset failed")

        done.fill(False)

        time = [0.0]
        angles_log = [obs[0, [5, 4, 3]].copy()]
        omega_log = [obs[0, 9:12].copy()]
        position_log = [obs[0, 0:3].copy()]

        torque_log = []
        thrust_log = []
        action_log = []

        saturation_count = 0

        for k in range(steps):

            angles = obs[0, [5, 4, 3]].astype(
                np.float64
            )

            omega = obs[0, 9:12].astype(
                np.float64
            )

            # Attitude PD controller
            torque = attitude_controller(
                target, angles, omega
            )

            # Fixed total thrust
            total_thrust = m * g

            # Control allocation
            u, thrusts, saturated = allocate(
                total_thrust, torque
            )

            saturation_count += int(saturated)

            action[0, :] = u

            if not env.step(
                action, obs, reward, done, extra_info
            ):
                raise RuntimeError("Step failed")

            if done[0]:
                raise RuntimeError(
                    f"{name}: terminated at "
                    f"{(k + 1) * dt:.2f} s"
                )

            if not np.isfinite(obs).all():
                raise RuntimeError("Invalid observation")

            time.append((k + 1) * dt)

            angles_log.append(
                obs[0, [5, 4, 3]].copy()
            )

            omega_log.append(
                obs[0, 9:12].copy()
            )

            position_log.append(
                obs[0, 0:3].copy()
            )

            torque_log.append(torque.copy())
            thrust_log.append(thrusts.copy())
            action_log.append(u.copy())

        t = np.asarray(time)

        angles_deg = np.rad2deg(
            np.asarray(angles_log)
        )

        omega = np.asarray(omega_log)
        position = np.asarray(position_log)

        torque = np.asarray(torque_log)
        thrusts = np.asarray(thrust_log)
        actions = np.asarray(action_log)

        response = angles_deg[:, axis]

        overshoot = max(
            0.0,
            (np.max(response) - target_deg)
            / target_deg * 100
        )

        outside = np.flatnonzero(
            np.abs(response - target_deg)
            > 0.02 * target_deg
        )

        if len(outside) == 0:
            settling_time = 0.0
        elif outside[-1] == len(t) - 1:
            settling_time = np.nan
        else:
            settling_time = t[outside[-1] + 1]

        other_axes = [
            i for i in range(3) if i != axis
        ]

        cross_axis = np.max(
            np.abs(angles_deg[:, other_axes])
        )

        # Ideal continuous-time critical damping
        theory = target_deg * (
            1 - (1 + wn * t) * np.exp(-wn * t)
        )

        max_theory_error = np.max(
            np.abs(response - theory)
        )

        results[name] = {
            "axis": axis,
            "t": t,
            "angles": angles_deg,
            "omega": omega,
            "position": position,
            "torque": torque,
            "thrusts": thrusts,
            "actions": actions,
            "theory": theory
        }

        print(f"\n========== {name} ==========")

        print("Kp:", kp)
        print("Kd:", kd)

        print("Target angle:", target_deg, "deg")

        print(
            "Final attitude [R, P, Y]:",
            angles_deg[-1], "deg"
        )

        print(
            "Final angle error:",
            f"{target_deg-response[-1]:.8f} deg"
        )

        print(
            "Final angular velocity:",
            omega[-1], "rad/s"
        )

        print(f"Overshoot: {overshoot:.6f} %")
        print(f"Settling time: {settling_time:.4f} s")

        print(
            "Max cross-axis angle:",
            f"{cross_axis:.8f} deg"
        )

        print(
            "Max continuous theory error:",
            f"{max_theory_error:.8f} deg"
        )

        print(
            "Initial torque:",
            torque[0], "N m"
        )

        print(
            "Initial rotor thrusts:",
            thrusts[0], "N"
        )

        print(
            "Action range:",
            f"[{actions.min():.6f}, "
            f"{actions.max():.6f}]"
        )

        print(
            "Saturated steps:",
            saturation_count
        )

        print(
            "Final position:",
            position[-1], "m"
        )

finally:
    env.close()


# 7. Plot results
fig, axes = plt.subplots(
    3, 3, figsize=(15, 11)
)

for row, (name, data) in enumerate(results.items()):

    t = data["t"]
    axis = data["axis"]

    # Attitude response
    ax = axes[row, 0]

    ax.plot(
        t, data["angles"][:, axis],
        label="Flightmare"
    )

    ax.plot(
        t, data["theory"],
        "--", label="Continuous theory"
    )

    ax.axhline(
        target_deg,
        color="black",
        linestyle=":",
        label="Reference"
    )

    ax.axhspan(
        target_deg * 0.98,
        target_deg * 1.02,
        color="gray", alpha=0.15
    )

    ax.set_title(f"{name} Angle")
    ax.set_ylabel("Angle (deg)")

    # Angular velocity
    ax = axes[row, 1]

    for i, label in enumerate(["p", "q", "r"]):
        ax.plot(
            t, data["omega"][:, i],
            label=label
        )

    ax.set_title(f"{name} Angular Velocity")
    ax.set_ylabel("Angular velocity (rad/s)")

    # Rotor thrusts
    ax = axes[row, 2]

    for i in range(4):
        ax.step(
            t[:-1],
            data["thrusts"][:, i],
            where="post",
            label=f"Rotor {i + 1}"
        )

    ax.axhline(
        thrust_mean,
        color="black",
        linestyle=":",
        label="Hover thrust"
    )

    ax.set_title(f"{name} Rotor Thrust Commands")
    ax.set_ylabel("Thrust (N)")

for ax in axes.flat:
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Flightmare Three-axis Attitude PD Control")
fig.tight_layout()

output = root / "learning/08_attitude_control.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("\nFigure saved:", output)
