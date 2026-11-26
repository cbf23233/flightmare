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
duration = 4.0
steps = round(duration / dt)

d = l / np.sqrt(2)

J = m * l**2 / 12 * np.diag([4.5, 4.5, 7.0])

B = np.array([
    [1,      1,      1,      1],
    [d,     -d,     -d,      d],
    [-d,    -d,      d,      d],
    [kappa, -kappa, kappa, -kappa]
], dtype=np.float64)

thrust_mean = m * g / 4
thrust_std = m * g / 2


# 3. Controller parameters
k_att = 2.0
k_rate = 8.0

kp_z = 4.0
kd_z = 4.0

z_target = 5.0

att_target = np.deg2rad(
    np.array([10.0, 0.0, 0.0])
)

cases = [
    "Fixed Thrust",
    "Height PD",
    "Height PD + Tilt Compensation"
]

state = np.zeros((1, 13), dtype=np.float32)
state[0, 2] = 5.0
state[0, 3] = 1.0

obs = np.zeros(
    (1, env.getObsDim()),
    dtype=np.float32
)

action = np.zeros(
    (1, env.getActDim()),
    dtype=np.float32
)

reward = np.zeros(1, dtype=np.float32)
done = np.zeros(1, dtype=np.bool_)

extra_info = np.zeros(
    (1, len(env.getExtraInfoNames())),
    dtype=np.float32
)

results = {}


# 4. Attitude cascade controller
def attitude_controller(target, angles, omega):

    error = target - angles

    error = (
        error + np.pi
    ) % (2 * np.pi) - np.pi

    omega_des = k_att * error

    alpha_des = k_rate * (
        omega_des - omega
    )

    torque_des = J @ alpha_des

    return torque_des


# 5. Height controller
def height_controller(z, vz, angles, mode):

    roll, pitch, yaw = angles

    tilt_factor = (
        np.cos(roll) * np.cos(pitch)
    )

    az_des = (
        kp_z * (z_target - z)
        - kd_z * vz
    )

    if mode == "Fixed Thrust":

        total_thrust = m * g

    elif mode == "Height PD":

        total_thrust = m * (g + az_des)

    elif mode == "Height PD + Tilt Compensation":

        # Guard against near-singular compensation
        if tilt_factor < 0.5:
            raise RuntimeError(
                "Tilt angle too large"
            )

        total_thrust = (
            m * (g + az_des)
            / tilt_factor
        )

    else:
        raise ValueError(mode)

    return total_thrust, az_des, tilt_factor


# 6. Control allocation
def allocate(total_thrust, torque):

    wrench = np.concatenate(
        ([total_thrust], torque)
    )

    rotor_thrusts = np.linalg.solve(
        B, wrench
    )

    u_raw = (
        rotor_thrusts - thrust_mean
    ) / thrust_std

    u = np.clip(u_raw, -0.5, 1.0)

    saturated = np.any(
        np.abs(u_raw - u) > 1e-10
    )

    return u, rotor_thrusts, saturated


# 7. Closed-loop simulation
try:
    for name in cases:

        if not env.resetState(state, obs):
            raise RuntimeError("Reset failed")

        done.fill(False)

        time = [0.0]
        position_log = [obs[0, :3].copy()]
        velocity_log = [obs[0, 6:9].copy()]
        angles_log = [
            obs[0, [5, 4, 3]].copy()
        ]

        thrust_log = []
        vertical_thrust_log = []
        rotor_log = []
        action_log = []
        az_des_log = []

        saturation_count = 0

        for k in range(steps):

            z = float(obs[0, 2])
            vz = float(obs[0, 8])

            angles = obs[
                0, [5, 4, 3]
            ].astype(np.float64)

            omega = obs[
                0, 9:12
            ].astype(np.float64)

            # Attitude control
            torque = attitude_controller(
                att_target, angles, omega
            )

            # Height control
            total_thrust, az_des, tilt_factor = (
                height_controller(
                    z, vz, angles, name
                )
            )

            # Control allocation
            u, rotor_thrusts, saturated = (
                allocate(total_thrust, torque)
            )

            saturation_count += int(saturated)

            action[0, :] = u

            if not env.step(
                action,
                obs,
                reward,
                done,
                extra_info
            ):
                raise RuntimeError("Step failed")

            if done[0]:
                raise RuntimeError(
                    f"{name}: terminated at "
                    f"{(k + 1) * dt:.2f} s"
                )

            if not np.isfinite(obs).all():
                raise RuntimeError(
                    "Invalid observation"
                )

            time.append((k + 1) * dt)

            position_log.append(
                obs[0, :3].copy()
            )

            velocity_log.append(
                obs[0, 6:9].copy()
            )

            angles_log.append(
                obs[0, [5, 4, 3]].copy()
            )

            thrust_log.append(total_thrust)

            vertical_thrust_log.append(
                total_thrust * tilt_factor
            )

            rotor_log.append(
                rotor_thrusts.copy()
            )

            action_log.append(u.copy())
            az_des_log.append(az_des)

        t = np.asarray(time)
        position = np.asarray(position_log)
        velocity = np.asarray(velocity_log)

        angles_deg = np.rad2deg(
            np.asarray(angles_log)
        )

        thrust = np.asarray(thrust_log)

        vertical_thrust = np.asarray(
            vertical_thrust_log
        )

        rotor_thrusts = np.asarray(rotor_log)
        actions = np.asarray(action_log)

        height = position[:, 2]
        height_error = z_target - height

        results[name] = {
            "t": t,
            "position": position,
            "velocity": velocity,
            "angles": angles_deg,
            "thrust": thrust,
            "vertical_thrust": vertical_thrust,
            "rotor_thrusts": rotor_thrusts,
            "actions": actions,
            "az_des": np.asarray(az_des_log)
        }

        print(f"\n========== {name} ==========")

        print(
            "Final attitude [R, P, Y]:",
            angles_deg[-1], "deg"
        )

        print(
            "Final height:",
            f"{height[-1]:.8f} m"
        )

        print(
            "Final height error:",
            f"{height_error[-1]:.8f} m"
        )

        print(
            "Max absolute height error:",
            f"{np.max(np.abs(height_error)):.8f} m"
        )

        print(
            "Height RMSE:",
            f"{np.sqrt(np.mean(height_error**2)):.8f} m"
        )

        print(
            "Final vertical velocity:",
            f"{velocity[-1, 2]:.8f} m/s"
        )

        print(
            "Total thrust range:",
            f"[{thrust.min():.8f}, "
            f"{thrust.max():.8f}] N"
        )

        print(
            "Final total thrust:",
            f"{thrust[-1]:.8f} N"
        )

        print(
            "Final vertical thrust estimate:",
            f"{vertical_thrust[-1]:.8f} N"
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


# 8. Theoretical steady-state values
tilt_factor_ss = (
    np.cos(att_target[0])
    * np.cos(att_target[1])
)

z_error_ss = (
    g / kp_z
    * (1 / tilt_factor_ss - 1)
)

thrust_ss = m * g / tilt_factor_ss

print("\n========== Theoretical values ==========")

print("Hover thrust:", m * g, "N")

print(
    "Tilt compensated thrust:",
    thrust_ss, "N"
)

print(
    "Height PD steady-state error:",
    z_error_ss, "m"
)

print(
    "Height PD steady-state height:",
    z_target - z_error_ss, "m"
)

print(
    "Compensated steady-state height:",
    z_target, "m"
)


# 9. Plot results
fig, axes = plt.subplots(
    3, 2, figsize=(13, 12)
)

for name, data in results.items():

    t = data["t"]
    pos = data["position"]
    vel = data["velocity"]
    angles = data["angles"]

    axes[0, 0].plot(
        t, pos[:, 2], label=name
    )

    axes[0, 1].plot(
        t, vel[:, 2], label=name
    )

    axes[1, 0].plot(
        t, angles[:, 0], label=name
    )

    axes[1, 1].step(
        t[:-1],
        data["thrust"],
        where="post",
        label=name
    )

    axes[2, 0].step(
        t[:-1],
        data["vertical_thrust"],
        where="post",
        label=name
    )

    axes[2, 1].plot(
        t,
        z_target - pos[:, 2],
        label=name
    )

axes[0, 0].axhline(
    z_target,
    color="black",
    linestyle="--",
    label="Reference"
)

axes[0, 0].axhline(
    z_target - z_error_ss,
    color="gray",
    linestyle=":",
    label="PD steady-state theory"
)

axes[1, 0].axhline(
    10.0,
    color="black",
    linestyle="--",
    label="Reference"
)

axes[1, 1].axhline(
    m * g,
    color="black",
    linestyle=":",
    label="Hover thrust"
)

axes[2, 0].axhline(
    m * g,
    color="black",
    linestyle=":",
    label="Gravity"
)

titles = [
    ("Height Response", "Height (m)"),
    ("Vertical Velocity", "Velocity (m/s)"),
    ("Roll Response", "Angle (deg)"),
    ("Total Thrust Command", "Thrust (N)"),
    ("Estimated Vertical Thrust", "Thrust (N)"),
    ("Height Error", "Error (m)")
]

for ax, (title, ylabel) in zip(
    axes.flat, titles
):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(
    "Flightmare Height and Attitude Control"
)

fig.tight_layout()

output = (
    root
    / "learning/10_height_attitude_control.png"
)

fig.savefig(output, dpi=180)
plt.close(fig)

print("\nFigure saved:", output)
