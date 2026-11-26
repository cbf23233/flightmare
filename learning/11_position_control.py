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
duration = 15.0
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
k_pos = np.array([0.4, 0.4, 1.0])
k_vel = np.array([1.6, 1.6, 4.0])

k_att = 2.0
k_rate = 8.0

max_vel_xy = 2.0
max_vel_z = 1.0

max_acc_xy = 3.0
max_acc_z = 3.0

max_tilt = np.deg2rad(20.0)
max_rate = np.deg2rad([90.0, 90.0, 45.0])


# 4. Test cases
cases = [
    ("X Position", [2, 0, 5], 0),
    ("Y Position", [0, 2, 5], 0),
    ("XYZ Position", [2, 2, 6], 0),
    ("XYZ with Yaw", [2, 2, 6], 45)
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


# 5. Vector norm limitation
def limit_norm(vector, maximum):

    norm = np.linalg.norm(vector)

    if norm > maximum:
        return vector * maximum / norm

    return vector.copy()


# 6. Position controller
def position_controller(target, position):

    vel_des = k_pos * (target - position)

    vel_des[:2] = limit_norm(
        vel_des[:2], max_vel_xy
    )

    vel_des[2] = np.clip(
        vel_des[2],
        -max_vel_z,
        max_vel_z
    )

    return vel_des


# 7. Velocity controller
def velocity_controller(vel_des, velocity):

    acc_des = k_vel * (
        vel_des - velocity
    )

    acc_des[:2] = limit_norm(
        acc_des[:2], max_acc_xy
    )

    acc_des[2] = np.clip(
        acc_des[2],
        -max_acc_z,
        max_acc_z
    )

    return acc_des


# 8. Acceleration to attitude and thrust
def acceleration_mapping(acc_des, yaw_des, angles):

    acc = acc_des.copy()

    # Limit desired total tilt angle
    vertical = g + acc[2]

    if vertical <= 0:
        raise RuntimeError(
            "Invalid vertical thrust demand"
        )

    horizontal_limit = (
        vertical * np.tan(max_tilt)
    )

    acc[:2] = limit_norm(
        acc[:2], horizontal_limit
    )

    # Desired thrust direction
    force_direction = (
        acc + np.array([0.0, 0.0, g])
    )

    b3 = (
        force_direction
        / np.linalg.norm(force_direction)
    )

    bx, by, bz = b3

    roll_des = np.arcsin(
        np.clip(
            bx * np.sin(yaw_des)
            - by * np.cos(yaw_des),
            -1.0, 1.0
        )
    )

    pitch_des = np.arctan2(
        bx * np.cos(yaw_des)
        + by * np.sin(yaw_des),
        bz
    )

    att_des = np.array([
        roll_des,
        pitch_des,
        yaw_des
    ])

    # Actual-attitude tilt compensation
    roll, pitch, yaw = angles

    tilt_factor = (
        np.cos(roll) * np.cos(pitch)
    )

    if tilt_factor < 0.5:
        raise RuntimeError(
            "Actual tilt angle too large"
        )

    total_thrust = (
        m * (g + acc[2])
        / tilt_factor
    )

    return att_des, total_thrust, acc


# 9. Attitude and rate controller
def attitude_controller(att_des, angles, omega):

    error = att_des - angles

    error = (
        error + np.pi
    ) % (2 * np.pi) - np.pi

    omega_des = k_att * error

    omega_des = np.clip(
        omega_des,
        -max_rate,
        max_rate
    )

    alpha_des = k_rate * (
        omega_des - omega
    )

    torque = J @ alpha_des

    return torque, omega_des


# 10. Control allocation
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

    u = np.clip(
        u_raw, -0.5, 1.0
    )

    saturated = np.any(
        np.abs(u_raw - u) > 1e-10
    )

    return u, saturated


# 11. Closed-loop simulation
try:
    for name, target, yaw_deg in cases:

        target = np.asarray(
            target, dtype=np.float64
        )

        yaw_des = np.deg2rad(yaw_deg)

        if not env.resetState(state, obs):
            raise RuntimeError("Reset failed")

        done.fill(False)

        time = [0.0]

        position_log = [obs[0, :3].copy()]
        velocity_log = [obs[0, 6:9].copy()]

        attitude_log = [
            obs[0, [5, 4, 3]].copy()
        ]

        vel_des_log = []
        acc_des_log = []
        att_des_log = []
        rate_des_log = []
        thrust_log = []
        action_log = []

        saturation_count = 0

        for k in range(steps):

            position = obs[
                0, :3
            ].astype(np.float64)

            velocity = obs[
                0, 6:9
            ].astype(np.float64)

            angles = obs[
                0, [5, 4, 3]
            ].astype(np.float64)

            omega = obs[
                0, 9:12
            ].astype(np.float64)

            # Position outer loop
            vel_des = position_controller(
                target, position
            )

            # Velocity inner loop
            acc_des = velocity_controller(
                vel_des, velocity
            )

            # Desired attitude and thrust
            att_des, thrust, acc_limited = (
                acceleration_mapping(
                    acc_des, yaw_des, angles
                )
            )

            # Attitude and angular rate control
            torque, omega_des = (
                attitude_controller(
                    att_des, angles, omega
                )
            )

            # Control allocation
            u, saturated = allocate(
                thrust, torque
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

            attitude_log.append(
                obs[0, [5, 4, 3]].copy()
            )

            vel_des_log.append(vel_des.copy())
            acc_des_log.append(acc_limited.copy())
            att_des_log.append(att_des.copy())
            rate_des_log.append(omega_des.copy())

            thrust_log.append(thrust)
            action_log.append(u.copy())

        t = np.asarray(time)

        position = np.asarray(position_log)
        velocity = np.asarray(velocity_log)

        attitude = np.asarray(attitude_log)
        att_des_array = np.asarray(att_des_log)

        error = target - position

        error_norm = np.linalg.norm(
            error, axis=1
        )

        initial_error = error_norm[0]

        outside = np.flatnonzero(
            error_norm > 0.02 * initial_error
        )

        if len(outside) == 0:
            settling_time = 0.0

        elif outside[-1] == len(t) - 1:
            settling_time = np.nan

        else:
            settling_time = t[outside[-1] + 1]

        tilt = np.arccos(
            np.clip(
                np.cos(attitude[:, 0])
                * np.cos(attitude[:, 1]),
                -1.0, 1.0
            )
        )

        yaw_error = (
            yaw_des - attitude[-1, 2] + np.pi
        ) % (2 * np.pi) - np.pi

        actions = np.asarray(action_log)

        results[name] = {
            "t": t,
            "target": target,
            "position": position,
            "velocity": velocity,
            "attitude": attitude,
            "vel_des": np.asarray(vel_des_log),
            "acc_des": np.asarray(acc_des_log),
            "att_des": att_des_array,
            "rate_des": np.asarray(rate_des_log),
            "thrust": np.asarray(thrust_log),
            "actions": actions,
            "error_norm": error_norm,
            "tilt": tilt
        }

        print(f"\n========== {name} ==========")

        print("Target position:", target, "m")
        print("Final position:", position[-1], "m")
        print("Final error:", error[-1], "m")

        print(
            "Final position error norm:",
            f"{error_norm[-1]:.8f} m"
        )

        print(
            "Position RMSE:",
            f"{np.sqrt(np.mean(error_norm**2)):.8f} m"
        )

        print(
            "Final velocity:",
            velocity[-1], "m/s"
        )

        print(
            "Final attitude [R, P, Y]:",
            np.rad2deg(attitude[-1]), "deg"
        )

        print(
            "Final yaw error:",
            f"{np.rad2deg(yaw_error):.8f} deg"
        )

        print(
            "Settling time (2%):",
            f"{settling_time:.4f} s"
        )

        print(
            "Max horizontal speed:",
            f"{np.max(np.linalg.norm(velocity[:, :2], axis=1)):.6f} m/s"
        )

        print(
            "Max actual tilt:",
            f"{np.rad2deg(tilt.max()):.6f} deg"
        )

        print(
            "Max altitude deviation:",
            f"{np.max(np.abs(error[:, 2])):.8f} m"
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

finally:
    env.close()


# 12. Plot results
fig, axes = plt.subplots(
    4, 3, figsize=(15, 15)
)

for row, (name, data) in enumerate(results.items()):

    t = data["t"]
    target = data["target"]

    position = data["position"]
    velocity = data["velocity"]

    attitude = np.rad2deg(
        data["attitude"]
    )

    att_des = np.rad2deg(
        data["att_des"]
    )

    for i, axis_name in enumerate(["X", "Y", "Z"]):

        line, = axes[row, 0].plot(
            t, position[:, i],
            label=axis_name
        )

        axes[row, 0].axhline(
            target[i],
            color=line.get_color(),
            linestyle=":",
            alpha=0.7
        )

        line, = axes[row, 1].plot(
            t, velocity[:, i],
            label=axis_name
        )

        axes[row, 1].plot(
            t[:-1],
            data["vel_des"][:, i],
            color=line.get_color(),
            linestyle="--",
            alpha=0.7
        )

    for i, angle_name in enumerate(
        ["Roll", "Pitch", "Yaw"]
    ):

        line, = axes[row, 2].plot(
            t, attitude[:, i],
            label=angle_name
        )

        axes[row, 2].plot(
            t[:-1], att_des[:, i],
            color=line.get_color(),
            linestyle="--",
            alpha=0.7
        )

    axes[row, 0].set_title(
        f"{name}: Position"
    )

    axes[row, 1].set_title(
        f"{name}: Velocity"
    )

    axes[row, 2].set_title(
        f"{name}: Attitude"
    )

    axes[row, 0].set_ylabel("Position (m)")
    axes[row, 1].set_ylabel("Velocity (m/s)")
    axes[row, 2].set_ylabel("Angle (deg)")

    for ax in axes[row]:
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

fig.suptitle(
    "Flightmare Cascaded Position Control"
)

fig.tight_layout()

output = root / "learning/11_position_control.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("\nFigure saved:", output)


# 13. Three-dimensional trajectories
fig = plt.figure(figsize=(9, 8))

ax = fig.add_subplot(
    111, projection="3d"
)

for name, data in results.items():

    p = data["position"]
    target = data["target"]

    line, = ax.plot(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        label=name
    )

    ax.scatter(
        *target,
        marker="x",
        s=70,
        color=line.get_color()
    )

ax.scatter(
    0, 0, 5,
    color="black",
    marker="o",
    label="Start"
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title(
    "Flightmare 3D Position Trajectories"
)

ax.legend()
fig.tight_layout()

output = root / "learning/11_position_trajectory.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("Figure saved:", output)
