import os
import time
from pathlib import Path
from io import StringIO

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1

from cascaded_controller import compute_control, dt


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
cfg["env"]["seed"] = 1
cfg["env"]["scene_id"] = 2
cfg["env"]["render"] = True

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)


# 2. Trajectory parameters

radius = 2.0
frequency = 0.4
period = 2 * np.pi / frequency
duration = 3 * period
steps = round(duration / dt)


# 4. Reference trajectory
def trajectory(t):
    phase = frequency * t

    position = np.array([
        radius * np.cos(phase),
        radius * np.sin(phase),
        5.0
    ])

    velocity = np.array([
        -radius * frequency * np.sin(phase),
        radius * frequency * np.cos(phase),
        0.0
    ])

    acceleration = np.array([
        -radius * frequency**2 * np.cos(phase),
        -radius * frequency**2 * np.sin(phase),
        0.0
    ])

    return position, velocity, acceleration


# 11. Experiment configuration
cases = [
    ("Feedback Only", False, False),
    ("Velocity FF", True, False),
    ("Velocity + Acceleration FF", True, True)
]

state = np.zeros((1, 13), dtype=np.float32)

p0, v0, _ = trajectory(0.0)

state[0, :3] = p0
state[0, 3] = 1.0
state[0, 7:10] = v0

obs = np.zeros(
    (1, env.getObsDim()), dtype=np.float32
)

action = np.zeros(
    (1, env.getActDim()), dtype=np.float32
)

reward = np.zeros(1, dtype=np.float32)
done = np.zeros(1, dtype=np.bool_)

extra_info = np.zeros(
    (1, len(env.getExtraInfoNames())),
    dtype=np.float32
)

yaw_des = 0.0
results = {}

t = np.arange(steps + 1) * dt

reference = np.array([
    trajectory(ti)[0] for ti in t
])

reference_velocity = np.array([
    trajectory(ti)[1] for ti in t
])

# Exclude the first revolution from steady-state metrics
steady = t >= period

print("========== Trajectory parameters ==========")
print("Radius:", radius, "m")
print("Angular frequency:", frequency, "rad/s")
print("Reference speed:", radius * frequency, "m/s")
print("Period:", period, "s")
print("Duration:", t[-1], "s")
print("Control frequency:", 1 / dt, "Hz")


# 12. Closed-loop simulation

connected = False
all_passed = True

baseline = {
    "Feedback Only": 1.61999377,
    "Velocity FF": 0.41669653,
    "Velocity + Acceleration FF": 0.08274990,
}

try:

    if not env.resetState(state, obs):
        raise RuntimeError("Initial reset failed")

    print("Connecting Unity...")

    connected = env.connectUnity()

    if not connected:
        raise RuntimeError("Unity connection failed")

    print("Unity connected")

    for case_id, (name, use_vel_ff, use_acc_ff) in enumerate(cases):

        if not env.resetState(state, obs):
            raise RuntimeError("Reset failed")

        done.fill(False)

        print(f"\n========== Running {name} ==========")

        if not env.render(case_id + 1):
            raise RuntimeError("Initial rendering failed")

        time.sleep(2)

        print("Initial position:", obs[0, :3])
        print("Initial velocity:", obs[0, 6:9])

        position_log = [obs[0, :3].copy()]
        velocity_log = [obs[0, 6:9].copy()]
        attitude_log = [obs[0, [5, 4, 3]].copy()]

        vel_des_log = []
        acc_des_log = []
        att_des_log = []
        action_log = []

        saturation_count = 0
        deadline_misses = 0
        step_times = []

        wall_start = time.perf_counter()

        for k in range(steps):

            loop_start = time.perf_counter()

            pos_ref, vel_ref, acc_ref = trajectory(k * dt)

            u, info = compute_control(
                obs[0],
                pos_ref,
                yaw_des,
                velocity_ff=vel_ref if use_vel_ff else None,
                acceleration_ff=acc_ref if use_acc_ff else None
            )

            vel_des = info["vel_des"]
            acc_limited = info["acc_des"]
            att_des = info["att_des"]
            saturated = info["saturated"]

            saturation_count += int(saturated)

            action[0, :] = u

            step_start = time.perf_counter()

            if not env.step(
                action, obs, reward, done, extra_info
            ):
                raise RuntimeError("Step failed")

            step_times.append(
                time.perf_counter() - step_start
            )

            if done[0]:
                raise RuntimeError(
                    f"{name}: terminated at {(k+1)*dt:.2f} s"
                )

            if not np.isfinite(obs).all():
                raise RuntimeError("Invalid observation")

            position_log.append(obs[0, :3].copy())
            velocity_log.append(obs[0, 6:9].copy())
            attitude_log.append(obs[0, [5, 4, 3]].copy())

            vel_des_log.append(vel_des.copy())
            acc_des_log.append(acc_limited.copy())
            att_des_log.append(att_des.copy())
            action_log.append(u.copy())

            if (k + 1) % round(5 / dt) == 0:

                current_ref = trajectory((k + 1) * dt)[0]

                current_error = np.linalg.norm(
                    current_ref - obs[0, :3]
                )

                print(
                    f"t={(k + 1) * dt:6.2f} s | "
                    f"position={np.round(obs[0, :3], 3)} | "
                    f"error={current_error:.4f} m"
                )

            elapsed = time.perf_counter() - loop_start

            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                deadline_misses += 1

        wall_time = time.perf_counter() - wall_start

        position = np.asarray(position_log)
        velocity = np.asarray(velocity_log)
        attitude = np.asarray(attitude_log)
        actions = np.asarray(action_log)

        error = reference - position
        velocity_error = reference_velocity - velocity

        error_norm = np.linalg.norm(error, axis=1)
        velocity_error_norm = np.linalg.norm(
            velocity_error, axis=1
        )

        actual_radius = np.linalg.norm(
            position[:, :2], axis=1
        )

        actual_phase = np.arctan2(
            position[:, 1], position[:, 0]
        )

        phase_error = (
            frequency * t - actual_phase + np.pi
        ) % (2 * np.pi) - np.pi

        tilt = np.arccos(
            np.clip(
                np.cos(attitude[:, 0])
                * np.cos(attitude[:, 1]),
                -1.0, 1.0
            )
        )

        results[name] = {
            "position": position,
            "velocity": velocity,
            "attitude": attitude,
            "att_des": np.asarray(att_des_log),
            "vel_des": np.asarray(vel_des_log),
            "acc_des": np.asarray(acc_des_log),
            "actions": actions,
            "error": error,
            "error_norm": error_norm,
            "velocity_error_norm": velocity_error_norm,
            "radius": actual_radius,
            "phase_error": phase_error,
            "tilt": tilt
        }

        mean_phase = np.arctan2(
            np.mean(np.sin(phase_error[steady])),
            np.mean(np.cos(phase_error[steady]))
        )

        print(f"\n========== {name} ==========")

        print("Velocity feedforward:", use_vel_ff)
        print("Acceleration feedforward:", use_acc_ff)

        print(
            "Full position RMSE:",
            f"{np.sqrt(np.mean(error_norm**2)):.8f} m"
        )

        print(
            "Steady position RMSE:",
            f"{np.sqrt(np.mean(error_norm[steady]**2)):.8f} m"
        )

        print(
            "Steady velocity RMSE:",
            f"{np.sqrt(np.mean(velocity_error_norm[steady]**2)):.8f} m/s"
        )

        print(
            "Max position error:",
            f"{error_norm.max():.8f} m"
        )

        print(
            "Steady max position error:",
            f"{error_norm[steady].max():.8f} m"
        )

        print(
            "Mean steady radius:",
            f"{actual_radius[steady].mean():.8f} m"
        )

        print(
            "Mean steady phase lag:",
            f"{np.rad2deg(mean_phase):.8f} deg"
        )

        print(
            "Max altitude error:",
            f"{np.max(np.abs(error[:, 2])):.8f} m"
        )

        print(
            "Max actual tilt:",
            f"{np.rad2deg(tilt.max()):.6f} deg"
        )

        print(
            "Action range:",
            f"[{actions.min():.6f}, {actions.max():.6f}]"
        )

        print("Saturated steps:", saturation_count)

        steady_rmse = np.sqrt(
            np.mean(error_norm[steady]**2)
        )

        baseline_rmse = baseline[name]

        rmse_difference = abs(
            steady_rmse - baseline_rmse
        )

        passed = (
            rmse_difference <= 1e-4
            and saturation_count == 0
        )

        all_passed = all_passed and passed

        print("\n---------- Unity performance ----------")
        print("Wall time:", wall_time, "s")
        print("Real-time factor:", t[-1] / wall_time)
        print("Mean step time:", np.mean(step_times) * 1000, "ms")
        print("P95 step time:", np.percentile(step_times, 95) * 1000, "ms")
        print("Maximum step time:", np.max(step_times) * 1000, "ms")
        print("Deadline misses:", deadline_misses)

        print("\n---------- Baseline comparison ----------")
        print("Baseline steady RMSE:", baseline_rmse, "m")
        print("Current steady RMSE:", steady_rmse, "m")
        print("Absolute difference:", rmse_difference, "m")
        print("PASS" if passed else "FAIL")

        time.sleep(2)

finally:

    if connected:
        env.disconnectUnity()
        print("\nUnity disconnected")

    env.close()
    print("Environment closed")


# 13. Plot trajectory comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].plot(
    reference[:, 0],
    reference[:, 1],
    "k--",
    label="Reference"
)

axes[0, 0].scatter(
    *reference[0, :2],
    color="black",
    marker="o",
    label="Start"
)

for name, data in results.items():

    p = data["position"]

    axes[0, 0].plot(
        p[:, 0], p[:, 1], label=name
    )

    axes[0, 1].plot(
        t, data["error_norm"], label=name
    )

    axes[0, 2].plot(
        t, data["velocity_error_norm"], label=name
    )

    axes[1, 0].plot(
        t, data["radius"], label=name
    )

    axes[1, 1].plot(
        t, np.rad2deg(data["phase_error"]), label=name
    )

    axes[1, 2].plot(
        t, data["error"][:, 2], label=name
    )

axes[0, 0].set_title("XY Trajectory")
axes[0, 0].set_xlabel("X (m)")
axes[0, 0].set_ylabel("Y (m)")
axes[0, 0].set_aspect("equal", adjustable="box")

axes[0, 1].set_title("Position Tracking Error")
axes[0, 1].set_ylabel("Error (m)")

axes[0, 2].set_title("Velocity Tracking Error")
axes[0, 2].set_ylabel("Error (m/s)")

axes[1, 0].set_title("Actual Trajectory Radius")
axes[1, 0].set_ylabel("Radius (m)")
axes[1, 0].axhline(radius, color="black", linestyle="--")

axes[1, 1].set_title("Phase Lag")
axes[1, 1].set_ylabel("Phase lag (deg)")
axes[1, 1].axhline(0, color="black", linestyle="--")

axes[1, 2].set_title("Altitude Error")
axes[1, 2].set_ylabel("Error (m)")

for i, ax in enumerate(axes.flat):
    if i != 0:
        ax.set_xlabel("Time (s)")
        ax.axvline(
            period, color="gray", linestyle=":",
            label="Steady-state start"
        )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Flightmare Unity Circular Trajectory Tracking")
fig.tight_layout()

output = root / "learning/19_unity_trajectory.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("\nFigure saved:", output)


# 14. Plot attitude and rotor actions
fig, axes = plt.subplots(3, 2, figsize=(13, 11))

for row, (name, data) in enumerate(results.items()):

    attitude = np.rad2deg(data["attitude"])
    att_des = np.rad2deg(data["att_des"])

    for i, label in enumerate(["Roll", "Pitch", "Yaw"]):

        line, = axes[row, 0].plot(
            t, attitude[:, i], label=label
        )

        axes[row, 0].plot(
            t[:-1], att_des[:, i],
            color=line.get_color(),
            linestyle="--", alpha=0.7
        )

    for i in range(4):

        axes[row, 1].plot(
            t[:-1], data["actions"][:, i],
            label=f"Rotor {i+1}"
        )

    axes[row, 0].set_title(f"{name}: Attitude")
    axes[row, 0].set_ylabel("Angle (deg)")

    axes[row, 1].set_title(f"{name}: Control Action")
    axes[row, 1].set_ylabel("Normalized action")

    for ax in axes[row]:
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

fig.suptitle("Flightmare Unity Trajectory Control Inputs")
fig.tight_layout()

output = root / "learning/19_unity_trajectory_inputs.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("Figure saved:", output)


if not all_passed:
    raise RuntimeError(
        "One or more trajectory regression tests failed"
    )

print("\nAll Unity trajectory tests passed.")
