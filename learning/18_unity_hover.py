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


# ============================================================
# 1. Experiment configuration
# ============================================================

root = Path(__file__).resolve().parents[1]
os.environ["FLIGHTMARE_PATH"] = str(root)

duration = 15.0
steps = round(duration / dt)

target = np.array([0.0, 0.0, 5.0])
yaw_des = 0.0

cases = [
    ("Altitude recovery", [0, 0, 3], [0, 0, 0]),
    ("Attitude recovery", [0, 0, 5], [10, -10, 30]),
    ("Position recovery", [-2, 2, 5], [0, 0, 0]),
    ("Combined recovery", [-2, 2, 3], [10, -10, 30]),
]


# ============================================================
# 2. Quaternion conversion
# ============================================================

def euler_to_quaternion(roll, pitch, yaw):

    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


# ============================================================
# 3. Environment initialization
# ============================================================

yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)

cfg["env"]["seed"] = 1
cfg["env"]["scene_id"] = 2
cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = True

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)

state = np.zeros((1, 13), dtype=np.float32)
obs = np.zeros((1, env.getObsDim()), dtype=np.float32)
action = np.zeros((1, env.getActDim()), dtype=np.float32)
reward = np.zeros(1, dtype=np.float32)
done = np.zeros(1, dtype=np.bool_)

extra_info = np.zeros(
    (1, len(env.getExtraInfoNames())),
    dtype=np.float32
)

results = {}
connected = False
all_passed = True


def reset_initial_state(position, rpy_deg):

    state.fill(0)

    state[0, :3] = position
    state[0, 3:7] = euler_to_quaternion(
        *np.deg2rad(rpy_deg)
    )

    if not env.resetState(state, obs):
        raise RuntimeError("Initial state reset failed")

    done.fill(False)


# ============================================================
# 4. Closed-loop simulation
# ============================================================

try:

    reset_initial_state(cases[0][1], cases[0][2])

    print("Connecting Unity...")

    connected = env.connectUnity()

    if not connected:
        raise RuntimeError("Unity connection failed")

    print("Unity connected")
    print(f"Control frequency: {1 / dt:.1f} Hz")

    for case_id, (name, position, rpy_deg) in enumerate(cases):

        print(f"\n========== {name} ==========")

        reset_initial_state(position, rpy_deg)

        # Display initial state without advancing dynamics.
        if not env.render(case_id + 1):
            raise RuntimeError("Initial rendering failed")

        time.sleep(2)

        print("Initial position:", obs[0, :3])
        print(
            "Initial RPY (deg):",
            np.rad2deg(obs[0, [5, 4, 3]])
        )

        observations = [obs[0].copy()]
        actions = []
        step_times = []

        saturation_count = 0
        deadline_misses = 0

        wall_start = time.perf_counter()

        for k in range(steps):

            loop_start = time.perf_counter()

            u, info = compute_control(
                obs[0],
                target,
                yaw_des
            )

            action[0, :] = u

            saturation_count += int(info["saturated"])

            step_start = time.perf_counter()

            if not env.step(
                action, obs, reward, done, extra_info
            ):
                raise RuntimeError("Simulation step failed")

            step_times.append(
                time.perf_counter() - step_start
            )

            if done[0]:
                raise RuntimeError(
                    f"{name}: terminated at {(k + 1) * dt:.2f} s"
                )

            if not np.isfinite(obs).all():
                raise RuntimeError("Invalid observation")

            observations.append(obs[0].copy())
            actions.append(u.copy())

            if (k + 1) % round(1 / dt) == 0:

                error = np.linalg.norm(
                    target - obs[0, :3]
                )

                print(
                    f"t={(k + 1) * dt:5.1f} s | "
                    f"position={np.round(obs[0, :3], 3)} | "
                    f"error={error:.4f} m"
                )

            elapsed = time.perf_counter() - loop_start

            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                deadline_misses += 1

        wall_time = time.perf_counter() - wall_start

        # ====================================================
        # 5. Performance evaluation
        # ====================================================

        data = np.asarray(observations)
        actions = np.asarray(actions)

        t = np.arange(len(data)) * dt

        position = data[:, :3]
        velocity = data[:, 6:9]
        attitude = data[:, [5, 4, 3]]
        omega = data[:, 9:12]

        position_error = np.linalg.norm(
            target - position, axis=1
        )

        attitude_error = np.rad2deg(
            (attitude + np.pi) % (2 * np.pi) - np.pi
        )

        tilt = np.rad2deg(
            np.arccos(
                np.clip(
                    np.cos(attitude[:, 0])
                    * np.cos(attitude[:, 1]),
                    -1.0, 1.0
                )
            )
        )

        # Absolute settling criterion, including attitude.
        settled = (
            (position_error <= 0.05)
            & (np.linalg.norm(velocity, axis=1) <= 0.05)
            & (np.max(np.abs(attitude_error), axis=1) <= 1.0)
            & (np.linalg.norm(omega, axis=1) <= np.deg2rad(1.0))
        )

        outside = np.flatnonzero(~settled)

        if len(outside) == 0:
            settling_time = 0.0
        elif outside[-1] == len(t) - 1:
            settling_time = np.nan
        else:
            settling_time = t[outside[-1] + 1]

        final_position_error = position_error[-1]

        final_velocity = np.linalg.norm(velocity[-1])

        final_attitude_error = np.max(
            np.abs(attitude_error[-1])
        )

        final_rate = np.rad2deg(
            np.linalg.norm(omega[-1])
        )

        passed = (
            final_position_error < 0.02
            and final_velocity < 0.02
            and final_attitude_error < 0.5
            and final_rate < 0.5
        )

        all_passed = all_passed and passed

        results[name] = {
            "t": t,
            "position": position,
            "velocity": velocity,
            "attitude": np.rad2deg(attitude),
            "error": position_error,
            "actions": actions,
        }

        print("\n---------- Results ----------")
        print("Final position:", position[-1])
        print("Final position error:", final_position_error, "m")
        print("Final speed:", final_velocity, "m/s")
        print("Final RPY:", np.rad2deg(attitude[-1]), "deg")
        print("Final angular speed:", final_rate, "deg/s")
        print("Settling time:", settling_time, "s")
        print("Maximum actual tilt:", tilt.max(), "deg")
        print("Action range:", actions.min(), actions.max())
        print("Saturated steps:", saturation_count)
        print("Deadline misses:", deadline_misses)
        print("Wall time:", wall_time, "s")
        print("Real-time factor:", duration / wall_time)
        print("Mean step time:", np.mean(step_times) * 1000, "ms")
        print("P95 step time:", np.percentile(step_times, 95) * 1000, "ms")
        print("Maximum step time:", np.max(step_times) * 1000, "ms")
        print("PASS" if passed else "FAIL")

        time.sleep(2)

finally:

    if connected:
        env.disconnectUnity()
        print("\nUnity disconnected")

    env.close()
    print("Environment closed")


# ============================================================
# 6. Plot results
# ============================================================

fig, axes = plt.subplots(
    len(results), 4, figsize=(17, 12)
)

for row, (name, data) in enumerate(results.items()):

    t = data["t"]

    for i, label in enumerate(["X", "Y", "Z"]):

        line, = axes[row, 0].plot(
            t, data["position"][:, i], label=label
        )

        axes[row, 0].axhline(
            target[i],
            color=line.get_color(),
            linestyle=":"
        )

        axes[row, 1].plot(
            t, data["velocity"][:, i], label=label
        )

    for i, label in enumerate(["Roll", "Pitch", "Yaw"]):

        axes[row, 2].plot(
            t, data["attitude"][:, i], label=label
        )

    for i in range(4):

        axes[row, 3].plot(
            t[:-1], data["actions"][:, i],
            label=f"Motor {i + 1}"
        )

    titles = ["Position", "Velocity", "Attitude", "Motor actions"]
    units = ["Position (m)", "Velocity (m/s)", "Angle (deg)", "Action"]

    for col in range(4):

        ax = axes[row, col]

        ax.set_title(f"{name}: {titles[col]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(units[col])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

fig.tight_layout()

output = root / "learning/18_unity_hover.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("Figure saved:", output)


# ============================================================
# 7. Three-dimensional trajectories
# ============================================================

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

for name, data in results.items():

    p = data["position"]

    line, = ax.plot(
        p[:, 0], p[:, 1], p[:, 2],
        label=name
    )

    ax.scatter(*p[0], color=line.get_color(), marker="o")

ax.scatter(
    *target, color="black", marker="x",
    s=100, label="Target"
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Closed-loop Hover Recovery")
ax.legend()

fig.tight_layout()

output = root / "learning/18_unity_hover_trajectory.png"

fig.savefig(output, dpi=180)
plt.close(fig)

print("Figure saved:", output)

if not all_passed:
    raise RuntimeError("One or more hover tests failed")

print("\nAll closed-loop hover tests passed.")
