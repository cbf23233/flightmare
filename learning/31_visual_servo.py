"""基于RGB图像的单轴视觉闭环控制实验。"""

import importlib
import time
from pathlib import Path

import cv2
import flightgym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cascaded_controller import compute_control, dt

mounting = importlib.import_module("20_rgb_camera_mounting")

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "31_visual_results"

DURATION = 180.0
YAW_REF = np.deg2rad(20.0)
CAMERA_HZ = 10

KU = 0.005
MAX_VX = 1.0
DEADBAND = 3.0

WIDTH = 640
HEIGHT = 360
CX = (WIDTH - 1) / 2

STEPS = round(DURATION / dt)
INTERVAL = round(1 / (CAMERA_HZ * dt))

LOWER = np.array([90, 70, 30], dtype=np.uint8)
UPPER = np.array([135, 255, 255], dtype=np.uint8)
KERNEL = np.ones((3, 3), dtype=np.uint8)

assert np.isclose(dt, 0.02)
assert np.isclose(INTERVAL * dt, 0.1)


def detect_target(rgb):

    height, width = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, KERNEL
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, KERNEL
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if not 500 <= area <= 20000:
            continue

        if w / h < 1.5:
            continue

        if w > width * 0.6:
            continue

        if (
            x == 0 or y == 0
            or x + w >= width
            or y + h >= height
        ):
            continue

        candidates.append((area, x, y, w, h))

    if not candidates:
        return None

    area, x, y, w, h = max(candidates)

    u = x + (w - 1) / 2
    v = y + (h - 1) / 2

    return u, v, area, (x, y, w, h)


def visual_control(result):

    if result is None:
        return 0.0

    error = result[0] - CX

    if abs(error) <= DEADBAND:
        return 0.0

    return float(
        np.clip(KU * error, -MAX_VX, MAX_VX)
    )


def annotate(rgb, result):

    image = rgb.copy()

    cv2.line(
        image,
        (round(CX), 0),
        (round(CX), HEIGHT - 1),
        (255, 0, 0),
        1,
    )

    if result is not None:

        u, v, area, (x, y, w, h) = result

        cv2.rectangle(
            image,
            (x, y),
            (x+w-1, y+h-1),
            (0, 255, 0),
            2,
        )

        cv2.circle(
            image,
            (round(u), round(v)),
            4,
            (0, 255, 0),
            -1,
        )

        cv2.line(
            image,
            (round(u), round(v)),
            (round(CX), round(v)),
            (255, 255, 0),
            2,
        )

    return image


def run():

    position, rotation = mounting.CASES["front_40cm"]

    env = flightgym.QuadrotorEnv_v1(
        mounting.make_config(position, rotation),
        False,
    )

    obs = np.zeros((1, 12), dtype=np.float32)
    action = np.zeros((1, 4), dtype=np.float32)
    reward = np.zeros(1, dtype=np.float32)
    done = np.zeros(1, dtype=bool)

    extra = np.zeros(
        (1, len(env.getExtraInfoNames())),
        dtype=np.float32,
    )

    state = np.zeros((1, 13), dtype=np.float32)

    state[0, :3] = [0, 0, 5]
    state[0, 3] = np.cos(YAW_REF / 2)
    state[0, 6] = np.sin(YAW_REF / 2)

    assert env.resetState(state, obs)

    observations = [obs[0].copy()]
    actions = []
    velocity_commands = []

    frame_ids = []
    frame_steps = []
    frame_times = []
    frame_states = []

    target_u = []
    target_v = []
    target_area = []
    visual_errors = []
    visual_commands = []

    render_times = []
    detection_times = []

    snapshots = {}

    vx_cmd = 0.0
    frame_id = 0
    lost_count = 0
    saturated_count = 0

    connected = False

    def sample(step_id):

        nonlocal frame_id, vx_cmd, lost_count

        frame_id += 1

        sample_state = obs[0].copy()

        t0 = time.perf_counter()

        assert env.render(frame_id)

        render_times.append(
            time.perf_counter() - t0
        )

        rgb = env.getRGBImage(0, 0)

        assert rgb is not None
        assert rgb.shape == (HEIGHT, WIDTH, 3)
        assert rgb.dtype == np.uint8
        assert env.getRGBImage(0, 0) is None

        np.testing.assert_array_equal(
            obs[0], sample_state
        )

        t0 = time.perf_counter()

        result = detect_target(rgb)
        vx_cmd = visual_control(result)

        detection_times.append(
            time.perf_counter() - t0
        )

        if result is None:

            lost_count += 1
            u = v = area = error = np.nan

        else:

            u, v, area, _ = result
            error = u - CX

        frame_ids.append(frame_id)
        frame_steps.append(step_id)
        frame_times.append(step_id * dt)
        frame_states.append(sample_state)

        target_u.append(u)
        target_v.append(v)
        target_area.append(area)
        visual_errors.append(error)
        visual_commands.append(vx_cmd)

        if step_id in (0, STEPS // 2, STEPS):

            snapshots[step_id] = (
                annotate(rgb, result),
                error,
            )

        if step_id == 0 or step_id % 500 == 0:

            print(
                f"t={step_id*dt:6.1f} s | "
                f"x={obs[0,0]:8.3f} m | "
                f"u={u:7.2f} px | "
                f"error={error:7.2f} px | "
                f"vx_cmd={vx_cmd:7.3f} m/s | "
                f"lost={lost_count}",
                flush=True,
            )

        return result

    try:

        assert env.connectUnity()
        connected = True

        wall_start = time.perf_counter()

        # 初始图像：t = 0
        assert sample(0) is not None, (
            "Initial target not detected"
        )

        for k in range(STEPS):

            # X方向位置误差置零；
            # Y、Z方向继续使用位置控制。
            pos_ref = np.array([
                obs[0, 0],
                0.0,
                5.0,
            ])

            velocity_ff = np.array([
                vx_cmd,
                0.0,
                0.0,
            ])

            u, info = compute_control(
                obs[0],
                pos_ref,
                YAW_REF,
                velocity_ff=velocity_ff,
            )

            saturated_count += int(
                info["saturated"]
            )

            action[0] = u

            actions.append(action[0].copy())
            velocity_commands.append(vx_cmd)

            env.testStep(
                action, obs, reward, done, extra
            )

            step_id = k + 1

            assert not done[0], (
                f"Terminal state at step {step_id}"
            )

            assert np.isfinite(obs).all()

            observations.append(obs[0].copy())

            # 每5个动力学步更新一次视觉指令
            if step_id % INTERVAL == 0:
                sample(step_id)

        elapsed = time.perf_counter() - wall_start

    finally:

        if connected:
            env.disconnectUnity()

        env.close()

    observations = np.asarray(observations)
    frame_states = np.asarray(frame_states)
    frame_steps = np.asarray(frame_steps)

    expected_steps = np.arange(
        0, STEPS + 1, INTERVAL
    )

    np.testing.assert_array_equal(
        frame_steps, expected_steps
    )

    np.testing.assert_array_equal(
        frame_ids,
        np.arange(1, len(expected_steps) + 1),
    )

    np.testing.assert_array_equal(
        frame_states,
        observations[frame_steps],
    )

    errors = np.asarray(visual_errors)
    valid = np.isfinite(errors)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    data = dict(
        sim_time=np.arange(STEPS + 1) * dt,
        observations=observations,
        actions=np.asarray(actions),
        velocity_commands=np.asarray(velocity_commands),
        frame_ids=np.asarray(frame_ids),
        frame_steps=frame_steps,
        frame_times=np.asarray(frame_times),
        frame_states=frame_states,
        target_u=np.asarray(target_u),
        target_v=np.asarray(target_v),
        target_area=np.asarray(target_area),
        visual_errors=errors,
        visual_commands=np.asarray(visual_commands),
        render_times=np.asarray(render_times),
        detection_times=np.asarray(detection_times),
        elapsed=elapsed,
        lost_count=lost_count,
        saturated_count=saturated_count,
        ku=KU,
        deadband=DEADBAND,
    )

    np.savez(
        RESULT_DIR / "visual_servo.npz",
        **data,
    )

    for step_id, (image, error) in snapshots.items():

        plt.imsave(
            RESULT_DIR / f"frame_{step_id:05d}.png",
            image,
        )

    print("\n========== Visual servo results ==========")

    print(f"Dynamics steps      : {STEPS}")
    print(f"RGB frames          : {len(frame_ids)}")
    print(f"Detected frames     : {valid.sum()}")
    print(f"Detection rate      : {valid.mean()*100:.2f}%")
    print(f"Lost frames         : {lost_count}")
    print(f"Initial pixel error : {errors[0]:.4f} px")
    print(f"Final pixel error   : {errors[-1]:.4f} px")

    tail = errors[-50:]

    if np.isfinite(tail).all():

        print(
            f"Final 5 s RMSE      : "
            f"{np.sqrt(np.mean(tail**2)):.4f} px"
        )

    print(f"Final position      : {observations[-1,:3]}")
    print(f"Final velocity      : {observations[-1,6:9]}")

    print(
        f"Maximum Y error     : "
        f"{np.abs(observations[:,1]).max():.6f} m"
    )

    print(
        f"Maximum Z error     : "
        f"{np.abs(observations[:,2]-5).max():.6f} m"
    )

    print(f"Saturated steps     : {saturated_count}")
    print(f"Simulation time     : {DURATION:.3f} s")
    print(f"Wall time           : {elapsed:.3f} s")
    print(f"RTF                 : {DURATION/elapsed:.4f}")

    print(
        f"Mean render time    : "
        f"{np.mean(render_times)*1000:.4f} ms"
    )

    print(
        f"Mean detection time : "
        f"{np.mean(detection_times)*1000:.4f} ms"
    )

    aligned = (
        np.isfinite(tail).all()
        and np.all(np.abs(tail) <= DEADBAND)
    )

    print(
        "Final alignment     : "
        + ("PASS" if aligned else "NOT CONVERGED")
    )

    print("Frame-state association: PASS")

    return data, snapshots


def plot_summary(data, snapshots):

    t = data["sim_time"]
    ft = data["frame_times"]
    obs = data["observations"]

    fig, axes = plt.subplots(
        3, 3,
        figsize=(16, 12),
        constrained_layout=True,
    )

    ax = axes[0, 0]

    ax.plot(ft, data["target_u"], label="Target u")
    ax.axhline(CX, color="black", linestyle="--",
               label="Desired center")

    ax.axhspan(
        CX-DEADBAND, CX+DEADBAND,
        color="green", alpha=0.15,
    )

    ax.set_title("Horizontal target center")
    ax.set_ylabel("Pixel coordinate (px)")
    ax.legend()

    ax = axes[0, 1]

    ax.plot(ft, data["visual_errors"])
    ax.axhline(0, color="black", linestyle="--")

    ax.axhspan(
        -DEADBAND, DEADBAND,
        color="green", alpha=0.15,
    )

    ax.set_title("Visual tracking error")
    ax.set_ylabel("Pixel error (px)")

    ax = axes[0, 2]

    ax.step(
        ft, data["visual_commands"],
        where="post", label="Command",
    )

    ax.plot(t, obs[:, 6], label="Actual")

    ax.set_title("Horizontal velocity")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()

    ax = axes[1, 0]

    ax.plot(t, obs[:, 0], label="X")
    ax.plot(t, obs[:, 1], label="Y")

    ax.set_title("Horizontal position")
    ax.set_ylabel("Position (m)")
    ax.legend()

    ax = axes[1, 1]

    ax.plot(t, obs[:, 2], label="Actual")
    ax.axhline(5, color="black", linestyle="--",
               label="Reference")

    ax.set_title("Altitude")
    ax.set_ylabel("Height (m)")
    ax.legend()

    ax = axes[1, 2]

    ax.plot(
        ft, data["render_times"]*1000,
        label="Render",
    )

    ax.plot(
        ft, data["detection_times"]*1000,
        label="Detection",
    )

    ax.axhline(
        100, color="gray", linestyle="--",
        label="Camera period",
    )

    ax.set_title("Visual processing duration")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.legend()

    for ax in axes[:2].flat:

        ax.set_xlabel("Simulation time (s)")
        ax.grid(alpha=0.3)

    for ax, step_id in zip(
        axes[2], sorted(snapshots)
    ):

        image, error = snapshots[step_id]

        ax.imshow(image)

        ax.set_title(
            f"t={step_id*dt:.1f} s | "
            f"error={error:.2f} px"
        )

        ax.axis("off")

    fig.suptitle(
        "Experiment 31: RGB Visual Servoing",
        fontsize=18,
    )

    output = ROOT / "31_visual_servo.png"

    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"\nSummary figure: {output}")


if __name__ == "__main__":

    data, snapshots = run()
    plot_summary(data, snapshots)
