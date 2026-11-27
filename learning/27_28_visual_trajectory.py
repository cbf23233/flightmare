"""多频率视觉联合仿真及图像状态同步采集。"""

import importlib
import time
from pathlib import Path

import flightgym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cascaded_controller import compute_control, dt

mounting = importlib.import_module("20_rgb_camera_mounting")

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "27_28_visual_results"

DURATION = 16.0
CAMERA_HZ = 10
RADIUS = 2.0
OMEGA = 0.4
HEIGHT = 5.0
YAW_DES = 0.0

STEPS = round(DURATION / dt)
INTERVAL = round(1.0 / (CAMERA_HZ * dt))
NUM_FRAMES = STEPS // INTERVAL

WIDTH = 640
HEIGHT_PX = 360

assert np.isclose(dt, 0.02)
assert np.isclose(INTERVAL * dt, 1.0 / CAMERA_HZ)


def trajectory(t):
    phase = OMEGA * t

    position = np.array([
        RADIUS * np.cos(phase),
        RADIUS * np.sin(phase),
        HEIGHT,
    ])

    velocity = np.array([
        -RADIUS * OMEGA * np.sin(phase),
        RADIUS * OMEGA * np.cos(phase),
        0.0,
    ])

    acceleration = np.array([
        -RADIUS * OMEGA**2 * np.cos(phase),
        -RADIUS * OMEGA**2 * np.sin(phase),
        0.0,
    ])

    return position, velocity, acceleration


def make_config():
    position, rotation = mounting.CASES["front_40cm"]
    return mounting.make_config(position, rotation)


def capture():
    env = flightgym.QuadrotorEnv_v1(
        make_config(), False
    )

    assert env.getNumOfEnvs() == 1
    assert env.getObsDim() == 12
    assert env.getActDim() == 4
    assert env.getNumOfCameras(0) == 1

    obs = np.zeros((1, 12), dtype=np.float32)
    action = np.zeros((1, 4), dtype=np.float32)
    reward = np.zeros(1, dtype=np.float32)
    done = np.zeros(1, dtype=bool)

    extra = np.zeros(
        (1, len(env.getExtraInfoNames())),
        dtype=np.float32,
    )

    state = np.zeros((1, 13), dtype=np.float32)

    p0, v0, _ = trajectory(0.0)

    state[0, :3] = p0
    state[0, 3] = 1.0
    state[0, 7:10] = v0

    assert env.resetState(state, obs)

    sim_time = np.arange(STEPS + 1) * dt

    reference = np.array([
        trajectory(t)[0] for t in sim_time
    ])

    observations = np.zeros(
        (STEPS + 1, 12), dtype=np.float32
    )

    actions = np.zeros(
        (STEPS, 4), dtype=np.float32
    )

    rewards = np.zeros(STEPS)
    controller_times = np.zeros(STEPS)
    dynamics_times = np.zeros(STEPS)

    frame_ids = np.zeros(NUM_FRAMES, dtype=np.int64)
    frame_steps = np.zeros(NUM_FRAMES, dtype=np.int64)
    frame_sim_times = np.zeros(NUM_FRAMES)

    frame_states = np.zeros(
        (NUM_FRAMES, 12), dtype=np.float32
    )

    request_times = np.zeros(NUM_FRAMES)
    receive_times = np.zeros(NUM_FRAMES)
    render_times = np.zeros(NUM_FRAMES)

    images = np.empty(
        (NUM_FRAMES, HEIGHT_PX, WIDTH, 3),
        dtype=np.uint8,
    )

    observations[0] = obs[0]

    frame_count = 0
    saturation_count = 0
    connected = False

    try:
        assert env.connectUnity()
        connected = True

        wall_start = time.perf_counter()

        for k in range(STEPS):

            # 1. Controller: 50 Hz
            pos_ref, vel_ref, acc_ref = trajectory(k * dt)

            t0 = time.perf_counter()

            u, info = compute_control(
                obs[0],
                pos_ref,
                YAW_DES,
                velocity_ff=vel_ref,
                acceleration_ff=acc_ref,
            )

            action[0] = u

            controller_times[k] = (
                time.perf_counter() - t0
            )

            saturation_count += int(info["saturated"])

            # 2. Dynamics: 50 Hz
            t0 = time.perf_counter()

            env.testStep(
                action, obs, reward, done, extra
            )

            dynamics_times[k] = (
                time.perf_counter() - t0
            )

            step_id = k + 1

            assert not done[0], (
                f"Terminal state at step {step_id}"
            )

            assert np.isfinite(obs).all()
            assert np.isfinite(reward).all()

            observations[step_id] = obs[0]
            actions[k] = action[0]
            rewards[k] = reward[0]

            # 3. RGB camera: 10 Hz
            if step_id % INTERVAL == 0:

                j = frame_count
                frame_id = j + 1

                # Snapshot before rendering
                sample_state = obs[0].copy()

                frame_ids[j] = frame_id
                frame_steps[j] = step_id
                frame_sim_times[j] = step_id * dt
                frame_states[j] = sample_state

                request_times[j] = (
                    time.perf_counter() - wall_start
                )

                t0 = time.perf_counter()

                assert env.render(frame_id)

                render_times[j] = (
                    time.perf_counter() - t0
                )

                image = env.getRGBImage(0, 0)

                receive_times[j] = (
                    time.perf_counter() - wall_start
                )

                assert image is not None
                assert image.shape == (
                    HEIGHT_PX, WIDTH, 3
                )
                assert image.dtype == np.uint8

                # Each image is consumed exactly once
                assert env.getRGBImage(0, 0) is None

                # Rendering must not change the observation
                np.testing.assert_array_equal(
                    obs[0], sample_state
                )

                images[j] = image
                frame_count += 1

            else:
                assert env.getRGBImage(0, 0) is None

            if step_id % round(2.0 / dt) == 0:

                error = np.linalg.norm(
                    obs[0, :3] - reference[step_id]
                )

                print(
                    f"t={step_id*dt:5.2f} s | "
                    f"position={np.round(obs[0, :3], 3)} | "
                    f"error={error:.5f} m | "
                    f"frames={frame_count}"
                )

        elapsed = time.perf_counter() - wall_start

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    assert frame_count == NUM_FRAMES

    # 4. Verify image-state association
    np.testing.assert_array_equal(
        frame_ids,
        np.arange(1, NUM_FRAMES + 1),
    )

    np.testing.assert_array_equal(
        frame_steps,
        np.arange(INTERVAL, STEPS + 1, INTERVAL),
    )

    np.testing.assert_allclose(
        frame_sim_times,
        sim_time[frame_steps],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        frame_states,
        observations[frame_steps],
    )

    assert np.all(
        receive_times >= request_times
    )

    assert np.all(
        np.diff(receive_times) > 0
    )

    assert np.all(
        request_times[1:] >= receive_times[:-1]
    )

    error = observations[:, :3] - reference
    error_norm = np.linalg.norm(error, axis=1)

    rmse = np.sqrt(np.mean(error_norm**2))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all raw RGB frames in acquisition order
    np.save(
        RESULT_DIR / "rgb_frames.npy",
        images,
    )

    # Row j of each frame array corresponds to RGB frame j
    np.savez(
        RESULT_DIR / "synchronized_states.npz",
        sim_time=sim_time,
        observations=observations,
        reference=reference,
        actions=actions,
        rewards=rewards,
        error_norm=error_norm,
        frame_ids=frame_ids,
        frame_steps=frame_steps,
        frame_sim_times=frame_sim_times,
        frame_states=frame_states,
        request_times=request_times,
        receive_times=receive_times,
        render_times=render_times,
        controller_times=controller_times,
        dynamics_times=dynamics_times,
        saturation_count=saturation_count,
        elapsed=elapsed,
        dt=dt,
        camera_hz=CAMERA_HZ,
        width=WIDTH,
        height=HEIGHT_PX,
        fov=70.0,
        camera_position=np.array([0.0, 0.4, 0.0]),
        camera_rotation=np.zeros(3),
    )

    print("\n========== Tracking ==========")
    print(f"Position RMSE     : {rmse:.8f} m")
    print(f"Maximum error     : {error_norm.max():.8f} m")
    print(f"Maximum Z error   : {np.abs(error[:, 2]).max():.8f} m")
    print(f"Saturated steps   : {saturation_count}")

    print("\n========== Synchronization ==========")
    print(f"Dynamics steps    : {STEPS}")
    print(f"RGB frames        : {frame_count}")
    print(f"First sample time : {frame_sim_times[0]:.6f} s")
    print(f"Last sample time  : {frame_sim_times[-1]:.6f} s")
    print(f"Sample interval   : {np.diff(frame_sim_times).mean():.6f} s")
    print("Frame sequence    : PASS")
    print("State association : PASS")
    print("Sampling schedule : PASS")
    print("Image queue       : PASS")

    print("\n========== Performance ==========")
    print(f"Simulation time   : {sim_time[-1]:.6f} s")
    print(f"Wall time         : {elapsed:.6f} s")
    print(f"RTF               : {sim_time[-1]/elapsed:.6f}")
    print(f"Controller mean   : {controller_times.mean()*1000:.6f} ms")
    print(f"Dynamics mean     : {dynamics_times.mean()*1000:.6f} ms")
    print(f"Render mean       : {render_times.mean()*1000:.6f} ms")
    print(f"Render P95        : {np.percentile(render_times,95)*1000:.6f} ms")
    print(f"Render maximum    : {render_times.max()*1000:.6f} ms")

    acquisition_times = receive_times - request_times

    print(
        f"Acquisition mean  : "
        f"{acquisition_times.mean()*1000:.6f} ms"
    )

    print(f"RGB data size     : {images.nbytes/1024**2:.2f} MiB")

    return images


def plot_summary(images):

    with np.load(
        RESULT_DIR / "synchronized_states.npz"
    ) as data:

        t = data["sim_time"]
        obs = data["observations"]
        ref = data["reference"]

        ft = data["frame_sim_times"]
        fs = data["frame_states"]
        fi = data["frame_ids"]

        fig, axes = plt.subplots(
            3, 3,
            figsize=(16, 12),
            constrained_layout=True,
        )

        # 1. XY trajectory and camera sampling positions
        ax = axes[0, 0]

        ax.plot(
            ref[:, 0], ref[:, 1],
            "k--", label="Reference",
        )

        ax.plot(
            obs[:, 0], obs[:, 1],
            label="Actual",
        )

        ax.scatter(
            fs[::10, 0], fs[::10, 1],
            s=15, color="tab:green",
            label="Camera samples (1 s)",
        )

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Circular trajectory")
        ax.legend(fontsize=8)

        # 2. Position tracking error
        ax = axes[0, 1]

        ax.plot(t, data["error_norm"])

        ax.set_title("Position tracking error")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Error (m)")

        # 3. Altitude and camera sampling
        ax = axes[0, 2]

        ax.plot(t, obs[:, 2], label="Actual")

        ax.plot(
            t, ref[:, 2],
            "k--", label="Reference",
        )

        ax.scatter(
            ft[::10], fs[::10, 2],
            s=12, color="tab:green",
            label="Camera samples (1 s)",
        )

        ax.set_title("Altitude")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Z (m)")
        ax.legend(fontsize=8)

        # 4. Simulated camera sampling interval
        ax = axes[1, 0]

        ax.plot(
            fi[1:], np.diff(ft)*1000,
            label="Sample interval",
        )

        ax.axhline(
            100, linestyle="--",
            color="gray",
        )

        ax.set_ylim(95, 105)
        ax.set_title("Camera sampling interval")
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Simulation interval (ms)")

        # 5. Rendering and image acquisition duration
        ax = axes[1, 1]

        ax.plot(
            fi, data["render_times"]*1000,
            label="Render",
        )

        ax.plot(
            fi,
            (data["receive_times"]
             - data["request_times"])*1000,
            alpha=0.7,
            label="Acquisition",
        )

        ax.axhline(
            100, linestyle="--",
            color="gray", label="Camera period",
        )

        ax.set_title("Wall-clock acquisition duration")
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Duration (ms)")
        ax.legend(fontsize=8)

        # 6. Simulation time versus wall time
        ax = axes[1, 2]

        ax.plot(
            data["receive_times"], ft,
            label="Simulation",
        )

        limit = max(
            float(data["elapsed"]),
            float(t[-1]),
        )

        ax.plot(
            [0, limit], [0, limit],
            "k--", label="RTF = 1",
        )

        ax.set_title("Simulation progress")
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("Simulation time (s)")
        ax.legend(fontsize=8)

        for ax in axes[:2].flat:
            ax.grid(alpha=0.3)

        # 7-9. RGB images with associated states
        selected = [
            0,
            len(images)//2,
            len(images)-1,
        ]

        for ax, j in zip(axes[2], selected):

            ax.imshow(images[j])

            x, y, z = fs[j, :3]
            yaw, pitch, roll = np.rad2deg(fs[j, 3:6])

            ax.set_title(
                f"Frame {fi[j]} | t={ft[j]:.1f} s\n"
                f"XYZ=({x:.2f}, {y:.2f}, {z:.2f}) m\n"
                f"RPY=({roll:.1f}, {pitch:.1f}, {yaw:.1f}) deg",
                fontsize=10,
            )

            ax.axis("off")

        fig.suptitle(
            "Experiments 27-28: Synchronized Visual Trajectory",
            fontsize=18,
        )

        output = ROOT / "27_28_visual_trajectory.png"

        fig.savefig(output, dpi=150)
        plt.close(fig)

        print(f"\nSummary figure: {output}")


if __name__ == "__main__":
    images = capture()
    plot_summary(images)
    print("\nExperiments 27-28 completed.")
