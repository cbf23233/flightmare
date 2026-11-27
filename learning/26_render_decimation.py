"""验证单无人机动力学与渲染频率解耦。"""

import argparse
import importlib
import time
from pathlib import Path

import flightgym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

mounting = importlib.import_module("20_rgb_camera_mounting")

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "26_render_results"

DT = 0.02
DURATION = 10.0
STEPS = round(DURATION / DT)

CASES = {
    "physics": 0,
    "sync": 1,
    "decimated": 5,
}

LABELS = {
    "physics": "Physics only",
    "sync": "RGB 50 Hz",
    "decimated": "RGB 10 Hz",
}


def make_config(case):
    position, rotation = mounting.CASES["front_40cm"]
    cfg = mounting.make_config(position, rotation)

    if case == "physics":
        cfg = cfg.replace("render: true", "render: false")
        cfg = cfg.replace("enabled: true", "enabled: false")

    return cfg


def make_action(t):
    """共同推力指令，产生平缓的垂直运动。"""
    value = 0.01 * np.sin(2.0 * np.pi * t / 5.0)

    return np.full((1, 4), value, dtype=np.float32)


def run(case):
    interval = CASES[case]
    use_unity = interval > 0

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    env = flightgym.QuadrotorEnv_v1(
        make_config(case), False
    )

    assert env.getNumOfEnvs() == 1
    assert env.getObsDim() == 12
    assert env.getActDim() == 4

    obs = np.zeros((1, 12), dtype=np.float32)
    reward = np.zeros(1, dtype=np.float32)
    done = np.zeros(1, dtype=bool)

    extra = np.zeros(
        (1, len(env.getExtraInfoNames())),
        dtype=np.float32,
    )

    state = np.zeros((1, 13), dtype=np.float32)
    state[0, 2] = 5.0
    state[0, 3] = 1.0

    assert env.resetState(state, obs)

    history = np.zeros((STEPS + 1, 12))
    actions = np.zeros((STEPS, 4))
    rewards = np.zeros(STEPS)
    step_times = np.zeros(STEPS)

    frame_steps = []
    frame_times = []
    render_times = []
    first_image = None
    last_image = None

    history[0] = obs[0]
    frame_id = 0
    connected = False

    try:
        if use_unity:
            assert env.getNumOfCameras(0) == 1
            assert env.connectUnity()
            connected = True

        start = time.perf_counter()

        for k in range(STEPS):
            action = make_action(k * DT)
            actions[k] = action[0]

            t0 = time.perf_counter()

            if case == "decimated":
                env.testStep(
                    action, obs, reward, done, extra
                )
            else:
                assert env.step(
                    action, obs, reward, done, extra
                )

            step_times[k] = time.perf_counter() - t0

            assert not done[0], f"Terminal state at step {k+1}"
            assert np.isfinite(obs).all()
            assert np.isfinite(reward).all()

            history[k + 1] = obs[0]
            rewards[k] = reward[0]

            if use_unity and (k + 1) % interval == 0:
                frame_id += 1

                if case == "decimated":
                    t0 = time.perf_counter()
                    assert env.render(frame_id)
                    render_times.append(
                        time.perf_counter() - t0
                    )

                image = env.getRGBImage(0, 0)

                assert image is not None
                assert image.shape == (360, 640, 3)
                assert image.dtype == np.uint8
                assert env.getRGBImage(0, 0) is None

                if first_image is None:
                    first_image = image.copy()

                last_image = image.copy()

                frame_steps.append(k + 1)
                frame_times.append((k + 1) * DT)

            elif case == "decimated":
                # 非渲染步不应产生新的图像。
                assert env.getRGBImage(0, 0) is None

            if (k + 1) % 100 == 0:
                print(
                    f"Step {k+1:3d}/{STEPS}, "
                    f"z={obs[0, 2]:.6f} m, "
                    f"frames={frame_id}"
                )

        elapsed = time.perf_counter() - start

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    expected_frames = (
        STEPS // interval if use_unity else 0
    )

    assert frame_id == expected_frames

    np.savez(
        RESULT_DIR / f"{case}.npz",
        time=np.arange(STEPS + 1) * DT,
        obs=history,
        actions=actions,
        rewards=rewards,
        step_times=step_times,
        frame_steps=np.asarray(frame_steps),
        frame_times=np.asarray(frame_times),
        render_times=np.asarray(render_times),
        elapsed=elapsed,
        frames=frame_id,
        interval=interval,
    )

    if use_unity:
        plt.imsave(
            RESULT_DIR / f"{case}_first.png",
            first_image,
        )
        plt.imsave(
            RESULT_DIR / f"{case}_last.png",
            last_image,
        )

    print("\n========== Results ==========")
    print(f"Case             : {case}")
    print(f"Simulation time  : {DURATION:.3f} s")
    print(f"Wall time        : {elapsed:.6f} s")
    print(f"RTF              : {DURATION / elapsed:.6f}")
    print(f"Dynamics steps   : {STEPS}")
    print(f"RGB frames       : {frame_id}")
    print(f"Mean step time   : {step_times.mean()*1000:.6f} ms")

    if render_times:
        print(
            f"Mean render time : "
            f"{np.mean(render_times)*1000:.6f} ms"
        )

    print(f"Final position   : {history[-1, :3]}")
    print(f"Final velocity   : {history[-1, 6:9]}")
    print("PASS")


def analyze():
    results = {
        case: np.load(RESULT_DIR / f"{case}.npz")
        for case in CASES
    }

    baseline = results["physics"]

    print("\n========== Comparison ==========")

    errors = []

    for case, data in results.items():
        np.testing.assert_array_equal(
            data["actions"], baseline["actions"]
        )

        diff = np.abs(data["obs"] - baseline["obs"])
        error = np.max(diff)
        errors.append(error)

        reward_error = np.max(
            np.abs(data["rewards"] - baseline["rewards"])
        )

        elapsed = float(data["elapsed"])

        print(
            f"{case:12s} "
            f"wall={elapsed:10.6f} s  "
            f"RTF={DURATION/elapsed:10.4f}  "
            f"frames={int(data['frames']):3d}  "
            f"obs_error={error:.8e}  "
            f"reward_error={reward_error:.8e}"
        )

    sync_time = float(results["sync"]["elapsed"])
    dec_time = float(results["decimated"]["elapsed"])

    print(f"\nSpeedup: {sync_time / dec_time:.4f} x")
    print(
        f"Wall time reduction: "
        f"{(1-dec_time/sync_time)*100:.2f}%"
    )

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9),
        constrained_layout=True,
    )

    colors = ["tab:blue", "tab:orange", "tab:green"]
    names = list(CASES)
    labels = [LABELS[c] for c in names]

    for case, color in zip(names, colors):
        data = results[case]

        axes[0, 0].plot(
            data["time"], data["obs"][:, 2],
            label=LABELS[case], color=color,
        )

        axes[0, 1].plot(
            data["time"], data["obs"][:, 8],
            label=LABELS[case], color=color,
        )

    axes[0, 0].set_title("Vertical position")
    axes[0, 0].set_ylabel("Height (m)")

    axes[0, 1].set_title("Vertical velocity")
    axes[0, 1].set_ylabel("Velocity (m/s)")

    for ax in axes[0, :2]:
        ax.set_xlabel("Simulation time (s)")
        ax.legend()
        ax.grid(alpha=0.3)

    wall_times = [
        float(results[c]["elapsed"]) for c in names
    ]

    bars = axes[0, 2].bar(
        labels, wall_times, color=colors
    )

    axes[0, 2].bar_label(bars, fmt="%.3f")
    axes[0, 2].axhline(
        DURATION, linestyle="--",
        color="gray", label="Real-time threshold",
    )
    axes[0, 2].set_title("Wall-clock execution time")
    axes[0, 2].set_ylabel("Time (s)")
    axes[0, 2].legend()

    rtfs = [DURATION / t for t in wall_times]

    bars = axes[1, 0].bar(
        labels, rtfs, color=colors
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].bar_label(bars, fmt="%.2f")
    axes[1, 0].axhline(1.0, linestyle="--", color="gray")
    axes[1, 0].set_title("Real-time factor (log scale)")
    axes[1, 0].set_ylabel("RTF")

    for case, color in zip(names, colors):
        data = results[case]

        cumulative = np.searchsorted(
            data["frame_steps"],
            np.arange(STEPS + 1),
            side="right",
        )

        axes[1, 1].step(
            data["time"], cumulative,
            where="post",
            label=LABELS[case],
            color=color,
        )

    axes[1, 1].set_title("Cumulative RGB frames")
    axes[1, 1].set_xlabel("Simulation time (s)")
    axes[1, 1].set_ylabel("Frame count")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    bars = axes[1, 2].bar(
        labels, errors, color=colors
    )
    axes[1, 2].bar_label(bars, fmt="%.2e")
    axes[1, 2].set_title("Maximum observation difference")
    axes[1, 2].set_ylabel("Absolute difference (mixed units)")

    fig.suptitle(
        "Experiment 26: Dynamics and Rendering Decimation",
        fontsize=18,
    )

    output = ROOT / "26_render_decimation.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"\nSummary figure: {output}")

    for case, data in results.items():
        np.testing.assert_allclose(
            data["obs"], baseline["obs"],
            rtol=0, atol=1e-6,
        )

        np.testing.assert_allclose(
            data["rewards"], baseline["rewards"],
            rtol=0, atol=1e-6,
        )

        interval = CASES[case]
        expected = (
            np.arange(interval, STEPS + 1, interval)
            if interval else np.array([])
        )

        np.testing.assert_array_equal(
            data["frame_steps"], expected
        )

    print("Dynamics consistency: PASS")
    print("Reward consistency: PASS")
    print("Rendering schedule: PASS")

    for data in results.values():
        data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case",
        choices=[*CASES, "analyze"],
    )

    args = parser.parse_args()

    if args.case == "analyze":
        analyze()
    else:
        run(args.case)
