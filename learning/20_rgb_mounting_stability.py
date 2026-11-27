"""Validate camera mounting with continuous RGB rendering."""

import argparse
import importlib
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


mounting = importlib.import_module(
    "20_rgb_camera_mounting"
)

OUTPUT_DIR = Path(__file__).resolve().parent

CASES = ["front_40cm", "down_10cm"]

NUM_FRAMES = 100
PERIOD = 0.1


def main(case):
    position, rotation = mounting.CASES[case]

    env = QuadrotorEnv_v1(
        mounting.make_config(position, rotation),
        False
    )

    obs = np.zeros((1, 12), dtype=np.float32)
    state = np.zeros((1, 13), dtype=np.float32)

    state[0, :3] = [0, 0, 5]
    state[0, 3] = 1.0

    connected = False
    images = []
    mae = []
    render_times = []

    try:
        assert env.resetState(state, obs)

        connected = env.connectUnity()
        assert connected

        start = time.perf_counter()

        for i in range(NUM_FRAMES):
            loop_start = time.perf_counter()

            assert env.render(i + 1)

            render_times.append(
                time.perf_counter() - loop_start
            )

            image = env.getRGBImage(0, 0)

            assert image is not None
            assert image.shape == (360, 640, 3)
            assert image.dtype == np.uint8

            if images:
                diff = np.abs(
                    image.astype(np.int16)
                    - images[-1].astype(np.int16)
                )

                mae.append(float(diff.mean()))

            images.append(image.copy())

            if (i + 1) % 10 == 0:
                print(
                    f"Frame {i + 1:3d}/100, "
                    f"MAE={mae[-1]:.6f}"
                )

            deadline = start + (i + 1) * PERIOD

            time.sleep(
                max(0.0, deadline - time.perf_counter())
            )

        elapsed = time.perf_counter() - start

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    mae = np.asarray(mae)
    render_times = np.asarray(render_times)

    print(f"\nCase: {case}")
    print(f"Position: {position}")
    print(f"Rotation: {rotation}")
    print(f"Captured frames: {len(images)}")
    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"Capture FPS: {len(images)/elapsed:.3f}")

    print(f"Mean MAE: {mae.mean():.6f}")
    print(f"Max MAE: {mae.max():.6f}")

    print(
        f"Mean render time: "
        f"{render_times.mean()*1000:.3f} ms"
    )

    print(
        f"P95 render time: "
        f"{np.percentile(render_times, 95)*1000:.3f} ms"
    )

    # Show the largest temporal image difference.
    max_index = int(np.argmax(mae)) + 1

    diff = np.abs(
        images[max_index].astype(np.int16)
        - images[max_index - 1].astype(np.int16)
    ).mean(axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(images[0])
    axes[0, 0].set_title("First frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(images[max_index])
    axes[0, 1].set_title(
        f"Maximum-change frame: {max_index + 1}"
    )
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(
        diff, cmap="inferno", vmin=0, vmax=50
    )

    axes[1, 0].set_title("Maximum-change difference")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0])

    axes[1, 1].plot(
        np.arange(2, NUM_FRAMES + 1), mae
    )

    axes[1, 1].set_xlabel("Frame ID")
    axes[1, 1].set_ylabel("Adjacent-frame MAE")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Camera mounting stability: {case}")
    fig.tight_layout()

    path = OUTPUT_DIR / (
        f"20_mounting_stability_{case}.png"
    )

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")

    # Save all frames for visual inspection.
    fig, axes = plt.subplots(
        10, 10, figsize=(24, 15)
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(str(i + 1), fontsize=8)
        ax.axis("off")

    fig.tight_layout()

    path = OUTPUT_DIR / (
        f"20_mounting_frames_{case}.png"
    )

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")
    print("PASS: Continuous RGB capture completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()

    main(args.case)
