"""Diagnose static RGB rendering stability."""

from pathlib import Path
from time import sleep, perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


NUM_FRAMES = 100
WIDTH = 640
HEIGHT = 360

OUTPUT_DIR = Path(__file__).resolve().parent

CONFIG = """
env:
  seed: 1
  scene_id: 2
  num_envs: 1
  num_threads: 1
  render: true

camera:
  enabled: true
  width: 640
  height: 360
  fov: 70.0
  position: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0]
"""


def main():
    env = QuadrotorEnv_v1(CONFIG, False)
    connected = False

    state = np.array(
        [[0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )

    obs = np.zeros((1, 12), dtype=np.float32)

    selected_ids = [1, 5, 10, 20, 50, 100]
    selected_images = {}

    full_diffs = []
    ground_diffs = []
    timestamps = []

    previous = None

    try:
        assert env.resetState(state, obs)

        connected = env.connectUnity()
        assert connected

        start = perf_counter()

        for frame_id in range(1, NUM_FRAMES + 1):
            assert env.render(frame_id)

            image = env.getRGBImage(0, 0)

            assert image is not None
            assert image.shape == (HEIGHT, WIDTH, 3)
            assert image.dtype == np.uint8

            if frame_id in selected_ids:
                selected_images[frame_id] = image.copy()

            current = image.astype(np.float32)

            if previous is not None:
                diff = np.abs(current - previous)

                full_diff = float(diff.mean())

                # Lower image region, mainly ground and scene.
                ground_diff = float(diff[180:, :, :].mean())

                full_diffs.append(full_diff)
                ground_diffs.append(ground_diff)
                timestamps.append(perf_counter() - start)

                if frame_id <= 10 or frame_id % 10 == 0:
                    print(
                        f"Frame {frame_id:03d}: "
                        f"full MAE={full_diff:.6f}, "
                        f"lower MAE={ground_diff:.6f}"
                    )

            previous = current

            sleep(0.05)

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    print("\n========== Stability summary ==========")

    for name, values in [
        ("Full image", full_diffs),
        ("Lower region", ground_diffs),
    ]:
        print(name)
        print(f"  First 10: {np.mean(values[:10]):.6f}")
        print(f"  Last 10:  {np.mean(values[-10:]):.6f}")
        print(f"  Last 20:  {np.mean(values[-20:]):.6f}")

    # Selected frame comparison.
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    for ax, frame_id in zip(axes.flat, selected_ids):
        ax.imshow(selected_images[frame_id])
        ax.set_title(f"Frame {frame_id}")
        ax.axis("off")

    fig.suptitle("Static camera rendering stability")
    fig.tight_layout()

    path = OUTPUT_DIR / "20_rgb_stability_frames.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")

    # Pixel difference history.
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(timestamps, full_diffs, label="Full image")
    ax.plot(timestamps, ground_diffs, label="Lower region")

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Mean absolute pixel difference")
    ax.set_title("Consecutive frame differences")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIR / "20_rgb_stability_error.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")
    print("\nStability diagnostic completed.")


if __name__ == "__main__":
    main()
