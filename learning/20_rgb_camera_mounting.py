"""Evaluate practical forward and downward camera mounting."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


OUTPUT_DIR = Path(__file__).resolve().parent

CASES = {
    "front_10cm": ([0, 0.10, 0], [0, 0, 0]),
    "front_20cm": ([0, 0.20, 0], [0, 0, 0]),
    "front_30cm": ([0, 0.30, 0], [0, 0, 0]),
    "front_40cm": ([0, 0.40, 0], [0, 0, 0]),
    "down_05cm": ([0, 0, -0.05], [-90, 0, 0]),
    "down_10cm": ([0, 0, -0.10], [-90, 0, 0]),
    "down_15cm": ([0, 0, -0.15], [-90, 0, 0]),
    "down_20cm": ([0, 0, -0.20], [-90, 0, 0]),
}


def make_config(position, rotation):
    return f"""
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
  position: {position}
  rotation: {rotation}
"""


def capture(case):
    position, rotation = CASES[case]

    env = QuadrotorEnv_v1(
        make_config(position, rotation), False
    )

    connected = False

    obs = np.zeros((1, 12), dtype=np.float32)
    state = np.zeros((1, 13), dtype=np.float32)

    state[0, :3] = [0, 0, 5]
    state[0, 3] = 1.0

    try:
        assert env.resetState(state, obs)

        np.testing.assert_allclose(
            obs[0, :3], [0, 0, 5], atol=1e-6
        )

        np.testing.assert_allclose(
            obs[0, 3:6], 0.0, atol=1e-6
        )

        assert env.getNumOfCameras(0) == 1

        connected = env.connectUnity()
        assert connected

        assert env.render(1)

        image = env.getRGBImage(0, 0)

        assert image is not None
        assert image.shape == (360, 640, 3)
        assert image.dtype == np.uint8
        assert image.max() > image.min()

        path = OUTPUT_DIR / (
            f"20_camera_mounting_{case}.png"
        )

        plt.imsave(path, image)

        print(f"Case: {case}")
        print(f"UAV position: {obs[0, :3]}")
        print(f"Camera position: {position}")
        print(f"Camera rotation: {rotation}")
        print(f"Image shape: {image.shape}")
        print(f"Saved: {path}")
        print("PASS: RGB image captured.")

    finally:
        if connected:
            env.disconnectUnity()

        env.close()


def plot_results():
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 6)
    )

    for ax, (case, config) in zip(
        axes.flat, CASES.items()
    ):
        position, rotation = config

        path = OUTPUT_DIR / (
            f"20_camera_mounting_{case}.png"
        )

        image = plt.imread(path)

        ax.imshow(image)

        ax.set_title(
            f"{case}\n"
            f"position={position}\n"
            f"rotation={rotation}",
            fontsize=10
        )

        ax.axis("off")

    fig.suptitle(
        "Forward and downward camera mounting"
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "20_rgb_camera_mounting.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case",
        choices=[*CASES.keys(), "plot"]
    )

    args = parser.parse_args()

    if args.case == "plot":
        plot_results()
    else:
        capture(args.case)
