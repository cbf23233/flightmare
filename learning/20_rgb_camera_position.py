"""Validate camera mounting translation."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


OUTPUT_DIR = Path(__file__).resolve().parent

CASES = {
    "initial": [0, 0, 0],
    "x_positive": [2, 0, 0],
    "x_negative": [-2, 0, 0],
    "y_positive": [0, 2, 0],
    "y_negative": [0, -2, 0],
    "z_positive": [0, 0, 2],
    "z_negative": [0, 0, -2],
}


def make_config(position):
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
  rotation: [0.0, 0.0, 0.0]
"""


def capture(case):
    position = CASES[case]

    env = QuadrotorEnv_v1(
        make_config(position), False
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

        expected = state[0, :3] + np.array(position)

        path = OUTPUT_DIR / (
            f"20_camera_position_{case}.png"
        )

        plt.imsave(path, image)

        print(f"Case: {case}")
        print(f"UAV position: {obs[0, :3]}")
        print(f"Camera mounting position: {position}")
        print(f"Expected camera world position: {expected}")
        print(f"Saved: {path}")
        print("PASS: RGB image captured.")

    finally:
        if connected:
            env.disconnectUnity()

        env.close()


def plot_results():
    fig, axes = plt.subplots(
        3, 3, figsize=(15, 9)
    )

    layout = {
        4: "initial",
        3: "x_positive",
        5: "x_negative",
        1: "y_positive",
        7: "y_negative",
        0: "z_positive",
        8: "z_negative",
    }

    for ax in axes.flat:
        ax.axis("off")

    for subplot_id, case in layout.items():
        path = OUTPUT_DIR / (
            f"20_camera_position_{case}.png"
        )

        image = plt.imread(path)

        ax = axes.flat[subplot_id]
        ax.imshow(image)
        ax.set_title(f"{case}: {CASES[case]}")

    fig.suptitle(
        "Camera mounting position identification"
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "20_rgb_camera_position.png"

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
