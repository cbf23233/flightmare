"""Validate camera mounting rotation."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


OUTPUT_DIR = Path(__file__).resolve().parent

CAMERA_POSITION = [0, 0, 2]

CASES = {
    "initial": [0, 0, 0],
    "roll_positive": [90, 0, 0],
    "roll_negative": [-90, 0, 0],
    "pitch_positive": [0, 90, 0],
    "pitch_negative": [0, -90, 0],
    "yaw_positive": [0, 0, 90],
    "yaw_negative": [0, 0, -90],
}


def make_config(rotation):
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
  position: {CAMERA_POSITION}
  rotation: {rotation}
"""


def rotation_matrix(rotation):
    roll, pitch, yaw = np.deg2rad(rotation)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    return rz @ ry @ rx


def capture(case):
    rotation = CASES[case]

    env = QuadrotorEnv_v1(
        make_config(rotation), False
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

        matrix = rotation_matrix(rotation)

        optical_axis = matrix @ np.array([0, 1, 0])
        image_right = matrix @ np.array([1, 0, 0])
        image_up = matrix @ np.array([0, 0, 1])

        path = OUTPUT_DIR / (
            f"20_camera_rotation_{case}.png"
        )

        plt.imsave(path, image)

        print(f"Case: {case}")
        print(f"UAV position: {obs[0, :3]}")
        print(f"Camera position: {CAMERA_POSITION}")
        print(f"Camera rotation: {rotation}")

        print(
            "Expected optical axis:",
            np.round(optical_axis, 6)
        )

        print(
            "Expected image right:",
            np.round(image_right, 6)
        )

        print(
            "Expected image up:",
            np.round(image_up, 6)
        )

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
        3: "yaw_positive",
        5: "yaw_negative",
        1: "roll_positive",
        7: "roll_negative",
        0: "pitch_positive",
        2: "pitch_negative",
    }

    for ax in axes.flat:
        ax.axis("off")

    for subplot_id, case in layout.items():
        path = OUTPUT_DIR / (
            f"20_camera_rotation_{case}.png"
        )

        image = plt.imread(path)

        ax = axes.flat[subplot_id]
        ax.imshow(image)
        ax.set_title(f"{case}: {CASES[case]}")

    fig.suptitle(
        "Camera mounting rotation identification"
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "20_rgb_camera_rotation.png"

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
