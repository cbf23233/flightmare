"""Identify the default camera optical axis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


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

CASES = [
    ("Initial", [0, 0, 5]),
    ("X positive", [2, 0, 5]),
    ("X negative", [-2, 0, 5]),
    ("Y positive", [0, 2, 5]),
    ("Y negative", [0, -2, 5]),
    ("Z positive", [0, 0, 7]),
    ("Z negative", [0, 0, 3]),
    ("Return", [0, 0, 5]),
]


def main():
    env = QuadrotorEnv_v1(CONFIG, False)
    connected = False

    obs = np.zeros((1, 12), dtype=np.float32)
    state = np.zeros((1, 13), dtype=np.float32)

    # Identity quaternion: w, x, y, z.
    state[0, 3] = 1.0
    state[0, :3] = CASES[0][1]

    images = []
    frame_id = 0

    try:
        assert env.resetState(state, obs)

        connected = env.connectUnity()
        assert connected

        for name, position in CASES:
            state[0, :3] = position

            assert env.resetState(state, obs)

            np.testing.assert_allclose(
                obs[0, :3], position, atol=1e-6
            )

            # Capture the first synchronized frame at each pose.
            frame_id += 1

            assert env.render(frame_id)

            image = env.getRGBImage(0, 0)

            assert image is not None
            assert image.shape == (360, 640, 3)
            assert image.dtype == np.uint8

            images.append(image.copy())

            print(
                f"{name:12s}: "
                f"position={obs[0, :3]}, "
                f"frame={frame_id}"
            )

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 9)
    )

    layout = {
        4: 0,
        3: 1,
        5: 2,
        1: 3,
        7: 4,
        0: 5,
        8: 6,
        6: 7,
    }

    for ax in axes.flat:
        ax.axis("off")

    for subplot_id, case_id in layout.items():
        ax = axes.flat[subplot_id]

        name, position = CASES[case_id]

        ax.imshow(images[case_id])
        ax.set_title(f"{name}: {position}")

    fig.suptitle(
        "Default camera optical axis identification"
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "20_rgb_optical_axis.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\nSaved: {path}")
    print("Optical axis diagnostic completed.")


if __name__ == "__main__":
    main()
