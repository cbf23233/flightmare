"""Diagnose Unity camera response to pose changes."""

from pathlib import Path
from time import perf_counter, sleep

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
    ("X +5 m", [5, 0, 5]),
    ("Y +5 m", [0, 5, 5]),
    ("Z +5 m", [0, 0, 10]),
    ("Return", [0, 0, 5]),
]

HOLD_TIME = 5.0
RENDER_PERIOD = 0.1


def main():
    env = QuadrotorEnv_v1(CONFIG, False)

    obs = np.zeros((1, 12), dtype=np.float32)
    state = np.zeros((1, 13), dtype=np.float32)

    state[0, 3] = 1.0
    state[0, :3] = CASES[0][1]

    connected = False
    frame_id = 0
    images = []
    baseline = None

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

            print(f"\n========== {name} ==========")
            print(f"Position: {obs[0, :3]}")

            start = perf_counter()
            next_report = 0.0
            count = 0

            while perf_counter() - start < HOLD_TIME:
                tick = perf_counter()

                frame_id += 1

                assert env.render(frame_id)

                image = env.getRGBImage(0, 0)

                assert image is not None
                assert image.shape == (360, 640, 3)
                assert image.dtype == np.uint8

                count += 1

                if baseline is None:
                    baseline = image.copy()

                diff = np.abs(
                    image[180:].astype(np.float32)
                    - baseline[180:].astype(np.float32)
                ).mean()

                elapsed = perf_counter() - start

                if elapsed >= next_report:
                    print(
                        f"t={elapsed:.2f} s, "
                        f"frame={frame_id}, "
                        f"background MAE={diff:.6f}"
                    )
                    next_report += RENDER_PERIOD

                sleep(max(
                    0.0,
                    RENDER_PERIOD - (perf_counter() - tick)
                ))

            images.append(image.copy())

            print(f"Frames received: {count}")

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax in axes.flat:
        ax.axis("off")

    for ax, image, case in zip(axes.flat, images, CASES):
        ax.imshow(image)
        ax.set_title(f"{case[0]}: {case[1]}")

    fig.suptitle("Camera response to sustained pose changes")
    fig.tight_layout()

    path = OUTPUT_DIR / "20_rgb_pose_response.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\nSaved: {path}")
    print("Pose response diagnostic completed.")


if __name__ == "__main__":
    main()
