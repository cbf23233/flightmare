"""Validate the Python RGB camera interface with Unity."""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


WIDTH = 640
HEIGHT = 360
FOV = 70.0
NUM_FRAMES = 10

OUTPUT_DIR = Path(__file__).resolve().parent

CONFIG = f"""
env:
  seed: 1
  scene_id: 2
  num_envs: 1
  num_threads: 1
  render: true

camera:
  enabled: true
  width: {WIDTH}
  height: {HEIGHT}
  fov: {FOV}
  position: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0]
"""


def validate_image(image):
    assert isinstance(image, np.ndarray), "No RGB image received"

    assert image.shape == (HEIGHT, WIDTH, 3), (
        f"Unexpected image shape: {image.shape}"
    )

    assert image.dtype == np.uint8, (
        f"Unexpected image dtype: {image.dtype}"
    )

    assert image.flags.c_contiguous, "Image is not contiguous"
    assert image.flags.owndata, "Image does not own its memory"

    assert image.max() > image.min(), "Image is uniform"


def main():
    print("========== Experiment 20: Unity RGB camera ==========")

    env = QuadrotorEnv_v1(CONFIG, False)
    connected = False

    obs = np.zeros((1, 12), dtype=np.float32)

    state = np.array(
        [[0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )

    images = []
    render_times = []
    read_times = []

    try:
        # 1. Check camera configuration.
        print("\n========== Test 1: Camera configuration ==========")

        assert env.getNumOfEnvs() == 1
        assert env.getNumOfCameras(0) == 1
        assert env.getObsDim() == 12
        assert env.getActDim() == 4
        assert env.getRGBImage(0, 0) is None

        print("Camera configuration: PASS")

        # 2. Set a deterministic static pose.
        print("\n========== Test 2: Initial state ==========")

        assert env.resetState(state, obs)

        initial_obs = obs.copy()

        print("Initial observation:", initial_obs)
        print("Deterministic initial state: PASS")

        # 3. Connect to Unity.
        print("\n========== Test 3: Unity connection ==========")

        connected = env.connectUnity()

        if not connected:
            raise RuntimeError("Failed to connect to Unity")

        print("Unity connection: PASS")

        # 4. Render and retrieve RGB images.
        print("\n========== Test 4: RGB image acquisition ==========")

        first_image_snapshot = None

        for frame_id in range(1, NUM_FRAMES + 1):
            t0 = perf_counter()

            assert env.render(frame_id), "Render failed"

            t1 = perf_counter()

            image = env.getRGBImage(0, 0)

            t2 = perf_counter()

            validate_image(image)

            render_times.append((t1 - t0) * 1000)
            read_times.append((t2 - t1) * 1000)

            images.append(image)

            if frame_id == 1:
                first_image_snapshot = image.copy()

            print(
                f"Frame {frame_id:02d}: "
                f"shape={image.shape}, "
                f"dtype={image.dtype}, "
                f"range=[{image.min()}, {image.max()}], "
                f"render={render_times[-1]:.3f} ms, "
                f"read={read_times[-1]:.3f} ms"
            )

            # The image queue is consumed by getRGBImage().
            assert env.getRGBImage(0, 0) is None, (
                "Image queue was not consumed"
            )

        print("RGB image acquisition: PASS")
        print("Image queue consumption: PASS")

        # 5. Check independent NumPy memory.
        print("\n========== Test 5: Image memory ownership ==========")

        for image in images[1:]:
            assert not np.shares_memory(images[0], image)

        np.testing.assert_array_equal(
            images[0], first_image_snapshot
        )

        print("Independent image memory: PASS")
        print("Previous image preserved: PASS")

        # 6. Compare images from the static camera.
        print("\n========== Test 6: Static image consistency ==========")

        first = images[0].astype(np.float32)

        differences = []

        for image in images[1:]:
            difference = np.abs(
                image.astype(np.float32) - first
            )

            differences.append(float(difference.mean()))

        print(
            "Mean pixel difference:",
            f"{np.mean(differences):.6f}"
        )

        print(
            "Maximum frame mean difference:",
            f"{np.max(differences):.6f}"
        )

        # Differences are diagnostic, not a strict pass criterion.
        # Unity may contain time-dependent rendering effects.

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    # 7. Verify image remains accessible after environment cleanup.
    del env

    validate_image(images[0])

    np.testing.assert_array_equal(
        images[0], first_image_snapshot
    )

    print("\nImage retained after environment cleanup: PASS")

    # 8. Save the first and last RGB images.
    first_path = OUTPUT_DIR / "20_unity_rgb_first.png"
    last_path = OUTPUT_DIR / "20_unity_rgb_last.png"
    summary_path = OUTPUT_DIR / "20_unity_rgb_camera.png"

    plt.imsave(first_path, images[0])
    plt.imsave(last_path, images[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(images[0])
    axes[0].set_title("First RGB frame")
    axes[0].axis("off")

    axes[1].imshow(images[-1])
    axes[1].set_title("Last RGB frame")
    axes[1].axis("off")

    fig.suptitle(
        f"Static onboard RGB camera | "
        f"{WIDTH} x {HEIGHT} | FOV {FOV:.0f} deg"
    )

    fig.tight_layout()
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)

    # 9. Report timing and image information.
    print("\n========== Experiment summary ==========")

    print(f"Frames received: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Image dtype: {images[0].dtype}")
    print(f"Image bytes: {images[0].nbytes}")

    print(
        f"Mean render latency: "
        f"{np.mean(render_times):.3f} ms"
    )

    print(
        f"P95 render latency: "
        f"{np.percentile(render_times, 95):.3f} ms"
    )

    print(
        f"Mean image read latency: "
        f"{np.mean(read_times):.3f} ms"
    )

    print(
        f"Mean RGB values: "
        f"{images[0].mean(axis=(0, 1))}"
    )

    print(f"Saved: {first_path}")
    print(f"Saved: {last_path}")
    print(f"Saved: {summary_path}")

    print("\nExperiment 20 static RGB camera: PASS")


if __name__ == "__main__":
    main()
