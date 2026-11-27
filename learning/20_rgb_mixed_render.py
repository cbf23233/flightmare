"""Validate mixed automatic and explicit RGB rendering."""

from time import perf_counter

import numpy as np
from flightgym import QuadrotorEnv_v1


WIDTH = 640
HEIGHT = 360
DT = 0.02

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

STATE = np.array(
    [[0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.float32,
)


def make_buffers(env):
    return (
        np.zeros((1, env.getObsDim()), dtype=np.float32),
        np.full((1, env.getActDim()), 0.05, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(
            (1, len(env.getExtraInfoNames())),
            dtype=np.float32,
        ),
    )


def validate_image(env):
    image = env.getRGBImage(0, 0)

    assert isinstance(image, np.ndarray)
    assert image.shape == (HEIGHT, WIDTH, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert image.flags.owndata
    assert image.max() > image.min()

    # Reading the image must consume the queue.
    assert env.getRGBImage(0, 0) is None

    return image


def reference_dynamics(num_steps):
    """Generate reference observations without Unity."""

    config = CONFIG.replace(
        "render: true", "render: false"
    ).replace(
        "enabled: true", "enabled: false"
    )

    env = QuadrotorEnv_v1(config, False)

    obs, act, reward, done, extra = make_buffers(env)

    history = []

    try:
        assert env.resetState(STATE, obs)

        for _ in range(num_steps):
            assert env.step(
                act, obs, reward, done, extra
            )
            assert not np.any(done)
            history.append(obs.copy())

    finally:
        env.close()

    return history


def main():
    print("========== Experiment 20: Mixed RGB rendering ==========")

    # Five dynamics steps without Unity as the reference.
    reference = reference_dynamics(5)

    env = QuadrotorEnv_v1(CONFIG, False)

    obs, act, reward, done, extra = make_buffers(env)

    connected = False

    step_times = []
    render_times = []
    images = []

    step_count = 0
    render_count = 0
    rejected_count = 0

    def automatic_step(expected_frame):
        nonlocal step_count

        start = perf_counter()

        assert env.step(
            act, obs, reward, done, extra
        ), "Automatic rendering failed"

        elapsed = (perf_counter() - start) * 1000

        assert not np.any(done)
        assert np.all(np.isfinite(obs))

        np.testing.assert_allclose(
            obs,
            reference[step_count],
            rtol=0,
            atol=1e-6,
        )

        step_count += 1
        step_times.append(elapsed)

        image = validate_image(env)
        images.append(image)

        print(
            f"STEP   frame={expected_frame:2d} | "
            f"z={obs[0, 2]:.6f} m | "
            f"vz={obs[0, 8]:+.6f} m/s | "
            f"time={elapsed:.3f} ms | PASS"
        )

    def explicit_render(frame_id):
        nonlocal render_count

        start = perf_counter()

        assert env.render(frame_id), (
            f"Explicit render failed: {frame_id}"
        )

        elapsed = (perf_counter() - start) * 1000

        image = validate_image(env)
        images.append(image)

        render_count += 1
        render_times.append(elapsed)

        print(
            f"RENDER frame={frame_id:2d} | "
            f"time={elapsed:.3f} ms | PASS"
        )

    def reject_render(frame_id):
        nonlocal rejected_count

        assert env.getRGBImage(0, 0) is None

        assert not env.render(frame_id), (
            f"Invalid frame ID accepted: {frame_id}"
        )

        assert env.getRGBImage(0, 0) is None, (
            "Rejected request generated an image"
        )

        rejected_count += 1

        print(
            f"REJECT frame={frame_id:2d} | PASS"
        )

    try:
        assert env.getNumOfCameras(0) == 1
        assert env.getRGBImage(0, 0) is None
        assert env.resetState(STATE, obs)

        connected = env.connectUnity()
        assert connected

        print("\n========== Mixed rendering sequence ==========")

        automatic_step(1)
        automatic_step(2)

        explicit_render(3)

        automatic_step(4)

        explicit_render(10)

        reject_render(10)
        reject_render(5)
        reject_render(0)

        automatic_step(11)

        explicit_render(12)

        automatic_step(13)

        print("\n========== Dynamics consistency ==========")

        assert step_count == 5

        np.testing.assert_allclose(
            obs,
            reference[-1],
            rtol=0,
            atol=1e-6,
        )

        print("Reference observation:", reference[-1])
        print("Mixed observation:    ", obs)

        print(
            "Maximum observation error:",
            np.max(np.abs(obs - reference[-1])),
        )

        print("Dynamics consistency: PASS")

        print("\n========== Image memory ==========")

        for image in images[1:]:
            assert not np.shares_memory(
                images[0], image
            )

        print("Independent image memory: PASS")

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    print("\n========== Experiment summary ==========")

    print(f"Dynamics steps: {step_count}")
    print(f"Explicit renders: {render_count}")
    print(f"Rejected requests: {rejected_count}")
    print(f"RGB images received: {len(images)}")
    print(f"Simulation time: {step_count * DT:.3f} s")

    print(
        f"Mean automatic step latency: "
        f"{np.mean(step_times):.3f} ms"
    )

    print(
        f"Mean explicit render latency: "
        f"{np.mean(render_times):.3f} ms"
    )

    print("\nPASS: Mixed RGB rendering")


if __name__ == "__main__":
    main()
