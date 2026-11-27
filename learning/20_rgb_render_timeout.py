"""Validate Unity rendering timeout and recovery."""

import os
import signal
import sys
from time import perf_counter, sleep

import numpy as np
from flightgym import QuadrotorEnv_v1


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


def validate_image(env):
    image = env.getRGBImage(0, 0)

    assert isinstance(image, np.ndarray)
    assert image.shape == (360, 640, 3)
    assert image.dtype == np.uint8
    assert image.max() > image.min()
    assert env.getRGBImage(0, 0) is None


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: script.py UNITY_PID")

    unity_pid = int(sys.argv[1])

    assert unity_pid > 1

    exe = os.readlink(f"/proc/{unity_pid}/exe")

    assert os.path.basename(exe) == "RPG_Flightmare.x86_64", (
        f"Unexpected process: {exe}"
    )

    print(f"Unity PID: {unity_pid}")
    print(f"Unity executable: {exe}")

    env = QuadrotorEnv_v1(CONFIG, False)

    obs = np.zeros((1, 12), dtype=np.float32)

    connected = False
    paused = False

    try:
        assert env.resetState(STATE, obs)

        connected = env.connectUnity()
        assert connected

        print("\n========== Test 1: Normal rendering ==========")

        assert env.render(1)
        validate_image(env)

        print("Normal rendering: PASS")

        print("\n========== Test 2: Pause Unity ==========")

        os.kill(unity_pid, signal.SIGSTOP)
        paused = True

        sleep(0.2)

        print("Unity paused")

        print("\n========== Test 3: Receive timeout ==========")

        start = perf_counter()

        ok = env.render(2)

        elapsed = perf_counter() - start

        print(f"Render result: {ok}")
        print(f"Timeout elapsed: {elapsed:.6f} s")

        assert not ok, "Render unexpectedly succeeded"

        assert 1.9 <= elapsed <= 3.0, (
            f"Unexpected timeout duration: {elapsed}"
        )

        assert env.getRGBImage(0, 0) is None

        print("Receive timeout: PASS")

        print("\n========== Test 4: Resume Unity ==========")

        os.kill(unity_pid, signal.SIGCONT)
        paused = False

        sleep(0.5)

        print("Unity resumed")

        print("\n========== Test 5: Reject reused frame ID ==========")

        assert not env.render(2)

        print("Reused frame ID rejected: PASS")

        print("\n========== Test 6: Rendering recovery ==========")

        start = perf_counter()

        assert env.render(3), "Rendering recovery failed"

        elapsed = (perf_counter() - start) * 1000

        validate_image(env)

        print(f"Recovery latency: {elapsed:.3f} ms")
        print("Rendering recovery: PASS")

        print("\nPASS: Unity rendering timeout and recovery")

    finally:
        if paused:
            os.kill(unity_pid, signal.SIGCONT)

        if connected:
            env.disconnectUnity()

        env.close()


if __name__ == "__main__":
    main()
