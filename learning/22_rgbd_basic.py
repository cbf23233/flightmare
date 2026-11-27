"""Validate RGB-D acquisition and continuous rendering."""

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
RESULT_DIR = OUTPUT_DIR / "22_rgbd_results"

WIDTH = 640
HEIGHT = 360
NUM_FRAMES = 100
PERIOD = 0.1

NEAR = 0.01
FAR = 100.0


def make_config():
    position, rotation = mounting.CASES["front_40cm"]

    return mounting.make_config(
        position, rotation
    ) + """
  depth: true
  segmentation: false
  optical_flow: false
"""


def valid_mask(depth):
    # Provisional geometric-depth validity convention.
    return (
        np.isfinite(depth)
        & (depth >= NEAR)
        & (depth < FAR)
    )


def check_rgb(rgb):
    assert rgb is not None
    assert rgb.shape == (HEIGHT, WIDTH, 3)
    assert rgb.dtype == np.uint8
    assert rgb.flags["OWNDATA"]
    assert rgb.flags["C_CONTIGUOUS"]
    assert rgb.max() > rgb.min()


def check_depth(depth):
    assert depth is not None
    assert depth.shape == (HEIGHT, WIDTH)
    assert depth.dtype == np.float32
    assert depth.flags["OWNDATA"]
    assert depth.flags["C_CONTIGUOUS"]


def capture():
    env = QuadrotorEnv_v1(make_config(), False)

    obs = np.zeros((1, 12), dtype=np.float32)
    state = np.zeros((1, 13), dtype=np.float32)

    state[0, :3] = [0, 0, 5]
    state[0, 3] = 1.0

    connected = False

    first_rgb = None
    first_depth = None
    first_rgb_copy = None
    first_depth_copy = None
    previous_depth = None

    render_times = []
    read_times = []
    depth_mae = []
    valid_ratios = []
    finite_ratios = []
    depth_medians = []
    deadline_misses = 0

    try:
        assert env.resetState(state, obs)

        np.testing.assert_allclose(
            obs[0, :3], [0, 0, 5], atol=1e-6
        )

        np.testing.assert_allclose(
            obs[0, 3:6], 0.0, atol=1e-6
        )

        assert env.getNumOfCameras(0) == 1

        assert env.getRGBImage() is None
        assert env.getDepthMap() is None

        connected = env.connectUnity()
        assert connected

        start = time.perf_counter()

        for i in range(NUM_FRAMES):
            frame_id = i + 1
            t0 = time.perf_counter()

            assert env.render(frame_id), (
                f"Render failed: {frame_id}"
            )

            t1 = time.perf_counter()

            # Alternate acquisition order to check
            # that the RGB and depth queues are independent.
            if i % 2 == 0:
                rgb = env.getRGBImage()
                assert env.getRGBImage() is None

                depth = env.getDepthMap()
                assert env.getDepthMap() is None
            else:
                depth = env.getDepthMap()
                assert env.getDepthMap() is None

                rgb = env.getRGBImage()
                assert env.getRGBImage() is None

            t2 = time.perf_counter()

            check_rgb(rgb)
            check_depth(depth)

            assert not np.shares_memory(rgb, depth)

            render_times.append(t1 - t0)
            read_times.append(t2 - t1)

            mask = valid_mask(depth)

            valid_ratios.append(float(mask.mean()))
            finite_ratios.append(
                float(np.isfinite(depth).mean())
            )

            depth_medians.append(
                float(np.median(depth[mask]))
                if mask.any() else np.nan
            )

            if first_rgb is None:
                first_rgb = rgb
                first_depth = depth

                first_rgb_copy = rgb.copy()
                first_depth_copy = depth.copy()

                print("\nFirst RGB-D frame:")
                print("RGB shape:", rgb.shape)
                print("RGB dtype:", rgb.dtype)
                print("RGB bytes:", rgb.nbytes)
                print("Depth shape:", depth.shape)
                print("Depth dtype:", depth.dtype)
                print("Depth bytes:", depth.nbytes)
                print("RGB owns memory:", rgb.flags["OWNDATA"])
                print("Depth owns memory:", depth.flags["OWNDATA"])

            else:
                assert not np.shares_memory(
                    first_rgb, rgb
                )

                assert not np.shares_memory(
                    first_depth, depth
                )

                common = (
                    valid_mask(previous_depth) & mask
                )

                if common.any():
                    mae = np.abs(
                        depth[common]
                        - previous_depth[common]
                    ).mean()

                    depth_mae.append(float(mae))
                else:
                    depth_mae.append(np.nan)

            previous_depth = depth

            if frame_id % 10 == 0:
                print(
                    f"Frame {frame_id:3d}/{NUM_FRAMES}, "
                    f"render={(t1-t0)*1000:.3f} ms, "
                    f"valid={mask.mean()*100:.2f}%, "
                    f"median={depth_medians[-1]:.4f}"
                )

            deadline = start + frame_id * PERIOD
            remaining = deadline - time.perf_counter()

            if remaining < 0:
                deadline_misses += 1
            else:
                time.sleep(remaining)

        elapsed = time.perf_counter() - start

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    # Verify that later rendering and environment cleanup
    # did not modify previously returned NumPy arrays.
    np.testing.assert_array_equal(
        first_rgb, first_rgb_copy
    )

    np.testing.assert_array_equal(
        first_depth, first_depth_copy
    )

    print("\nPASS: RGB-D queue independence.")
    print("PASS: NumPy memory ownership and lifetime.")

    return {
        "rgb": first_rgb,
        "depth": first_depth,
        "render_times": np.asarray(render_times),
        "read_times": np.asarray(read_times),
        "depth_mae": np.asarray(depth_mae),
        "valid_ratios": np.asarray(valid_ratios),
        "finite_ratios": np.asarray(finite_ratios),
        "depth_medians": np.asarray(depth_medians),
        "elapsed": elapsed,
        "deadline_misses": deadline_misses,
    }


def print_statistics(data):
    depth = data["depth"]

    finite = np.isfinite(depth)
    valid = valid_mask(depth)

    print("\n========== Depth statistics ==========")

    print("Total pixels:", depth.size)
    print("Finite pixels:", np.count_nonzero(finite))
    print("NaN pixels:", np.count_nonzero(np.isnan(depth)))
    print("Inf pixels:", np.count_nonzero(np.isinf(depth)))

    print(
        "Below near clip:",
        np.count_nonzero(finite & (depth < NEAR))
    )

    print(
        "At or above far clip:",
        np.count_nonzero(finite & (depth >= FAR))
    )

    print(
        "Provisional valid ratio:",
        f"{valid.mean()*100:.4f}%"
    )

    if finite.any():
        values = depth[finite]

        print(f"Raw min: {values.min():.6f}")
        print(f"Raw max: {values.max():.6f}")
        print(f"Raw mean: {values.mean():.6f}")

    if valid.any():
        values = depth[valid]

        print(f"Valid min: {values.min():.6f}")
        print(f"Valid max: {values.max():.6f}")
        print(f"Valid mean: {values.mean():.6f}")
        print(f"Valid median: {np.median(values):.6f}")

    print(
        "Center pixel:",
        depth[HEIGHT // 2, WIDTH // 2]
    )

    mae = data["depth_mae"]
    mae = mae[np.isfinite(mae)]

    print("\n========== Temporal stability ==========")

    if mae.size:
        print(f"Mean depth MAE: {mae.mean():.8f}")
        print(f"Max depth MAE: {mae.max():.8f}")
    else:
        print("No common valid pixels for MAE.")

    print(
        "Minimum finite ratio:",
        f"{data['finite_ratios'].min()*100:.4f}%"
    )

    print(
        "Minimum valid ratio:",
        f"{data['valid_ratios'].min()*100:.4f}%"
    )

    print("\n========== Rendering performance ==========")

    render_ms = data["render_times"] * 1000
    read_ms = data["read_times"] * 1000

    print("Captured frames:", len(render_ms))
    print(f"Elapsed time: {data['elapsed']:.3f} s")
    print(f"Capture FPS: {len(render_ms)/data['elapsed']:.3f}")

    print(f"Mean render time: {render_ms.mean():.3f} ms")
    print(f"P95 render time: {np.percentile(render_ms,95):.3f} ms")
    print(f"Max render time: {render_ms.max():.3f} ms")
    print(f"Mean read/check time: {read_ms.mean():.3f} ms")
    print("Deadline misses:", data["deadline_misses"])


def save_results(data):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    plt.imsave(
        RESULT_DIR / "rgb.png",
        data["rgb"]
    )

    np.save(
        RESULT_DIR / "depth.npy",
        data["depth"]
    )

    np.savez_compressed(
        RESULT_DIR / "statistics.npz",
        **{
            key: value
            for key, value in data.items()
            if key not in ("rgb", "depth")
        },
        width=WIDTH,
        height=HEIGHT,
        fov=70.0,
        near=NEAR,
        far=FAR,
        period=PERIOD,
    )


def plot_summary(data):
    rgb = data["rgb"]
    depth = data["depth"]

    valid = valid_mask(depth)
    values = depth[valid]

    fig, axes = plt.subplots(
        3, 2,
        figsize=(14, 12),
        layout="constrained"
    )

    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("RGB image")
    ax.axis("off")

    ax = axes[0, 1]

    # Keep a fixed depth scale for physical interpretation.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    im = ax.imshow(
        np.ma.masked_where(~valid, depth),
        cmap=cmap,
        vmin=0,
        vmax=FAR
    )

    ax.set_title(
        f"Depth map | Valid: {valid.mean()*100:.2f}%"
    )
    ax.axis("off")

    fig.colorbar(
        im, ax=ax,
        label="Depth (expected m)",
        fraction=0.046,
        pad=0.04
    )

    ax = axes[1, 0]

    ax.hist(
        values,
        bins=80,
        range=(0, FAR),
        edgecolor="none"
    )

    ax.set_title("Valid depth distribution")
    ax.set_xlabel("Depth (expected m)")
    ax.set_ylabel("Pixel count")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]

    horizontal = np.where(
        valid[HEIGHT // 2],
        depth[HEIGHT // 2],
        np.nan
    )

    ax.plot(np.arange(WIDTH), horizontal)

    ax.set_title("Horizontal center profile")
    ax.set_xlabel("Pixel u")
    ax.set_ylabel("Depth (expected m)")
    ax.grid(alpha=0.25)

    ax = axes[2, 0]

    vertical = np.where(
        valid[:, WIDTH // 2],
        depth[:, WIDTH // 2],
        np.nan
    )

    ax.plot(np.arange(HEIGHT), vertical)

    ax.set_title("Vertical center profile")
    ax.set_xlabel("Pixel v")
    ax.set_ylabel("Depth (expected m)")
    ax.grid(alpha=0.25)

    ax = axes[2, 1]

    render_ms = data["render_times"] * 1000

    ax.plot(
        np.arange(1, NUM_FRAMES + 1),
        render_ms,
        label="Render time"
    )

    ax.axhline(
        render_ms.mean(),
        linestyle="--",
        label="Mean"
    )

    ax.axhline(
        PERIOD * 1000,
        linestyle=":",
        label="10 Hz period"
    )

    ax.set_title("RGB-D rendering time")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Time (ms)")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Experiment 22: RGB-D Acquisition\n"
        "640 x 360 | Vertical FOV 70 deg | "
        "100 frames | Fixed pose",
        fontsize=16
    )

    path = OUTPUT_DIR / "22_rgbd_basic.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\nSaved summary: {path}")
    print(f"Saved raw data: {RESULT_DIR}")


def main():
    data = capture()

    print_statistics(data)
    save_results(data)
    plot_summary(data)

    print("\nPASS: RGB-D acquisition completed.")


if __name__ == "__main__":
    main()
