"""Validate camera projection across resolutions and FOVs."""

import argparse
import importlib.util
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "21_intrinsics_results"

CASES = {
    "baseline":       (640, 360, 70.0),
    "low_resolution": (320, 180, 70.0),
    "aspect_ratio":   (640, 480, 70.0),
    "narrow_fov":     (640, 360, 50.0),
    "wide_fov":       (640, 360, 90.0),
}

LABELS = {
    "baseline": "Baseline",
    "low_resolution": "Low resolution",
    "aspect_ratio": "Aspect ratio",
    "narrow_fov": "Narrow FOV",
    "wide_fov": "Wide FOV",
}

# Reuse the validated projection experiment.
spec = importlib.util.spec_from_file_location(
    "projection",
    ROOT / "21_rgb_projection.py"
)

projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def camera_intrinsics(width, height, fov):
    f = height / (
        2.0 * np.tan(np.deg2rad(fov / 2.0))
    )

    return np.array([
        [f, 0.0, (width - 1) / 2.0],
        [0.0, f, (height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def make_config(width, height, fov):
    return f"""
env:
  seed: 1
  scene_id: 2
  num_envs: 1
  num_threads: 1
  render: true

camera:
  enabled: true
  width: {width}
  height: {height}
  fov: {fov}
  position: [0.0, 0.4, 0.0]
  rotation: [0.0, 0.0, 0.0]
"""


def capture(name):
    width, height, fov = CASES[name]

    K = camera_intrinsics(width, height, fov)

    env = QuadrotorEnv_v1(
        make_config(width, height, fov),
        False
    )

    connected = False

    state = np.zeros((1, 13), dtype=np.float32)
    obs = np.zeros((1, 12), dtype=np.float32)

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
        assert image.shape == (height, width, 3)
        assert image.dtype == np.uint8
        assert image.max() > image.min()

        OUTPUT_DIR.mkdir(
            parents=True, exist_ok=True
        )

        plt.imsave(
            OUTPUT_DIR / f"{name}.png",
            image
        )

        np.savez(
            OUTPUT_DIR / f"{name}.npz",
            image=image,
            K=K,
            width=width,
            height=height,
            fov=fov,
        )

        print(f"\nCase: {name}")
        print(f"Image shape: {image.shape}")
        print(f"Vertical FOV: {fov:.2f} deg")
        print("K:")
        print(K)
        print("PASS: Image captured.")

    finally:
        if connected:
            env.disconnectUnity()

        env.close()


def load_case(name):
    path = OUTPUT_DIR / f"{name}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Capture {name} first."
        )

    with np.load(path) as data:
        return data["image"], data["K"]


def analyze():
    cv2.setRNGSeed(1)

    reference, K1 = load_case("baseline")

    fig, axes = plt.subplots(
        2, 3, figsize=(17, 9)
    )

    summary = []

    for i, name in enumerate(CASES):
        image, K2 = load_case(name)
        width, height, fov = CASES[name]

        ax = axes.flat[i]
        ax.imshow(image)

        title = (
            f"{LABELS[name]}\n"
            f"{width} x {height}, "
            f"FOV={fov:.0f} deg\n"
            f"f={K2[0, 0]:.3f} px"
        )

        if name != "baseline":
            p1, p2, H_est, num_matches = (
                projection.match_features(
                    reference, image
                )
            )

            H_theory = K2 @ np.linalg.inv(K1)

            predicted = projection.predict_pixels(
                p1, H_theory
            )

            predicted_est = (
                projection.predict_pixels(
                    p1, H_est
                )
            )

            err = projection.evaluate_error(
                predicted, p2
            )

            err_est = projection.evaluate_error(
                predicted_est, p2
            )

            print(f"\n{name}")
            print(
                f"  Matches: {num_matches}, "
                f"inliers: {len(p1)}, "
                f"ratio: {len(p1)/num_matches:.3f}"
            )

            for label, stats in [
                ("Theoretical H", err),
                ("Estimated H", err_est),
            ]:
                print(
                    f"  {label:14s} "
                    f"mean={stats['mean']:.4f} "
                    f"RMSE={stats['rmse']:.4f} "
                    f"median={stats['median']:.4f} "
                    f"P95={stats['p95']:.4f} "
                    f"max={stats['max']:.4f}"
                )

            summary.append([
                name,
                err["rmse"],
                err_est["rmse"],
                len(p1),
            ])

            np.savez(
                OUTPUT_DIR / f"{name}_matches.npz",
                reference=p1,
                observed=p2,
                predicted=predicted,
                H_theory=H_theory,
                H_estimated=H_est,
            )

            ax.scatter(
                p2[:, 0], p2[:, 1],
                s=14, facecolors="none",
                edgecolors="lime",
                linewidths=0.8,
            )

            ax.scatter(
                predicted[:, 0],
                predicted[:, 1],
                s=14, marker="+",
                color="red",
                linewidths=0.8,
            )

            title += (
                f"\nInliers: {len(p1)}, "
                f"RMSE: {err['rmse']:.3f} px"
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.axis("off")

    ax = axes.flat[5]

    x = np.arange(len(summary))

    for offset, index, label in [
        (-0.18, 1, "Theoretical H"),
        (0.18, 2, "Estimated H"),
    ]:
        bars = ax.bar(
            x + offset,
            [row[index] for row in summary],
            width=0.36,
            label=label,
        )

        ax.bar_label(
            bars, fmt="%.3f",
            fontsize=9, padding=2
        )

    ax.set_xticks(x)
    ax.set_xticklabels([
        "Low res.",
        "Aspect",
        "Narrow",
        "Wide",
    ])

    ax.set_ylabel("RMSE [px]")
    ax.set_title("Projection error comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.margins(y=0.2)

    fig.suptitle(
        "Flightmare Camera Intrinsics Validation",
        fontsize=20,
        fontweight="bold",
    )

    fig.text(
        0.5, 0.02,
        "Green circles: observed features | "
        "Red crosses: predicted features",
        ha="center",
        fontsize=12,
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 0.94]
    )

    output = ROOT / "21_rgb_intrinsics.png"

    fig.savefig(
        output, dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSummary figure: {output}")
    print("DONE: Intrinsics evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case",
        choices=[*CASES.keys(), "analyze"]
    )

    args = parser.parse_args()

    if args.case == "analyze":
        analyze()
    else:
        capture(args.case)
