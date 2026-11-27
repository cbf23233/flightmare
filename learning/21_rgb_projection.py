"""Validate camera intrinsics using pure-rotation homographies."""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flightgym import QuadrotorEnv_v1


WIDTH = 640
HEIGHT = 360
FOV = 70.0

CAMERA_OFFSET = np.array([0.0, 0.4, 0.0])
CAMERA_CENTER = np.array([0.0, 0.4, 5.0])

# OpenCV camera coordinates -> Flightmare body coordinates
R_BC = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
])

CASES = {
    "initial":        (0, 0, 0),
    "roll_positive":  (10, 0, 0),
    "roll_negative":  (-10, 0, 0),
    "pitch_positive": (0, 10, 0),
    "pitch_negative": (0, -10, 0),
    "yaw_positive":   (0, 0, 10),
    "yaw_negative":   (0, 0, -10),
}

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "21_projection_results"
)


def make_config():
    return f"""
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
  position: {CAMERA_OFFSET.tolist()}
  rotation: [0, 0, 0]
"""


def camera_intrinsics(pixel_center=True):
    f = HEIGHT / (
        2.0 * np.tan(np.deg2rad(FOV / 2.0))
    )

    offset = 1.0 if pixel_center else 0.0

    return np.array([
        [f, 0.0, (WIDTH - offset) / 2.0],
        [0.0, f, (HEIGHT - offset) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_matrix(angles):
    roll, pitch, yaw = np.deg2rad(angles)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    return Rz @ Ry @ Rx


def rotation_to_quaternion(R):
    # All experimental rotations are close to identity.
    # Quaternion order: w, x, y, z.
    w = 0.5 * np.sqrt(1.0 + np.trace(R))

    q = np.array([
        w,
        (R[2, 1] - R[1, 2]) / (4.0 * w),
        (R[0, 2] - R[2, 0]) / (4.0 * w),
        (R[1, 0] - R[0, 1]) / (4.0 * w),
    ])

    return q / np.linalg.norm(q)


def set_pose(env, angles):
    R_WB = rotation_matrix(angles)

    # Compensate the camera mounting offset.
    position = CAMERA_CENTER - R_WB @ CAMERA_OFFSET

    state = np.zeros((1, 13), dtype=np.float32)
    obs = np.zeros((1, 12), dtype=np.float32)

    state[0, :3] = position
    state[0, 3:7] = rotation_to_quaternion(R_WB)

    assert env.resetState(state, obs)

    np.testing.assert_allclose(
        obs[0, :3], position, atol=1e-6
    )

    # Observation attitude order: yaw, pitch, roll.
    np.testing.assert_allclose(
        obs[0, 3:6],
        np.deg2rad(angles[::-1]),
        atol=1e-5,
    )

    actual_center = (
        obs[0, :3] + R_WB @ CAMERA_OFFSET
    )

    center_error = np.linalg.norm(
        actual_center - CAMERA_CENTER
    )

    assert center_error < 1e-6

    return R_WB @ R_BC, center_error


def capture_images():
    env = QuadrotorEnv_v1(make_config(), False)

    connected = False
    results = {}

    try:
        set_pose(env, CASES["initial"])

        assert env.getNumOfCameras(0) == 1

        connected = env.connectUnity()
        assert connected

        for frame_id, (name, angles) in enumerate(
            CASES.items(), start=1
        ):
            R_WC, center_error = set_pose(env, angles)

            assert env.render(frame_id)

            image = env.getRGBImage(0, 0)

            assert image is not None
            assert image.shape == (HEIGHT, WIDTH, 3)
            assert image.dtype == np.uint8
            assert image.max() > image.min()

            results[name] = (image.copy(), R_WC)

            plt.imsave(
                OUTPUT_DIR / f"{name}.png",
                image,
            )

            print(
                f"{name:16s} "
                f"frame={frame_id} "
                f"center_error={center_error:.3e} m"
            )

    finally:
        if connected:
            env.disconnectUnity()

        env.close()

    return results


def match_features(image1, image2):
    gray1 = cv2.cvtColor(
        image1, cv2.COLOR_RGB2GRAY
    )

    gray2 = cv2.cvtColor(
        image2, cv2.COLOR_RGB2GRAY
    )

    sift = cv2.SIFT_create(nfeatures=3000)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("No SIFT descriptors found.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    pairs = matcher.knnMatch(des1, des2, k=2)

    good = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        raise RuntimeError(
            f"Too few feature matches: {len(good)}"
        )

    p1 = np.array([
        kp1[m.queryIdx].pt for m in good
    ], dtype=np.float64)

    p2 = np.array([
        kp2[m.trainIdx].pt for m in good
    ], dtype=np.float64)

    # Estimate a homography independently of the
    # theoretical camera intrinsics and rotation.
    H_est, mask = cv2.findHomography(
        p1, p2, cv2.RANSAC, 3.0
    )

    if H_est is None or mask is None:
        raise RuntimeError("Homography estimation failed.")

    inliers = mask.ravel().astype(bool)

    if np.count_nonzero(inliers) < 20:
        raise RuntimeError("Too few geometric inliers.")

    return (
        p1[inliers],
        p2[inliers],
        H_est,
        len(good),
    )


def predict_pixels(points, H):
    homogeneous = np.column_stack([
        points,
        np.ones(len(points)),
    ])

    projected = (H @ homogeneous.T).T

    return (
        projected[:, :2]
        / projected[:, 2:3]
    )


def evaluate_error(predicted, observed):
    errors = np.linalg.norm(
        predicted - observed, axis=1
    )

    return {
        "mean": np.mean(errors),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "median": np.median(errors),
        "p95": np.percentile(errors, 95),
        "max": np.max(errors),
    }


def theoretical_homography(K, R1, R2):
    R21 = R2.T @ R1

    return K @ R21 @ np.linalg.inv(K)


def plot_case(name, image, observed, predicted):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.imshow(image)

    ax.scatter(
        observed[:, 0],
        observed[:, 1],
        s=18,
        facecolors="none",
        edgecolors="lime",
        label="Observed",
    )

    ax.scatter(
        predicted[:, 0],
        predicted[:, 1],
        s=15,
        marker="+",
        color="red",
        label="Predicted",
    )

    ax.set_title(name)
    ax.set_xlim(-0.5, WIDTH - 0.5)
    ax.set_ylim(HEIGHT - 0.5, -0.5)
    ax.legend()
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{name}_projection.png",
        dpi=150,
    )

    plt.close(fig)



def plot_summary(results, summary):
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))

    for i, (name, (image, _)) in enumerate(results.items()):
        ax = axes.flat[i]
        ax.imshow(image)

        roll, pitch, yaw = CASES[name]

        title = (
            f"{name}\n"
            f"Roll={roll}, Pitch={pitch}, Yaw={yaw} deg"
        )

        if name != "initial":
            with np.load(
                OUTPUT_DIR / f"{name}_matches.npz"
            ) as data:
                observed = data["observed"]
                predicted = data["predicted_center"]

            ax.scatter(
                observed[:, 0], observed[:, 1],
                s=12, facecolors="none",
                edgecolors="lime", linewidths=0.8
            )

            ax.scatter(
                predicted[:, 0], predicted[:, 1],
                s=12, marker="+",
                color="red", linewidths=0.8
            )

            row = next(r for r in summary if r[0] == name)

            title += (
                f"\nInliers: {row[2]}, "
                f"RMSE: {row[3]:.3f} px"
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.5, WIDTH - 0.5)
        ax.set_ylim(HEIGHT - 0.5, -0.5)
        ax.axis("off")

    gs = axes[2, 0].get_gridspec()

    axes[2, 1].remove()
    axes[2, 2].remove()

    ax = fig.add_subplot(gs[2, 1:])
    x = np.arange(len(summary))

    for offset, index, label in [
        (-0.25, 3, "Pixel center"),
        (0.00, 4, "Image edge"),
        (0.25, 5, "Estimated H"),
    ]:
        bars = ax.bar(
            x + offset,
            [row[index] for row in summary],
            width=0.25,
            label=label
        )

        ax.bar_label(
            bars, fmt="%.2f",
            fontsize=8, padding=2
        )

    ax.set_xticks(x)
    ax.set_xticklabels([
        "Roll +", "Roll -",
        "Pitch +", "Pitch -",
        "Yaw +", "Yaw -",
    ])

    ax.set_ylabel("RMSE [px]")
    ax.set_title("Projection error comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.margins(y=0.2)

    fig.suptitle(
        "Flightmare RGB Camera Projection Validation",
        fontsize=20, fontweight="bold"
    )

    fig.text(
        0.5, 0.02,
        "Green circles: observed features | "
        "Red crosses: predicted features",
        ha="center", fontsize=12
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    output = (
        Path(__file__).resolve().parent
        / "21_rgb_projection.png"
    )

    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Summary figure: {output}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2.setRNGSeed(1)

    K_center = camera_intrinsics(True)
    K_edge = camera_intrinsics(False)

    print("K (pixel center):")
    print(K_center)

    print("\nCapturing images...")

    results = capture_images()

    image1, R1 = results["initial"]

    print("\nProjection error [px]:")

    summary = []

    for name, (image2, R2) in results.items():
        if name == "initial":
            continue

        p1, p2, H_est, num_matches = match_features(
            image1, image2
        )

        H_center = theoretical_homography(
            K_center, R1, R2
        )

        H_edge = theoretical_homography(
            K_edge, R1, R2
        )

        pred_center = predict_pixels(p1, H_center)
        pred_edge = predict_pixels(p1, H_edge)
        pred_est = predict_pixels(p1, H_est)

        err_center = evaluate_error(pred_center, p2)
        err_edge = evaluate_error(pred_edge, p2)
        err_est = evaluate_error(pred_est, p2)

        print(f"\n{name}")
        print(
            f"  Matches: {num_matches}, "
            f"inliers: {len(p1)}, "
            f"ratio: {len(p1) / num_matches:.3f}"
        )

        for label, stats in [
            ("Pixel center", err_center),
            ("Image edge", err_edge),
            ("Estimated H", err_est),
        ]:
            print(
                f"  {label:13s} "
                f"mean={stats['mean']:.4f} "
                f"RMSE={stats['rmse']:.4f} "
                f"median={stats['median']:.4f} "
                f"P95={stats['p95']:.4f} "
                f"max={stats['max']:.4f}"
            )

        summary.append([
            name,
            num_matches,
            len(p1),
            err_center["rmse"],
            err_edge["rmse"],
            err_est["rmse"],
        ])

        np.savez(
            OUTPUT_DIR / f"{name}_matches.npz",
            reference=p1,
            observed=p2,
            predicted_center=pred_center,
            predicted_edge=pred_edge,
            H_center=H_center,
            H_edge=H_edge,
            H_estimated=H_est,
        )



    plot_summary(results, summary)

    print("DONE: Projection evaluation completed.")


if __name__ == "__main__":
    main()
