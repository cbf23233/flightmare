import numpy as np

np.set_printoptions(precision=8, suppress=True)

g = 9.81
m = 0.73


# 1. Vector normalization
def normalize(v):
    norm = np.linalg.norm(v)

    if norm < 1e-8:
        raise ValueError("Near-zero vector")

    return v / norm


# 2. ZYX Euler angles to rotation matrix
def euler_to_rotation(roll, pitch, yaw):

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


# 3. Rotation matrix to Euler angles
def rotation_to_euler(R):

    roll = np.arctan2(
        R[2, 1], R[2, 2]
    )

    pitch = np.arcsin(
        np.clip(-R[2, 0], -1, 1)
    )

    yaw = np.arctan2(
        R[1, 0], R[0, 0]
    )

    return np.array([roll, pitch, yaw])


# 4. Cross-product method
def mapping_cross_product(acc, yaw):

    force = m * (
        acc + np.array([0, 0, g])
    )

    thrust = np.linalg.norm(force)

    b3 = normalize(force)

    b1_ref = np.array([
        np.cos(yaw),
        np.sin(yaw),
        0.0
    ])

    b2 = normalize(
        np.cross(b3, b1_ref)
    )

    b1 = np.cross(b2, b3)

    R = np.column_stack(
        (b1, b2, b3)
    )

    return thrust, R


# 5. Direct Euler-angle method
def mapping_euler(acc, yaw):

    force = m * (
        acc + np.array([0, 0, g])
    )

    thrust = np.linalg.norm(force)

    bx, by, bz = normalize(force)

    roll = np.arcsin(
        np.clip(
            bx * np.sin(yaw)
            - by * np.cos(yaw),
            -1, 1
        )
    )

    pitch = np.arctan2(
        bx * np.cos(yaw)
        + by * np.sin(yaw),
        bz
    )

    R = euler_to_rotation(
        roll, pitch, yaw
    )

    return thrust, R


# 6. Recover acceleration from thrust and attitude
def recover_acceleration(thrust, R):

    return (
        thrust / m * R[:, 2]
        - np.array([0, 0, g])
    )


# 7. Test cases
cases = [
    ("Hover", [0, 0, 0], 0),
    ("Positive X", [2, 0, 0], 0),
    ("Positive Y", [0, 2, 0], 0),
    ("Combined XY", [2, 1, 0], 30),
    ("Yaw 90", [2, 0, 0], 90),
    ("Combined XYZ", [2, -1, 1], 45),
    ("Negative yaw", [-2, 1, -1], -60)
]

methods = [
    ("Cross product", mapping_cross_product),
    ("Direct Euler", mapping_euler)
]

for name, acc, yaw_deg in cases:

    acc = np.asarray(acc, dtype=np.float64)
    yaw = np.deg2rad(yaw_deg)

    print(f"\n========== {name} ==========")
    print("Desired acceleration:", acc)
    print("Desired yaw:", yaw_deg, "deg")

    for method_name, method in methods:

        thrust, R = method(acc, yaw)

        angles = rotation_to_euler(R)

        acc_actual = recover_acceleration(
            thrust, R
        )

        orthogonal_error = np.max(
            np.abs(R.T @ R - np.eye(3))
        )

        determinant = np.linalg.det(R)

        acc_error = np.linalg.norm(
            acc_actual - acc
        )

        yaw_error = (
            angles[2] - yaw + np.pi
        ) % (2 * np.pi) - np.pi

        print(f"\n--- {method_name} ---")

        print("Thrust:", thrust, "N")

        print(
            "Euler [R, P, Y]:",
            np.rad2deg(angles), "deg"
        )

        print("Rotation matrix:\n", R)

        print(
            "Recovered acceleration:",
            acc_actual
        )

        print(
            "Acceleration error:",
            f"{acc_error:.3e}"
        )

        print(
            "Orthogonality error:",
            f"{orthogonal_error:.3e}"
        )

        print(
            "Determinant:",
            f"{determinant:.10f}"
        )

        print(
            "Yaw error:",
            f"{np.rad2deg(yaw_error):.8f} deg"
        )

        assert acc_error < 1e-10
        assert orthogonal_error < 1e-10
        assert abs(determinant - 1) < 1e-10

        if method_name == "Direct Euler":
            assert abs(yaw_error) < 1e-10

print("\nAll attitude mapping tests PASSED.")
