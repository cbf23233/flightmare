"""Reusable cascaded quadrotor controller.

Extracted from learning/11_position_control.py.
No simulation or rendering is performed by this module.
"""

from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


# Physical and controller parameters

root = Path(__file__).resolve().parents[1]

yaml = YAML()

with open(root / "flightlib/configs/quadrotor_env.yaml") as f:
    quad_cfg = yaml.load(f)



dyn = quad_cfg["quadrotor_dynamics"]

m = float(dyn["mass"])

l = float(dyn["arm_l"])

kappa = float(dyn["kappa"])

g = 9.81

dt = float(quad_cfg["quadrotor_env"]["sim_dt"])

d = l / np.sqrt(2)

J = m * l**2 / 12 * np.diag([4.5, 4.5, 7.0])

B = np.array([
    [1,      1,      1,      1],
    [d,     -d,     -d,      d],
    [-d,    -d,      d,      d],
    [kappa, -kappa, kappa, -kappa]
], dtype=np.float64)

thrust_mean = m * g / 4

thrust_std = m * g / 2

k_pos = np.array([0.4, 0.4, 1.0])

k_vel = np.array([1.6, 1.6, 4.0])

k_att = 2.0

k_rate = 8.0

max_vel_xy = 2.0

max_vel_z = 1.0

max_acc_xy = 3.0

max_acc_z = 3.0

max_tilt = np.deg2rad(20.0)

max_rate = np.deg2rad([90.0, 90.0, 45.0])

# Controller functions

def limit_norm(vector, maximum):

    norm = np.linalg.norm(vector)

    if norm > maximum:
        return vector * maximum / norm

    return vector.copy()

def position_controller(target, position, vel_ff=None):

    if vel_ff is None:
        vel_ff = np.zeros(3)

    vel_des = vel_ff + k_pos * (target - position)

    vel_des[:2] = limit_norm(
        vel_des[:2], max_vel_xy
    )

    vel_des[2] = np.clip(
        vel_des[2],
        -max_vel_z,
        max_vel_z
    )

    return vel_des

def velocity_controller(vel_des, velocity, acc_ff=None):

    if acc_ff is None:
        acc_ff = np.zeros(3)

    acc_des = acc_ff + k_vel * (
        vel_des - velocity
    )

    acc_des[:2] = limit_norm(
        acc_des[:2], max_acc_xy
    )

    acc_des[2] = np.clip(
        acc_des[2],
        -max_acc_z,
        max_acc_z
    )

    return acc_des

def acceleration_mapping(acc_des, yaw_des, angles):

    acc = acc_des.copy()

    # Limit desired total tilt angle
    vertical = g + acc[2]

    if vertical <= 0:
        raise RuntimeError(
            "Invalid vertical thrust demand"
        )

    horizontal_limit = (
        vertical * np.tan(max_tilt)
    )

    acc[:2] = limit_norm(
        acc[:2], horizontal_limit
    )

    # Desired thrust direction
    force_direction = (
        acc + np.array([0.0, 0.0, g])
    )

    b3 = (
        force_direction
        / np.linalg.norm(force_direction)
    )

    bx, by, bz = b3

    roll_des = np.arcsin(
        np.clip(
            bx * np.sin(yaw_des)
            - by * np.cos(yaw_des),
            -1.0, 1.0
        )
    )

    pitch_des = np.arctan2(
        bx * np.cos(yaw_des)
        + by * np.sin(yaw_des),
        bz
    )

    att_des = np.array([
        roll_des,
        pitch_des,
        yaw_des
    ])

    # Actual-attitude tilt compensation
    roll, pitch, yaw = angles

    tilt_factor = (
        np.cos(roll) * np.cos(pitch)
    )

    if tilt_factor < 0.5:
        raise RuntimeError(
            "Actual tilt angle too large"
        )

    total_thrust = (
        m * (g + acc[2])
        / tilt_factor
    )

    return att_des, total_thrust, acc

def attitude_controller(att_des, angles, omega):

    error = att_des - angles

    error = (
        error + np.pi
    ) % (2 * np.pi) - np.pi

    omega_des = k_att * error

    omega_des = np.clip(
        omega_des,
        -max_rate,
        max_rate
    )

    alpha_des = k_rate * (
        omega_des - omega
    )

    torque = J @ alpha_des

    return torque, omega_des

def allocate(total_thrust, torque):

    wrench = np.concatenate(
        ([total_thrust], torque)
    )

    rotor_thrusts = np.linalg.solve(
        B, wrench
    )

    u_raw = (
        rotor_thrusts - thrust_mean
    ) / thrust_std

    u = np.clip(
        u_raw, -0.5, 1.0
    )

    saturated = np.any(
        np.abs(u_raw - u) > 1e-10
    )

    return u, saturated


def compute_control(
    observation,
    target_position,
    target_yaw,
    velocity_ff=None,
    acceleration_ff=None,
):
    """Compute normalized rotor thrust commands.

    Observation order:
        x, y, z, yaw, pitch, roll,
        vx, vy, vz, wx, wy, wz

    target_position: world-frame position in meters.
    target_yaw: desired yaw angle in radians.
    velocity_ff: world-frame velocity feedforward in m/s.
    acceleration_ff: world-frame acceleration feedforward in m/s^2.
    Both feedforward inputs default to zero.
    """

    observation = np.asarray(
        observation, dtype=np.float64
    )

    target_position = np.asarray(
        target_position, dtype=np.float64
    )

    target_yaw = float(target_yaw)

    if observation.shape != (12,):
        raise ValueError("Expected 12-dimensional observation")

    if target_position.shape != (3,):
        raise ValueError("Expected 3-dimensional target position")

    if (
        not np.isfinite(observation).all()
        or not np.isfinite(target_position).all()
        or not np.isfinite(target_yaw)
    ):
        raise ValueError("Non-finite controller input")

    velocity_ff = (
        np.zeros(3)
        if velocity_ff is None
        else np.asarray(velocity_ff, dtype=np.float64)
    )

    acceleration_ff = (
        np.zeros(3)
        if acceleration_ff is None
        else np.asarray(acceleration_ff, dtype=np.float64)
    )

    if velocity_ff.shape != (3,):
        raise ValueError("Expected 3-dimensional velocity feedforward")

    if acceleration_ff.shape != (3,):
        raise ValueError("Expected 3-dimensional acceleration feedforward")

    if (
        not np.isfinite(velocity_ff).all()
        or not np.isfinite(acceleration_ff).all()
    ):
        raise ValueError("Non-finite feedforward input")

    position = observation[:3]
    velocity = observation[6:9]
    angles = observation[[5, 4, 3]]
    omega = observation[9:12]

    vel_des = position_controller(
        target_position, position, velocity_ff
    )

    acc_des = velocity_controller(
        vel_des, velocity, acceleration_ff
    )

    att_des, thrust, acc_limited = acceleration_mapping(
        acc_des, target_yaw, angles
    )

    torque, omega_des = attitude_controller(
        att_des, angles, omega
    )

    action, saturated = allocate(thrust, torque)

    info = {
        "vel_des": vel_des,
        "acc_des": acc_limited,
        "att_des": att_des,
        "omega_des": omega_des,
        "thrust": thrust,
        "torque": torque,
        "saturated": bool(saturated),
    }

    return action, info
