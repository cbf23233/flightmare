import numpy as np

dt = 0.02
duration = 10.0
wn = 2.0

cases = [
    ("Underdamped", 0.3, 38.3611, 6.74),
    ("Critically damped", 1.0, 0.0, 2.92),
    ("Overdamped", 1.5, 0.0, 5.34)
]

t = np.arange(round(duration / dt) + 1) * dt


def metrics(z):
    overshoot = max(0.0, (np.max(z) - 6.0) * 100)

    outside = np.flatnonzero(np.abs(z - 6.0) > 0.02)

    if len(outside) == 0:
        settling = 0.0
    elif outside[-1] == len(t) - 1:
        settling = np.nan
    else:
        settling = t[outside[-1] + 1]

    return overshoot, settling


for name, zeta, sim_mp, sim_ts in cases:

    kp = wn ** 2
    kd = 2 * zeta * wn

    # Continuous-time analytical solution
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta ** 2)

        error = np.exp(-zeta * wn * t) * (
            np.cos(wd * t)
            + zeta * wn / wd * np.sin(wd * t)
        )

    elif zeta == 1:
        error = np.exp(-wn * t) * (1 + wn * t)

    else:
        r1 = -wn * (zeta - np.sqrt(zeta ** 2 - 1))
        r2 = -wn * (zeta + np.sqrt(zeta ** 2 - 1))

        error = (
            r2 * np.exp(r1 * t)
            - r1 * np.exp(r2 * t)
        ) / (r2 - r1)

    z_cont = 6.0 - error

    # Discrete-time closed-loop model
    A = np.array([
        [1 - 0.5 * kp * dt**2,
         dt - 0.5 * kd * dt**2],
        [-kp * dt,
         1 - kd * dt]
    ])

    x = np.array([-1.0, 0.0])

    z_disc = [5.0]

    for _ in range(len(t) - 1):
        x = A @ x
        z_disc.append(6.0 + x[0])

    z_disc = np.asarray(z_disc)

    cont_mp, cont_ts = metrics(z_cont)
    disc_mp, disc_ts = metrics(z_disc)

    print(f"\n========== {name} ==========")

    print("Discrete system matrix:")
    print(A)

    print("Discrete poles:")
    print(np.linalg.eigvals(A))

    print(
        f"{'Model':<18}"
        f"{'Overshoot (%)':>16}"
        f"{'Settling (s)':>16}"
    )

    for model, mp, ts in [
        ("Continuous", cont_mp, cont_ts),
        ("Discrete", disc_mp, disc_ts),
        ("Flightmare", sim_mp, sim_ts)
    ]:
        print(f"{model:<18}{mp:>16.6f}{ts:>16.4f}")

    print(
        "Discrete vs Flightmare overshoot difference:",
        f"{abs(disc_mp - sim_mp):.6f} percentage points"
    )

    print(
        "Discrete vs Flightmare settling difference:",
        f"{abs(disc_ts - sim_ts):.6f} s"
    )
