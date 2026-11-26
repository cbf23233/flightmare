import os
from pathlib import Path
from io import StringIO
from time import perf_counter
import csv

os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1


# 1. Environment configuration
root = Path(__file__).resolve().parents[1]
os.environ["FLIGHTMARE_PATH"] = str(root)

yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    base_cfg = yaml.load(f)

with open(root / "flightlib/configs/quadrotor_env.yaml") as f:
    quad_cfg = yaml.load(f)

dt = float(quad_cfg["quadrotor_env"]["sim_dt"])

env_counts = [1, 10, 50, 100, 500, 1000, 2000, 5000]
thread_counts = [1, 2, 4, 8, 12]

warmup_steps = 100
pilot_steps = 100
repeats = 5
min_test_time = 0.5
seed = 42

results = []


# 2. Create vectorized environment
def create_env(n, threads):
    base_cfg["env"]["num_envs"] = n
    base_cfg["env"]["num_threads"] = threads
    base_cfg["env"]["render"] = "no"
    base_cfg["env"]["seed"] = seed

    stream = StringIO()
    yaml.dump(base_cfg, stream)

    env = QuadrotorEnv_v1(stream.getvalue(), False)
    env.setSeed(seed)

    return env


# 3. Benchmark one configuration
def benchmark(n, threads):
    env = create_env(n, threads)

    state = np.zeros((n, 13), dtype=np.float32)
    state[:, 2] = 5.0
    state[:, 3] = 1.0

    obs = np.zeros(
        (n, env.getObsDim()), dtype=np.float32
    )

    action = np.zeros(
        (n, env.getActDim()), dtype=np.float32
    )

    reward = np.zeros(n, dtype=np.float32)
    done = np.zeros(n, dtype=np.bool_)

    extra_info = np.zeros(
        (n, len(env.getExtraInfoNames())),
        dtype=np.float32
    )

    step = env.step

    def reset():
        if not env.resetState(state, obs):
            raise RuntimeError("Reset failed")

        done.fill(False)

    def run_steps(steps):
        start = perf_counter()

        for _ in range(steps):
            if not step(
                action, obs, reward, done, extra_info
            ):
                raise RuntimeError("Step failed")

        return perf_counter() - start

    try:
        # Warm-up
        reset()
        run_steps(warmup_steps)

        # Calibrate benchmark length
        reset()
        pilot_time = run_steps(pilot_steps)

        steps = max(
            100,
            int(np.ceil(
                pilot_steps * min_test_time
                / max(pilot_time, 1e-9)
            ))
        )

        # Limit simulation duration to avoid
        # excessive long-term hover drift.
        steps = min(steps, 2500)

        elapsed_times = []
        throughputs = []
        total_steps = []

        # Repeated measurements
        for _ in range(repeats):
            elapsed = 0.0
            measured_steps = 0

            while elapsed < min_test_time:
                reset()

                elapsed += run_steps(steps)
                measured_steps += steps

                if not np.isfinite(obs).all():
                    raise RuntimeError(
                        "Non-finite observation"
                    )

                if done.any():
                    raise RuntimeError(
                        "Unexpected terminal state"
                    )

            elapsed_times.append(elapsed)
            total_steps.append(measured_steps)

            throughputs.append(
                n * measured_steps / elapsed
            )

        q = np.asarray(throughputs)

        median_q = float(np.median(q))
        mean_q = float(np.mean(q))
        std_q = float(np.std(q))

        return {
            "num_envs": n,
            "num_threads": threads,
            "steps_per_block": steps,
            "median_measured_steps": float(
                np.median(total_steps)
            ),
            "median_elapsed_s": float(
                np.median(elapsed_times)
            ),
            "throughput": median_q,
            "throughput_mean": mean_q,
            "throughput_std": std_q,
            "cv_percent": 100 * std_q / mean_q,
            "rtf": median_q * dt / n,
        }

    finally:
        env.close()


# 4. Run benchmark matrix
print("========== Benchmark configuration ==========")
print("Environment counts:", env_counts)
print("Thread counts:", thread_counts)
print("Simulation dt:", dt)
print("Warm-up steps:", warmup_steps)
print("Repeats:", repeats)
print("Minimum measured time:", min_test_time, "s")
print("Total configurations:",
      len(env_counts) * len(thread_counts))

for n in env_counts:
    print(f"\n========== {n} environments ==========")

    baseline = None

    for threads in thread_counts:
        result = benchmark(n, threads)

        if threads == 1:
            baseline = result["throughput"]

        speedup = result["throughput"] / baseline

        result["speedup"] = speedup
        result["efficiency_percent"] = (
            100 * speedup / threads
        )

        results.append(result)

        print(
            f"Threads: {threads:2d} | "
            f"Throughput: "
            f"{result['throughput']:12,.0f} /s | "
            f"Speedup: {speedup:6.3f}x | "
            f"RTF: {result['rtf']:9.2f}x | "
            f"CV: {result['cv_percent']:5.2f}%",
            flush=True
        )


# 5. Save numerical results
csv_path = root / "learning/15_parallel_benchmark.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys()), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(results)


# 6. Print best configurations
print("\n========== Best thread count ==========")

for n in env_counts:
    group = [
        r for r in results
        if r["num_envs"] == n
    ]

    best = max(
        group, key=lambda r: r["throughput"]
    )

    print(
        f"Envs: {n:5d} | "
        f"Best threads: {best['num_threads']:2d} | "
        f"Throughput: {best['throughput']:12,.0f} /s | "
        f"RTF: {best['rtf']:9.2f}x"
    )

best = max(
    results, key=lambda r: r["throughput"]
)

print("\n========== Maximum throughput ==========")
for key, value in best.items():
    print(f"{key}: {value}")


# 7. Plot benchmark results
fig, axes = plt.subplots(
    2, 2, figsize=(13, 9)
)

for threads in thread_counts:
    group = [
        r for r in results
        if r["num_threads"] == threads
    ]

    n = [r["num_envs"] for r in group]
    label = f"{threads} thread(s)"

    axes[0, 0].plot(
        n,
        [r["throughput"] / 1e6 for r in group],
        "o-", label=label
    )

    axes[0, 1].plot(
        n,
        [r["speedup"] for r in group],
        "o-", label=label
    )

    axes[1, 0].plot(
        n,
        [r["efficiency_percent"] for r in group],
        "o-", label=label
    )

    axes[1, 1].plot(
        n,
        [r["rtf"] for r in group],
        "o-", label=label
    )

axes[0, 0].set_title("Simulation Throughput")
axes[0, 0].set_ylabel("Million transitions/s")

axes[0, 1].set_title("Parallel Speedup")
axes[0, 1].set_ylabel("Speedup")

axes[1, 0].set_title("Parallel Efficiency")
axes[1, 0].set_ylabel("Efficiency (%)")

axes[1, 1].set_title("Real-Time Factor")
axes[1, 1].set_ylabel("RTF")
axes[1, 1].set_yscale("log")

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_xlabel("Number of environments")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

fig.suptitle("Flightmare Parallel Simulation Benchmark")
fig.tight_layout()

figure_path = (
    root / "learning/15_parallel_benchmark.png"
)

fig.savefig(figure_path, dpi=180)
plt.close(fig)


# 8. Throughput heatmap
matrix = np.zeros(
    (len(env_counts), len(thread_counts))
)

for r in results:
    i = env_counts.index(r["num_envs"])
    j = thread_counts.index(r["num_threads"])
    matrix[i, j] = r["throughput"] / 1e6

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(matrix, aspect="auto", cmap="viridis")

ax.set_xticks(range(len(thread_counts)))
ax.set_xticklabels(thread_counts)

ax.set_yticks(range(len(env_counts)))
ax.set_yticklabels(env_counts)

ax.set_xlabel("Number of threads")
ax.set_ylabel("Number of environments")
ax.set_title("Throughput (Million transitions/s)")

for i in range(len(env_counts)):
    for j in range(len(thread_counts)):
        value = matrix[i, j]

        color = (
            "white"
            if value < matrix.max() * 0.55
            else "black"
        )

        ax.text(
            j, i, f"{value:.3f}",
            ha="center", va="center",
            color=color
        )

fig.colorbar(im, ax=ax)
fig.tight_layout()

heatmap_path = (
    root / "learning/15_parallel_heatmap.png"
)

fig.savefig(heatmap_path, dpi=180)
plt.close(fig)

print("\nCSV saved:", csv_path)
print("Figure saved:", figure_path)
print("Figure saved:", heatmap_path)
print("\nBenchmark completed.")
