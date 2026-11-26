import os
from pathlib import Path
from io import StringIO

import numpy as np
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1


# 1. 配置项目路径
root = Path(__file__).resolve().parents[1]
os.environ["FLIGHTMARE_PATH"] = str(root)

# 2. 配置单无人机环境
yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)

cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = "no"
cfg["env"]["seed"] = 1

stream = StringIO()
yaml.dump(cfg, stream)

env = QuadrotorEnv_v1(stream.getvalue(), False)

try:
    # 3. 获取环境维度
    n = env.getNumOfEnvs()
    obs_dim = env.getObsDim()
    act_dim = env.getActDim()
    extra_dim = len(env.getExtraInfoNames())

    # 4. 分配输入输出数组
    action = np.zeros((n, act_dim), dtype=np.float32)
    obs = np.zeros((n, obs_dim), dtype=np.float32)
    reward = np.zeros(n, dtype=np.float32)
    done = np.zeros(n, dtype=np.bool_)
    extra_info = np.zeros((n, extra_dim), dtype=np.float32)

    # 5. 重置环境
    env.setSeed(1)

    if not env.reset(obs):
        raise RuntimeError("Environment reset failed")

    # 保存初始观测量
    obs_before = obs.copy()

    # 6. 设置控制输入
    action[0, :] = 0.0

    # 7. 推进一个仿真步长
    result = env.step(
        action,
        obs,
        reward,
        done,
        extra_info
    )

    if not result:
        raise RuntimeError("Simulation step failed")

    obs_after = obs.copy()

    # 8. 输出仿真结果
    np.set_printoptions(precision=6, suppress=True)

    print("========== Simulation ==========")
    print("Simulation dt: 0.02 s")
    print("Action:", action)
    print("Step result:", result)
    print("Reward:", reward)
    print("Done:", done)

    print("\n========== State comparison ==========")

    names = [
        "x", "y", "z",
        "yaw", "pitch", "roll",
        "vx", "vy", "vz",
        "wx", "wy", "wz"
    ]

    print(f"{'State':<10}{'Before':>14}"
          f"{'After':>14}{'Change':>14}")

    for i, name in enumerate(names):
        before = obs_before[0, i]
        after = obs_after[0, i]

        print(f"{name:<10}{before:>14.6f}"
              f"{after:>14.6f}{after-before:>14.6f}")

    print("\nAll finite:", np.isfinite(obs).all())

finally:
    env.close()
    print("Environment closed.")
