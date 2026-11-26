import os
from pathlib import Path
from io import StringIO

import numpy as np
from ruamel.yaml import YAML
from flightgym import QuadrotorEnv_v1


# 1. 获取 Flightmare 根目录
root = Path(__file__).resolve().parents[1]
os.environ["FLIGHTMARE_PATH"] = str(root)

# 2. 读取官方环境配置
yaml = YAML()

with open(root / "flightlib/configs/vec_env.yaml") as f:
    cfg = yaml.load(f)

# 3. 修改为单无人机仿真
cfg["env"]["num_envs"] = 1
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = "no"
cfg["env"]["seed"] = 1

# 4. 将配置转换为 YAML 字符串
stream = StringIO()
yaml.dump(cfg, stream)

# 5. 创建仿真环境
env = QuadrotorEnv_v1(stream.getvalue(), False)

try:
    # 6. 获取环境基本信息
    num_envs = env.getNumOfEnvs()
    obs_dim = env.getObsDim()
    act_dim = env.getActDim()
    extra_names = env.getExtraInfoNames()

    print("========== Environment ==========")
    print("Number of environments:", num_envs)
    print("Observation dimension:", obs_dim)
    print("Action dimension:", act_dim)
    print("Extra info names:", extra_names)

    # 7. 创建观测量数组
    obs = np.zeros(
        (num_envs, obs_dim),
        dtype=np.float32
    )

    # 8. 设置随机种子并重置环境
    env.setSeed(1)
    result = env.reset(obs)

    print("\n========== Reset ==========")
    print("Reset result:", result)
    print("Observation shape:", obs.shape)
    print("Observation:", obs)
    print("All finite:", np.isfinite(obs).all())

    # 9. 分别输出各类状态
    print("\n========== Initial state ==========")
    print("Position [m]:", obs[0, 0:3])
    print("Yaw/Pitch/Roll [rad]:", obs[0, 3:6])
    print("Yaw/Pitch/Roll [deg]:",
          np.rad2deg(obs[0, 3:6]))
    print("Linear velocity [m/s]:", obs[0, 6:9])
    print("Angular velocity [rad/s]:", obs[0, 9:12])

finally:
    env.close()
    print("\nEnvironment closed.")
