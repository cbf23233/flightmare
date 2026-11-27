#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/common/logger.hpp"

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>

using namespace flightlib;

static constexpr int OBS_DIM = 12;
static constexpr int ACT_DIM = 4;
static constexpr int SIM_STEPS_N = 20;
static constexpr Scalar TOL = 1e-3;

TEST(QuadrotorEnv, Constructor) {
  Logger logger("QuadrotorEnv");

  // create env and load configuration from a yaml file.
  std::string config_path =
    getenv("FLIGHTMARE_PATH") +
    std::string("/flightlib/configs/quadrotor_env.yaml");
  logger.info("Environment configuration path \"%s\".", config_path.c_str());
  QuadrotorEnv env0(config_path);

  Vector<3> debug{3.0, 3.0, 3.0};

  // check observation and action dimensions
  int obs_dim = env0.getObsDim();
  int act_dim = env0.getActDim();
  EXPECT_EQ(obs_dim, OBS_DIM);
  EXPECT_EQ(act_dim, ACT_DIM);

  // load parameters via the yaml file
  YAML::Node cfg = YAML::LoadFile(config_path);
  QuadrotorEnv env1;
  EXPECT_TRUE(env1.loadParam(cfg));

  // evaluate parameters
  Scalar expect_sim_dt = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();

  const int obs_dim1 = env1.getObsDim();
  const int act_dim1 = env1.getActDim();
  const Scalar sim_dt = env1.getSimTimeStep();
  Matrix<OBS_DIM, OBS_DIM> Q;
  Matrix<ACT_DIM, ACT_DIM> Q_act;

  EXPECT_EQ(obs_dim1, OBS_DIM);
  EXPECT_EQ(act_dim1, ACT_DIM);
  EXPECT_EQ(expect_sim_dt, sim_dt);

  //
  std::cout << env1 << std::endl;
}

TEST(QuadrotorEnv, ResetEnv) {
  QuadrotorEnv env;

  // randomly reset the environment
  Vector<OBS_DIM> obs;
  Vector<OBS_DIM> exp_obs;

  EXPECT_TRUE(env.reset(obs));
  env.getObs(exp_obs);
  EXPECT_TRUE(exp_obs.isApprox(obs));

  // reset the environment to the zero state
  EXPECT_TRUE(env.reset(obs, false));
  EXPECT_EQ(obs(0), 0.0);
  EXPECT_EQ(obs(1), 0.0);
  EXPECT_EQ(obs(2), 0.0);

  env.reset(obs);
}

TEST(QuadrotorEnv, StepEnv) {
  QuadrotorEnv env;

  Vector<ACT_DIM> act{0, 0, 0, 0};
  Vector<OBS_DIM> obs;
  Vector<OBS_DIM> next_obs;

  env.reset(obs, false);
  Scalar reward = env.step(act, next_obs);

  // check control command
  Command cmd;
  EXPECT_TRUE(env.getAct(&cmd));
  if (cmd.isRatesThrust()) {
    EXPECT_EQ(cmd.collective_thrust, -Gz);
    EXPECT_TRUE(cmd.omega.allFinite());
  } else if (cmd.isSingleRotorThrusts()) {
    // TODO: Not tested yet.
    EXPECT_TRUE(cmd.thrusts.allFinite());
  }

  Vector<ACT_DIM> executed_act;
  EXPECT_TRUE(env.getAct(executed_act));

  for (int i = 0; i < SIM_STEPS_N; i++) {
    reward = env.step(act, next_obs);
  }

  std::cout << reward << std::endl;
  // in case this failed, decrease motor_tau in the Quadrotor class.
  // EXPECT_TRUE(((next_obs - obs).norm() < 1.0));
}

TEST(QuadrotorEnv, CameraDisabled) {
  QuadrotorEnv env;

  EXPECT_EQ(env.getNumOfCameras(), 0u);

  const YAML::Node cfg = YAML::Load("enabled: false");

  EXPECT_TRUE(env.configureCamera(cfg));
  EXPECT_EQ(env.getNumOfCameras(), 0u);

  cv::Mat image;
  EXPECT_FALSE(env.getRGBImage(0, image));
}

TEST(QuadrotorEnv, CameraDefaultConfig) {
  QuadrotorEnv env;

  const YAML::Node cfg = YAML::Load("enabled: true");

  ASSERT_TRUE(env.configureCamera(cfg));
  EXPECT_EQ(env.getNumOfCameras(), 1u);

  cv::Mat image;
  EXPECT_FALSE(env.getRGBImage(0, image));
  EXPECT_FALSE(env.getRGBImage(1, image));
}

TEST(QuadrotorEnv, CameraCustomConfig) {
  QuadrotorEnv env;

  const YAML::Node cfg = YAML::Load(R"(
enabled: true
width: 640
height: 360
fov: 90.0
position: [0.1, -0.2, 0.3]
rotation: [10.0, -20.0, 30.0]
)");

  ASSERT_TRUE(env.configureCamera(cfg));
  EXPECT_EQ(env.getNumOfCameras(), 1u);
}

TEST(QuadrotorEnv, CameraDuplicateConfig) {
  QuadrotorEnv env;

  const YAML::Node cfg = YAML::Load("enabled: true");

  ASSERT_TRUE(env.configureCamera(cfg));
  EXPECT_FALSE(env.configureCamera(cfg));
  EXPECT_EQ(env.getNumOfCameras(), 1u);
}

TEST(QuadrotorEnv, CameraInvalidDimensions) {
  QuadrotorEnv env;

  for (const auto& config : {
         "{enabled: true, width: 0}",
         "{enabled: true, width: -640}",
         "{enabled: true, height: 0}",
         "{enabled: true, height: -480}"}) {
    EXPECT_FALSE(env.configureCamera(YAML::Load(config)));
    EXPECT_EQ(env.getNumOfCameras(), 0u);
  }
}

TEST(QuadrotorEnv, CameraInvalidFOV) {
  QuadrotorEnv env;

  for (const auto& config : {
         "{enabled: true, fov: 0}",
         "{enabled: true, fov: -10}",
         "{enabled: true, fov: 180}",
         "{enabled: true, fov: 200}",
         "{enabled: true, fov: .nan}",
         "{enabled: true, fov: .inf}"}) {
    EXPECT_FALSE(env.configureCamera(YAML::Load(config)));
    EXPECT_EQ(env.getNumOfCameras(), 0u);
  }
}

TEST(QuadrotorEnv, CameraInvalidExtrinsics) {
  QuadrotorEnv env;

  for (const auto& config : {
         "{enabled: true, position: [0, 0]}",
         "{enabled: true, position: [0, 0, 0, 0]}",
         "{enabled: true, position: [.nan, 0, 0]}",
         "{enabled: true, position: [0, .inf, 0]}",
         "{enabled: true, rotation: [0, 0]}",
         "{enabled: true, rotation: [0, 0, 0, 0]}",
         "{enabled: true, rotation: [.nan, 0, 0]}",
         "{enabled: true, rotation: [0, .inf, 0]}"}) {
    EXPECT_FALSE(env.configureCamera(YAML::Load(config)));
    EXPECT_EQ(env.getNumOfCameras(), 0u);
  }

  // Invalid configurations must not prevent a later valid setup.
  ASSERT_TRUE(
    env.configureCamera(YAML::Load("enabled: true")));
  EXPECT_EQ(env.getNumOfCameras(), 1u);
}
