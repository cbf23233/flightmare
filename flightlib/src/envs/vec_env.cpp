#include <stdexcept>

#include "flightlib/envs/vec_env.hpp"

namespace flightlib {

template<typename EnvBase>
VecEnv<EnvBase>::VecEnv()
  : VecEnv(getenv("FLIGHTMARE_PATH") +
           std::string("/flightlib/configs/vec_env.yaml")) {}

template<typename EnvBase>
VecEnv<EnvBase>::VecEnv(const YAML::Node& cfg_node) : cfg_(cfg_node) {
  // initialization
  init();
}

template<typename EnvBase>
VecEnv<EnvBase>::VecEnv(const std::string& cfgs, const bool from_file) {
  // load environment configuration
  if (from_file) {
    // load directly from a yaml file
    cfg_ = YAML::LoadFile(cfgs);
  } else {
    // load from a string or dictionary
    cfg_ = YAML::Load(cfgs);
  }
  // initialization
  init();
}

template<typename EnvBase>
void VecEnv<EnvBase>::init(void) {
  //
  unity_render_ = cfg_["env"]["render"].as<bool>();
  seed_ = cfg_["env"]["seed"].as<int>();
  num_envs_ = cfg_["env"]["num_envs"].as<int>();
  scene_id_ = cfg_["env"]["scene_id"].as<SceneID>();

  // set threads
  omp_set_num_threads(cfg_["env"]["num_threads"].as<int>());

  // create & setup environments
  const YAML::Node camera_cfg = cfg_["camera"];

  const bool camera_enabled =
    camera_cfg && camera_cfg["enabled"].as<bool>(false);

  if (camera_enabled && !unity_render_) {
    throw std::invalid_argument(
      "Camera requires Unity rendering to be enabled.");
  }

  for (int i = 0; i < num_envs_; i++) {
    auto env = std::make_unique<EnvBase>();

    if (camera_enabled &&
        !env->configureCamera(camera_cfg)) {
      throw std::invalid_argument(
        "Invalid camera configuration.");
    }

    envs_.push_back(std::move(env));
  }

  // Initialize random generators
  setSeed(seed_);

  // set Unity
  setUnity(unity_render_);

  obs_dim_ = envs_[0]->getObsDim();
  act_dim_ = envs_[0]->getActDim();

  // generate reward names
  // compute it once to get reward names. actual value is not used
  envs_[0]->updateExtraInfo();
  for (auto& re : envs_[0]->extra_info_) {
    extra_info_names_.push_back(re.first);
  }
}

template<typename EnvBase>
VecEnv<EnvBase>::~VecEnv() {}

template<typename EnvBase>
bool VecEnv<EnvBase>::reset(Ref<MatrixRowMajor<>> obs) {
  if (obs.rows() != num_envs_ || obs.cols() != obs_dim_) {
    logger_.error(
      "Input matrix dimensions do not match with that of the environment.");
    return false;
  }

  receive_id_ = 0;
  for (int i = 0; i < num_envs_; i++) {
    envs_[i]->reset(obs.row(i));
  }
  return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::resetState(
    const Ref<const MatrixRowMajor<>> state,
    Ref<MatrixRowMajor<>> obs) {
  if (state.rows() != num_envs_ || state.cols() != 13 ||
      obs.rows() != num_envs_ || obs.cols() != obs_dim_ ||
      !state.allFinite()) {
    return false;
  }

  // Validate all quaternions before resetting environments.
  for (int i = 0; i < num_envs_; i++) {
    const Scalar q_norm = state.row(i).segment(3, 4).norm();
    if (!std::isfinite(q_norm) || q_norm < 1e-6f) {
      return false;
    }
  }

  receive_id_ = 0;

  for (int i = 0; i < num_envs_; i++) {
    Vector<> initial_state = state.row(i).transpose();

    if (!envs_[i]->resetState(initial_state, obs.row(i))) {
      return false;
    }
  }

  return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs,
                           Ref<Vector<>> reward, Ref<BoolVector<>> done,
                           Ref<MatrixRowMajor<>> extra_info) {
  if (act.rows() != num_envs_ || act.cols() != act_dim_ ||
      obs.rows() != num_envs_ || obs.cols() != obs_dim_ ||
      reward.rows() != num_envs_ || reward.cols() != 1 ||
      done.rows() != num_envs_ || done.cols() != 1 ||
      extra_info.rows() != num_envs_ ||
      extra_info.cols() != extra_info_names_.size()) {
    logger_.error(
      "Input matrix dimensions do not match with that of the environment.");
    return false;
  }

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_envs_; i++) {
    perAgentStep(i, act, obs, reward, done, extra_info);
  }

  if (unity_render_ && unity_ready_) {
    if (!render(last_render_frame_id_ + 1)) {
      logger_.warn("Automatic Unity rendering failed.");
      return false;
    }
  }
  return true;
}

template<typename EnvBase>
void VecEnv<EnvBase>::testStep(Ref<MatrixRowMajor<>> act,
                               Ref<MatrixRowMajor<>> obs, Ref<Vector<>> reward,
                               Ref<BoolVector<>> done,
                               Ref<MatrixRowMajor<>> extra_info) {
  perAgentStep(0, act, obs, reward, done, extra_info);
  envs_[0]->getObs(obs.row(0));
}

template<typename EnvBase>
void VecEnv<EnvBase>::close() {
  for (int i = 0; i < num_envs_; i++) {
    envs_[i]->close();
  }
}

template<typename EnvBase>
void VecEnv<EnvBase>::setSeed(const int seed) {
  int seed_inc = seed;
  for (int i = 0; i < num_envs_; i++) envs_[i]->setSeed(seed_inc++);
}

template<typename EnvBase>
void VecEnv<EnvBase>::getObs(Ref<MatrixRowMajor<>> obs) {
  for (int i = 0; i < num_envs_; i++) envs_[i]->getObs(obs.row(i));
}


template<typename EnvBase>
size_t VecEnv<EnvBase>::getEpisodeLength(void) {
  if (envs_.size() <= 0) {
    return 0;
  } else {
    return (size_t)envs_[0]->getMaxT() / envs_[0]->getSimTimeStep();
  }
}

template<typename EnvBase>
void VecEnv<EnvBase>::perAgentStep(int agent_id, Ref<MatrixRowMajor<>> act,
                                   Ref<MatrixRowMajor<>> obs,
                                   Ref<Vector<>> reward, Ref<BoolVector<>> done,
                                   Ref<MatrixRowMajor<>> extra_info) {
  reward(agent_id) =
    envs_[agent_id]->step(act.row(agent_id), obs.row(agent_id));

  Scalar terminal_reward = 0;
  done(agent_id) = envs_[agent_id]->isTerminalState(terminal_reward);

  envs_[agent_id]->updateExtraInfo();
  for (int j = 0; j < extra_info.cols(); j++)
    extra_info(agent_id, j) =
      envs_[agent_id]->extra_info_[extra_info_names_[j]];

  if (done[agent_id]) {
    envs_[agent_id]->reset(obs.row(agent_id));
    reward(agent_id) += terminal_reward;
  }
}

template<typename EnvBase>
bool VecEnv<EnvBase>::setUnity(bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    // add objects to Unity
    for (int i = 0; i < num_envs_; i++) {
      envs_[i]->addObjectsToUnity(unity_bridge_ptr_);
    }
    logger_.info("Flightmare Bridge is created.");
  }
  return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

template<typename EnvBase>
void VecEnv<EnvBase>::isTerminalState(Ref<BoolVector<>> terminal_state) {}

template<typename EnvBase>
bool VecEnv<EnvBase>::render(const FrameID frame_id) {
  if (!unity_render_ || !unity_ready_ ||
      unity_bridge_ptr_ == nullptr) {
    logger_.warn("Flightmare Unity is not connected.");
    return false;
  }

  // Frame IDs must be strictly increasing.
  if (frame_id <= last_render_frame_id_) {
    logger_.warn("Unity frame ID must be strictly increasing.");
    return false;
  }

  // Reserve the frame ID before sending the request.
  // Do not reuse it even if communication fails.
  last_render_frame_id_ = frame_id;

  if (!unity_bridge_ptr_->getRender(frame_id)) {
    return false;
  }

  return unity_bridge_ptr_->handleOutput();
}

template<typename EnvBase>
size_t VecEnv<EnvBase>::getNumOfCameras(const int env_id) const {
  if (env_id < 0 ||
      static_cast<size_t>(env_id) >= envs_.size()) {
    throw std::out_of_range("Invalid environment index.");
  }

  return envs_[env_id]->getNumOfCameras();
}

template<typename EnvBase>
bool VecEnv<EnvBase>::getRGBImage(const int env_id,
                                  const int camera_id,
                                  cv::Mat &image) {
  const size_t num_cameras = getNumOfCameras(env_id);

  if (camera_id < 0) {
    throw std::out_of_range("Invalid camera index.");
  }

  if (num_cameras == 0) {
    return false;
  }

  if (static_cast<size_t>(camera_id) >= num_cameras) {
    throw std::out_of_range("Invalid camera index.");
  }

  if (!unity_render_ || !unity_ready_) {
    return false;
  }

  return envs_[env_id]->getRGBImage(camera_id, image);
}

template<typename EnvBase>
bool VecEnv<EnvBase>::getDepthMap(const int env_id,
                                  const int camera_id,
                                  cv::Mat &image) {
  const size_t num_cameras = getNumOfCameras(env_id);

  if (camera_id < 0) {
    throw std::out_of_range("Invalid camera index.");
  }

  if (num_cameras == 0) {
    return false;
  }

  if (static_cast<size_t>(camera_id) >= num_cameras) {
    throw std::out_of_range("Invalid camera index.");
  }

  if (!unity_render_ || !unity_ready_) {
    return false;
  }

  return envs_[env_id]->getDepthMap(camera_id, image);
}

template<typename EnvBase>
void VecEnv<EnvBase>::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

template<typename EnvBase>
void VecEnv<EnvBase>::curriculumUpdate(void) {
  for (int i = 0; i < num_envs_; i++) envs_[i]->curriculumUpdate();
}

// template<typename EnvBase>
// std::ostream& operator<<(std::ostream& os, const VecEnv<EnvBase>& env) {
//   os.precision(3);
//   os << "Vectorized Environment:\n"
//      << "obs dim =            [" << env.obs_dim_ << "]\n"
//      << "act dim =            [" << env.act_dim_ << "]\n"
//      << "num_envs =           [" << env.num_envs_ << "]\n"
//      << "seed =               [" << env.seed_ << "]\n"
//      << "scene_id =           [" << env.scene_id_ << std::endl;
//   os.precision();
//   return os;
// }

// IMPORTANT. Otherwise:
// Segmentation fault (core dumped)
template class VecEnv<QuadrotorEnv>;

}  // namespace flightlib
