
// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

// flightlib
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/envs/test_env.hpp"
#include "flightlib/envs/vec_env.hpp"

namespace py = pybind11;
using namespace flightlib;

PYBIND11_MODULE(flightgym, m) {
  py::class_<VecEnv<QuadrotorEnv>>(m, "QuadrotorEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<QuadrotorEnv>::reset)
    .def("resetState", &VecEnv<QuadrotorEnv>::resetState)
    .def("step", &VecEnv<QuadrotorEnv>::step)
    .def("testStep", &VecEnv<QuadrotorEnv>::testStep)
    .def("setSeed", &VecEnv<QuadrotorEnv>::setSeed)
    .def("close", &VecEnv<QuadrotorEnv>::close)
    .def("isTerminalState", &VecEnv<QuadrotorEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<QuadrotorEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<QuadrotorEnv>::connectUnity)
    .def("render", &VecEnv<QuadrotorEnv>::render)
    .def("getNumOfCameras",
         &VecEnv<QuadrotorEnv>::getNumOfCameras,
         py::arg("env_id") = 0)
    .def("getRGBImage",
         [](VecEnv<QuadrotorEnv>& env,
            const int env_id,
            const int camera_id) -> py::object {
           cv::Mat bgr;

           if (!env.getRGBImage(env_id, camera_id, bgr)) {
             return py::none();
           }

           if (bgr.empty() || bgr.type() != CV_8UC3) {
             throw std::runtime_error(
               "Invalid RGB image format.");
           }

           cv::Mat rgb;
           cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

           py::array_t<uint8_t> result({
             static_cast<py::ssize_t>(rgb.rows),
             static_cast<py::ssize_t>(rgb.cols),
             static_cast<py::ssize_t>(3)
           });

           const size_t row_bytes =
             static_cast<size_t>(rgb.cols) * 3;

           uint8_t* dst = result.mutable_data();

           for (int row = 0; row < rgb.rows; ++row) {
             std::memcpy(
               dst + static_cast<size_t>(row) * row_bytes,
               rgb.ptr<uint8_t>(row),
               row_bytes);
           }

           return result;
         },
         py::arg("env_id") = 0,
         py::arg("camera_id") = 0)
    .def("getDepthMap",
         [](VecEnv<QuadrotorEnv>& env,
            const int env_id,
            const int camera_id) -> py::object {
           cv::Mat depth;

           if (!env.getDepthMap(env_id, camera_id, depth)) {
             return py::none();
           }

           if (depth.empty() || depth.type() != CV_32FC1) {
             throw std::runtime_error(
               "Invalid depth image format.");
           }

           py::array_t<float> result({
             static_cast<py::ssize_t>(depth.rows),
             static_cast<py::ssize_t>(depth.cols)
           });

           const size_t row_bytes =
             static_cast<size_t>(depth.cols) * sizeof(float);

           float* dst = result.mutable_data();

           for (int row = 0; row < depth.rows; ++row) {
             std::memcpy(
               dst + static_cast<size_t>(row) * depth.cols,
               depth.ptr<float>(row),
               row_bytes);
           }

           return result;
         },
         py::arg("env_id") = 0,
         py::arg("camera_id") = 0)
    .def("disconnectUnity", &VecEnv<QuadrotorEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<QuadrotorEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<QuadrotorEnv>::getObsDim)
    .def("getActDim", &VecEnv<QuadrotorEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<QuadrotorEnv>::getExtraInfoNames)
    .def("__repr__", [](const VecEnv<QuadrotorEnv>& a) {
      return "RPG Drone Racing Environment";
    });

  py::class_<TestEnv<QuadrotorEnv>>(m, "TestEnv_v0")
    .def(py::init<>())
    .def("reset", &TestEnv<QuadrotorEnv>::reset)
    .def("__repr__", [](const TestEnv<QuadrotorEnv>& a) { return "Test Env"; });
}