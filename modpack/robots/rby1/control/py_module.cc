#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "force_control/admittance_controller.h"
#include "realtime_driver.h"

namespace py = pybind11;
using rby1::control::RealtimeDriver;

namespace {

void BindSnapshot(py::module_& m) {
  py::class_<RealtimeDriver::RobotSnapshot>(m, "RobotSnapshot")
      .def_property_readonly(
          "timestamp_ns",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.timestamp_ns;
          })
      .def_property_readonly(
          "joint_is_ready",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_is_ready;
          })
      .def_property_readonly(
          "joint_position",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_position;
          })
      .def_property_readonly(
          "joint_velocity",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_velocity;
          })
      .def_property_readonly(
          "joint_current",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_current;
          })
      .def_property_readonly(
          "joint_torque",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_torque;
          })
      .def_property_readonly(
          "joint_target_position",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_target_position;
          })
      .def_property_readonly(
          "joint_target_velocity",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_target_velocity;
          })
      .def_property_readonly(
          "joint_feedback_gain",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_feedback_gain;
          })
      .def_property_readonly(
          "joint_feedforward_torque",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_feedforward_torque;
          })
      .def_property_readonly(
          "odom_SE2",
      [](const RealtimeDriver::RobotSnapshot& self) {
        return self.odom_SE2;
      })
      .def_property_readonly(
          "left_ee_wrench",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.left_ee_wrench;
          })
      .def_property_readonly(
          "right_ee_wrench",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.right_ee_wrench;
          })
      .def_property_readonly(
          "left_ft_valid",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.left_ft_valid;
          })
      .def_property_readonly(
          "right_ft_valid",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.right_ft_valid;
          })
      .def_readonly("is_valid", &RealtimeDriver::RobotSnapshot::is_valid);
}

void BindConfig(py::module_& m) {
  py::class_<RealtimeDriver::Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("robot_address", &RealtimeDriver::Config::robot_address)
      .def_readwrite("low_pass_freq_hz",
                     &RealtimeDriver::Config::low_pass_freq_hz)
      .def_readwrite("expect_wheel_velocity",
                     &RealtimeDriver::Config::expect_wheel_velocity)
      .def_readwrite("command_timeout_us",
                     &RealtimeDriver::Config::command_timeout_us);
}

void BindDriver(py::module_& m) {
  py::class_<RealtimeDriver>(m, "RealtimeDriver")
      .def(py::init<RealtimeDriver::Config>(),
           py::arg("config") = RealtimeDriver::Config(),
           "Create a realtime controller with the given configuration.")
      .def(
          "run",
          [](RealtimeDriver& self) {
            py::gil_scoped_release release;
            self.Run();
          },
          "Run the control loop in the current thread.")
      .def("start",
           [](RealtimeDriver& self) {
             py::gil_scoped_release release;
             self.Start();
           },
           "Start the control loop on a background thread.")
      .def("stop",
           [](RealtimeDriver& self) {
             py::gil_scoped_release release;
             self.Stop();
           },
           "Stop the control loop and wait for shutdown.")
      .def("is_running", &RealtimeDriver::IsRunning,
           "Return true while the control loop is active.")
      .def("wait_until_ready", &RealtimeDriver::WaitUntilReady,
           py::arg("timeout_sec"), py::call_guard<py::gil_scoped_release>(),
           "Block until all robot components report ready or timeout occurs.")
      .def("set_component_position_targets",
           &RealtimeDriver::SetComponentPositionTargets, py::arg("name"),
           py::arg("targets"),
           "Set joint position targets for a named component (e.g. 'LEFT').")
      .def("set_component_velocity_targets",
           &RealtimeDriver::SetComponentVelocityTargets, py::arg("name"),
           py::arg("targets"),
           "Set joint velocity targets for a named component.")
      .def("set_body_position_targets",
           &RealtimeDriver::SetBodyPositionTargets, py::arg("targets"),
           "Apply joint position targets across torso, left arm, right arm, "
           "and head in SDK order.")
      .def("set_base_twist_command", &RealtimeDriver::SetBaseTwistCommand,
           py::arg("twist_body"),
           "Command the base using body-frame velocities [vx, vy, wz].")
      .def("set_wheel_velocity_targets",
           &RealtimeDriver::SetWheelVelocityTargets, py::arg("targets"),
           "Direct wheel velocity targets when expect_wheel_velocity is true.")
      .def(
          "get_latest_robot_state",
          [](const RealtimeDriver& self) -> py::object {
            // Release the GIL only while invoking the C++ getter, then
            // reacquire it before interacting with Python objects.
            const auto snapshot = [&self]() {
              py::gil_scoped_release release;
              return self.GetLatestRobotState();
            }();
            if (!snapshot.has_value()) {
              return py::none();
            }
            return py::cast(snapshot.value());
          },
          "Return the most recent robot state snapshot, or None if unavailable.")
      .def("__enter__",
           [](RealtimeDriver& self) -> RealtimeDriver& {
             self.Start();
             return self;
           })
      .def("__exit__",
           [](RealtimeDriver& self, py::object, py::object, py::object) {
             self.Stop();
           return false;
          });
}

void BindAdmittanceController(py::module_& m) {
  using Controller = ::AdmittanceController;
  using Config = Controller::AdmittanceControllerConfig;
  using Compliance = Config::ComplianceParameters6d;
  using PID = Config::PIDGains;

  auto config_cls = py::class_<Config>(m, "AdmittanceControllerConfig");
  py::class_<Compliance>(config_cls, "ComplianceParameters6d")
      .def(py::init<>())
      .def_readwrite("stiffness", &Compliance::stiffness,
                     "Diagonal stiffness matrix (6x6).")
      .def_readwrite("damping", &Compliance::damping,
                     "Diagonal damping matrix (6x6).")
      .def_readwrite("inertia", &Compliance::inertia,
                     "Diagonal inertia matrix (6x6).")
      .def_readwrite("stiction", &Compliance::stiction,
                     "Static friction vector (6,).");

  py::class_<PID>(config_cls, "PIDGains")
      .def(py::init<>())
      .def_readwrite("P_trans", &PID::P_trans)
      .def_readwrite("I_trans", &PID::I_trans)
      .def_readwrite("D_trans", &PID::D_trans)
      .def_readwrite("P_rot", &PID::P_rot)
      .def_readwrite("I_rot", &PID::I_rot)
      .def_readwrite("D_rot", &PID::D_rot);

  config_cls
      .def(py::init<>())
      .def_readwrite("dt", &Config::dt)
      .def_readwrite("log_to_file", &Config::log_to_file)
      .def_readwrite("log_file_path", &Config::log_file_path)
      .def_readwrite("alert_overrun", &Config::alert_overrun)
      .def_readwrite("compliance6d", &Config::compliance6d)
      .def_readwrite("max_spring_force_magnitude",
                     &Config::max_spring_force_magnitude)
      .def_readwrite("max_spring_torque_magnitude",
                     &Config::max_spring_torque_magnitude)
      .def_readwrite("direct_force_control_gains",
                     &Config::direct_force_control_gains)
      .def_readwrite("direct_force_control_I_limit",
                     &Config::direct_force_control_I_limit);

  py::class_<Controller>(m, "AdmittanceController")
      .def(py::init<>())
      .def(
          "init",
          [](Controller& self, const Config& config,
             const Eigen::Matrix<double, 7, 1>& pose_current,
             std::optional<int64_t> time_ns) {
            RUT::TimePoint time_point;
            if (time_ns.has_value()) {
              auto duration = std::chrono::nanoseconds(time_ns.value());
              time_point = RUT::TimePoint(
                  std::chrono::duration_cast<RUT::Clock::duration>(duration));
            } else {
              time_point = RUT::Clock::now();
            }
            return self.init(time_point, config, pose_current);
          },
          py::arg("config"), py::arg("pose_current"),
          py::arg("time_ns") = std::nullopt,
          "Initialize the controller with the provided configuration. "
          "Optionally provide a start time in nanoseconds.")
      .def(
          "set_robot_status",
          [](Controller& self, const Eigen::Matrix<double, 7, 1>& pose_WT,
             const Eigen::Matrix<double, 6, 1>& wrench_T) {
            py::gil_scoped_release release;
            self.setRobotStatus(pose_WT, wrench_T);
          },
          py::arg("pose_WT"), py::arg("wrench_T"),
          "Update the current robot pose and measured wrench.")
      .def(
          "set_robot_reference",
          [](Controller& self, const Eigen::Matrix<double, 7, 1>& pose_WT,
             const Eigen::Matrix<double, 6, 1>& wrench_WTr) {
            py::gil_scoped_release release;
            self.setRobotReference(pose_WT, wrench_WTr);
          },
          py::arg("pose_WT"), py::arg("wrench_WTr"),
          "Set the desired pose and wrench reference.")
      .def(
          "set_force_controlled_axis",
          [](Controller& self, const Eigen::Matrix<double, 6, 6>& Tr,
             int n_af) {
            py::gil_scoped_release release;
            self.setForceControlledAxis(Tr, n_af);
          },
          py::arg("Tr"), py::arg("n_af"),
          "Specify the force-controlled axes selection.")
      .def(
          "set_stiffness_matrix",
          [](Controller& self, const Eigen::Matrix<double, 6, 6>& stiffness) {
            py::gil_scoped_release release;
            self.setStiffnessMatrix(stiffness);
          },
          py::arg("stiffness"))
      .def(
          "set_damping_matrix",
          [](Controller& self, const Eigen::Matrix<double, 6, 6>& damping) {
            py::gil_scoped_release release;
            self.setDampingMatrix(damping);
          },
          py::arg("damping"))
      .def(
          "step",
          [](Controller& self) {
            RUT::Vector7d pose = RUT::Vector7d::Zero();
            int status;
            {
              py::gil_scoped_release release;
              status = self.step(pose);
            }
            return py::make_tuple(status, pose);
          },
          "Run one control step and return (status, pose).");
}

}  // namespace

PYBIND11_MODULE(rby1_controller, m) {
  m.doc() = "Pybind11 bindings for the RBY1 realtime controller";

  BindSnapshot(m);
  BindConfig(m);
  BindDriver(m);
  BindAdmittanceController(m);

  m.def(
      "debug_echo",
      [](const std::string& message) {
        std::cout << "[pybind debug] " << message << std::endl;
        return message;
      },
      py::arg("message"),
      "Print the supplied message from within the pybind module and return it.");
}
