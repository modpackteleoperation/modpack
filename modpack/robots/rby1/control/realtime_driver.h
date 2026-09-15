#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "rby1-sdk/model.h"
#include "rby1-sdk/net/real_time_control_protocol.h"
#include "rby1-sdk/robot.h"

namespace rby1::control {

// Forward declaration.
class Rby1Component;

class RealtimeDriver {
 public:
  struct RobotSnapshot {
    int64_t timestamp_ns = 0;
    std::vector<bool> joint_is_ready;
    std::vector<double> joint_position;
    std::vector<double> joint_velocity;
    std::vector<double> joint_current;
    std::vector<double> joint_torque;
    std::vector<double> joint_target_position;
    std::vector<double> joint_target_velocity;
    std::vector<double> joint_feedback_gain;
    std::vector<double> joint_feedforward_torque;
    Eigen::Matrix3d odom_SE2 = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 1> left_ee_wrench =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> right_ee_wrench =
        Eigen::Matrix<double, 6, 1>::Zero();
    bool left_ft_valid = false;
    bool right_ft_valid = false;
    bool is_valid = false;
  };

  struct Config {
    std::string robot_address{"192.168.30.1:50051"};
    double low_pass_freq_hz{1.0};
    bool expect_wheel_velocity{false};
    int64_t command_timeout_us{1000000};
  };

  RealtimeDriver();
  explicit RealtimeDriver(const Config& config);
  ~RealtimeDriver();

  RealtimeDriver(const RealtimeDriver&) = delete;
  RealtimeDriver& operator=(const RealtimeDriver&) = delete;

  // Blocking control loop. Returns when Stop() is requested or the robot
  // reports finish.
  void Run();

  // Starts the control loop on a background thread. Start() is a noop if the
  // driver is already running.
  void Start();

  // Requests the control loop to finish and waits for the background thread to
  // exit (if one is running).
  void Stop();

  // Returns true if the control loop is currently running.
  bool IsRunning() const { return running_.load(); }

  // Blocks until all components report ready or timeout_sec is reached.
  bool WaitUntilReady(double timeout_sec);

  bool SetComponentPositionTargets(const std::string& component_name,
                                   const std::vector<double>& targets);
  bool SetComponentVelocityTargets(const std::string& component_name,
                                   const std::vector<double>& targets);

  // Convenience helper that dispatches the provided joint positions across the
  // torso, left arm, right arm, and head components (in that order).
  bool SetBodyPositionTargets(const std::vector<double>& targets);

  // Sets the base command as a body-frame twist [vx, vy, wz].
  bool SetBaseTwistCommand(const Eigen::Vector3d& twist_body);

  // Direct access to wheel velocity targets when running in velocity mode.
  bool SetWheelVelocityTargets(const std::vector<double>& wheel_targets);

  std::optional<RobotSnapshot> GetLatestRobotState() const;

 private:
  using y1_instance = rb::y1_model::M;

  struct DriverState;

  void InitializeRobot();
  void InitializeComponents();
  void AddComponent(const std::string& name, std::vector<int> dof);
  void ResetControl();
  void HandleStateUpdate(const rb::RobotState<y1_instance>& state);
  rb::ControlInput<y1_instance> DoControl(
      const rb::ControlState<y1_instance>& state);
  bool AllComponentsReady() const;

  Config config_;
  std::shared_ptr<rb::Robot<y1_instance>> robot_;

  std::unique_ptr<DriverState> state_;
  std::thread control_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
};

}  // namespace rby1::control
