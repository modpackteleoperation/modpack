#include "realtime_driver.h"

#include <csignal>
#include <cstring>

#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <thread>
#include <utility>

#include <Eigen/Geometry>

namespace rby1::control {

using y1_instance = rb::y1_model::M;

constexpr double kControlPeriodSec = 0.002;     // 500 Hz
constexpr double kWheelTrack = 0.49;            // m
constexpr double kWheelBase = 0.49;             // m
constexpr double kWheelDiameter = 0.1525;       // m

template <typename Clock>
int64_t GetTimeUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             Clock::now().time_since_epoch())
      .count();
}

class MecanumWheelVelocityKinematics {
 public:
  MecanumWheelVelocityKinematics(double track, double wheelbase,
                                 double wheel_radius)
      : inv_wheel_radius_(1.0 / wheel_radius) {
    const double lx = track * 0.5;
    const double ly = wheelbase * 0.5;
    // [FR, FL, RR, RL]
    vehicle_to_wheel_map_.row(0) = Eigen::Vector3d(1.0, 1.0, (lx + ly));
    vehicle_to_wheel_map_.row(1) = Eigen::Vector3d(1.0, -1.0, -(lx + ly));
    vehicle_to_wheel_map_.row(2) = Eigen::Vector3d(1.0, -1.0, (lx + ly));
    vehicle_to_wheel_map_.row(3) = Eigen::Vector3d(1.0, 1.0, -(lx + ly));

    wheel_to_vehicle_map_.row(0) = Eigen::Vector4d(1.0, 1.0, 1.0, 1.0);
    wheel_to_vehicle_map_.row(1) = Eigen::Vector4d(1.0, -1.0, -1.0, 1.0);
    wheel_to_vehicle_map_.row(2) = Eigen::Vector4d(
        1.0 / (lx + ly), -1.0 / (lx + ly), 1.0 / (lx + ly), -1.0 / (lx + ly));
  }

  Eigen::Vector4d CalcWheelVelocity(const Eigen::Vector3d& vehicle_velocity)
      const {
    return (vehicle_to_wheel_map_ * vehicle_velocity) * inv_wheel_radius_;
  }

 private:
  double inv_wheel_radius_;
  Eigen::Matrix<double, 4, 3> vehicle_to_wheel_map_;
  Eigen::Matrix<double, 3, 4> wheel_to_vehicle_map_;
};

class Rby1Component {
 public:
  enum class Mode : int8_t { kPosition = 1, kVelocity = 2 };

  struct CommandData {
    int64_t utime = 0;
    std::vector<Mode> mode;
    std::vector<double> target;
  };

  Rby1Component(std::string name, std::vector<int> dof,
                double low_pass_freq_hz, int64_t command_timeout_us)
      : name_(std::move(name)),
        dof_(std::move(dof)),
        low_pass_freq_hz_(low_pass_freq_hz),
        command_timeout_us_(command_timeout_us),
        ready_(dof_.size(), false),
        position_(dof_.size(), 0.0),
        previous_command_(dof_.size(), 0.0) {}

  size_t dof_count() const { return dof_.size(); }

  bool is_ready() const {
    std::lock_guard<std::mutex> guard(state_mutex_);
    for (bool ready : ready_) {
      if (!ready) {
        return false;
      }
    }
    return true;
  }

  void UpdateState(const rb::RobotState<y1_instance>& state) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    for (size_t i = 0; i < dof_.size(); ++i) {
      const int idx = dof_[i];
      ready_[i] = state.is_ready[idx];
      position_[i] = state.position[idx];
    }
  }

  void ResetControl(rb::ControlInput<y1_instance>* control_input) {
    const std::vector<double> snapshot = CurrentPositionSnapshot();
    for (size_t i = 0; i < dof_.size(); ++i) {
      const int idx = dof_[i];
      control_input->mode[idx] = rb::kPositionControlMode;
      control_input->target[idx] = snapshot[i];
    }
  }

  bool SetCommand(const CommandData& command) {
    if (command.mode.size() != dof_.size() ||
        command.target.size() != dof_.size()) {
      std::cerr << "Command size mismatch for component " << name_
                << std::endl;
      return false;
    }

    std::lock_guard<std::mutex> guard(command_mutex_);
    command_ = command;
    last_command_time_ = GetTimeUs<std::chrono::steady_clock>();
    return true;
  }

  bool SetPositionTargets(const std::vector<double>& targets) {
    CommandData command;
    command.utime = GetTimeUs<std::chrono::steady_clock>();
    command.mode.assign(dof_.size(), Mode::kPosition);
    command.target = targets;
    return SetCommand(command);
  }

  bool SetVelocityTargets(const std::vector<double>& targets) {
    CommandData command;
    command.utime = GetTimeUs<std::chrono::steady_clock>();
    command.mode.assign(dof_.size(), Mode::kVelocity);
    command.target = targets;
    return SetCommand(command);
  }

  void Apply(int64_t steady_now_us, rb::ControlInput<y1_instance>* control_input,
             double delta_time) {
    std::lock_guard<std::mutex> guard(command_mutex_);
    if (!command_) {
      return;
    }

    if ((steady_now_us - last_command_time_) > command_timeout_us_) {
      std::cerr << "Stopping commands for " << name_ << std::endl;
      last_command_time_ = 0;
      command_.reset();
      for (size_t i = 0; i < dof_.size(); ++i) {
        const int idx = dof_[i];
        if (control_input->mode[idx] == rb::kVelocityControlMode) {
          control_input->target[idx] = 0.0;
        }
      }
      return;
    }

    const std::vector<double> filtered = LowPass(command_->target, delta_time);

    for (size_t i = 0; i < dof_.size(); ++i) {
      const int idx = dof_[i];
      if (command_->mode[i] == Mode::kPosition) {
        control_input->mode[idx] = rb::kPositionControlMode;
      } else {
        control_input->mode[idx] = rb::kVelocityControlMode;
      }
      control_input->target[idx] = filtered[i];
    }
  }

 private:
  std::vector<double> CurrentPositionSnapshot() const {
    std::lock_guard<std::mutex> guard(state_mutex_);
    return position_;
  }

  std::vector<double> LowPass(const std::vector<double>& raw,
                              double delta_time) {
    if (raw.size() != previous_command_.size()) {
      previous_command_ = raw;
      lpf_initialized_ = true;
      return previous_command_;
    }

    if (!lpf_initialized_ || low_pass_freq_hz_ <= 0.0) {
      previous_command_ = raw;
      lpf_initialized_ = true;
      return previous_command_;
    }

    const double time_constant = 1.0 / (2.0 * M_PI * low_pass_freq_hz_);
    const double alpha = delta_time / (time_constant + delta_time);
    for (size_t i = 0; i < raw.size(); ++i) {
      previous_command_[i] =
          alpha * raw[i] + (1.0 - alpha) * previous_command_[i];
    }
    return previous_command_;
  }

  std::string name_;
  std::vector<int> dof_;
  double low_pass_freq_hz_;
  int64_t command_timeout_us_;

  mutable std::mutex state_mutex_;
  std::vector<bool> ready_;
  std::vector<double> position_;

  std::mutex command_mutex_;
  std::optional<CommandData> command_;
  int64_t last_command_time_ = 0;

  std::vector<double> previous_command_;
  bool lpf_initialized_ = false;
};

struct RealtimeDriver::DriverState {
  explicit DriverState(const Config& cfg)
      : mecanum(kWheelTrack, kWheelBase, kWheelDiameter * 0.5),
        config(cfg) {}

  Config config;
  rb::ControlInput<y1_instance> control_input;
  std::vector<std::unique_ptr<Rby1Component>> component_storage;
  std::vector<Rby1Component*> components;
  std::unordered_map<std::string, Rby1Component*> component_lookup;
  Rby1Component* wheel_component = nullptr;
  MecanumWheelVelocityKinematics mecanum;
  std::atomic<bool> state_update_running{false};
  mutable std::mutex snapshot_mutex;
  mutable RealtimeDriver::RobotSnapshot snapshot;
};

RealtimeDriver::RealtimeDriver() : RealtimeDriver(Config{}) {}

RealtimeDriver::RealtimeDriver(const Config& config)
    : config_(config),
      robot_(rb::Robot<y1_instance>::Create(config_.robot_address)),
      state_(std::make_unique<DriverState>(config_)) {
  if (!robot_) {
    throw std::runtime_error("Failed to create robot instance");
  }
  std::cout << "Attempting to connect to the robot..." << std::endl;
  if (!robot_->Connect()) {
    throw std::runtime_error("Unable to establish connection to the robot");
  }
  std::cout << "Successfully connected to the robot." << std::endl;

  InitializeRobot();
  InitializeComponents();
}

RealtimeDriver::~RealtimeDriver() {
  Stop();
}

void RealtimeDriver::Run() {
  if (running_.exchange(true)) {
    return;
  }
  stop_requested_.store(false);

  robot_->StartStateUpdate(
      [this](const rb::RobotState<y1_instance>& state, const auto&) {
        HandleStateUpdate(state);
      },
      100);
  state_->state_update_running.store(true);

  // Wait for readiness.
  while (!AllComponentsReady()) {
    if (stop_requested_.load()) {
      running_.store(false);
      robot_->StopStateUpdate();
      state_->state_update_running.store(false);
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ResetControl();
  std::cout << "Starting realtime control loop." << std::endl;
  robot_->Control(
      [this](const rb::ControlState<y1_instance>& state) {
        return DoControl(state);
      },
      0, 10);

  if (state_->state_update_running.exchange(false)) {
    robot_->StopStateUpdate();
  }
  running_.store(false);
}

void RealtimeDriver::Start() {
  if (running_.load()) {
    return;
  }
  control_thread_ = std::thread([this]() {
    try {
      Run();
    } catch (const std::exception& e) {
      std::cerr << "RealtimeDriver thread terminated: " << e.what()
                << std::endl;
      running_.store(false);
    }
  });
}

void RealtimeDriver::Stop() {
  stop_requested_.store(true);
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  if (state_ && state_->state_update_running.exchange(false)) {
    robot_->StopStateUpdate();
  }
  running_.store(false);
  stop_requested_.store(false);
}

bool RealtimeDriver::WaitUntilReady(double timeout_sec) {
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_sec));
  while (!AllComponentsReady()) {
    if (stop_requested_.load()) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

bool RealtimeDriver::SetComponentPositionTargets(
    const std::string& component_name, const std::vector<double>& targets) {
  const auto it = state_->component_lookup.find(component_name);
  if (it == state_->component_lookup.end()) {
    std::cerr << "Unknown component: " << component_name << std::endl;
    return false;
  }
  return it->second->SetPositionTargets(targets);
}

bool RealtimeDriver::SetComponentVelocityTargets(
    const std::string& component_name, const std::vector<double>& targets) {
  const auto it = state_->component_lookup.find(component_name);
  if (it == state_->component_lookup.end()) {
    std::cerr << "Unknown component: " << component_name << std::endl;
    return false;
  }
  return it->second->SetVelocityTargets(targets);
}

bool RealtimeDriver::SetBodyPositionTargets(const std::vector<double>& targets) {
  const std::vector<std::string> body_components = {
      "TORSO", "LEFT", "RIGHT", "HEAD"};
  size_t expected = 0;
  for (const std::string& name : body_components) {
    const auto it = state_->component_lookup.find(name);
    if (it == state_->component_lookup.end()) {
      std::cerr << "Component missing: " << name << std::endl;
      return false;
    }
    expected += it->second->dof_count();
  }

  if (targets.size() != expected) {
    std::cerr << "Body target size mismatch. Expected " << expected
              << " values, got " << targets.size() << std::endl;
    return false;
  }

  size_t offset = 0;
  for (const std::string& name : body_components) {
    Rby1Component* component = state_->component_lookup.at(name);
    const size_t count = component->dof_count();
    if (!component->SetPositionTargets(std::vector<double>(
            targets.begin() + offset, targets.begin() + offset + count))) {
      return false;
    }
    offset += count;
  }
  return true;
}

bool RealtimeDriver::SetBaseTwistCommand(const Eigen::Vector3d& twist_body) {
  if (config_.expect_wheel_velocity) {
    std::cerr << "Driver configured for direct wheel velocity commands."
              << std::endl;
    return false;
  }
  if (state_->wheel_component == nullptr) {
    std::cerr << "Wheel component not initialized." << std::endl;
    return false;
  }

  const Eigen::Vector4d wheel_velocity =
      state_->mecanum.CalcWheelVelocity(twist_body);

  return state_->wheel_component->SetVelocityTargets(
    std::vector<double>(wheel_velocity.data(),
                        wheel_velocity.data() + wheel_velocity.size()));
}

bool RealtimeDriver::SetWheelVelocityTargets(
    const std::vector<double>& wheel_targets) {
  if (!config_.expect_wheel_velocity) {
    std::cerr << "Driver configured for base twist commands." << std::endl;
    return false;
  }
  if (state_->wheel_component == nullptr) {
    std::cerr << "Wheel component not initialized." << std::endl;
    return false;
  }
  return state_->wheel_component->SetVelocityTargets(wheel_targets);
}

void RealtimeDriver::InitializeRobot() {
  std::cout << "Checking power status..." << std::endl;
  if (!robot_->IsPowerOn(".*")) {
    std::cout << "Power is currently OFF. Attempting to power on..."
              << std::endl;
    if (!robot_->PowerOn(".*")) {
      throw std::runtime_error("Failed to power on the robot.");
    }
    std::cout << "Robot powered on successfully." << std::endl;
  } else {
    std::cout << "Power is already ON." << std::endl;
  }

  std::cout << "Checking servo status..." << std::endl;
  if (!robot_->IsServoOn(".*")) {
    std::cout << "Servo is currently OFF. Attempting to activate servo..."
              << std::endl;
    if (!robot_->ServoOn(".*")) {
      throw std::runtime_error("Failed to activate servo.");
    }
    std::cout << "Servo activated successfully." << std::endl;
  } else {
    std::cout << "Servo is already ON." << std::endl;
  }

  rb::ControlManagerState control_manager_state =
      robot_->GetControlManagerState();
  if (control_manager_state.state ==
          rb::ControlManagerState::State::kMajorFault ||
      control_manager_state.state ==
          rb::ControlManagerState::State::kMinorFault) {
    std::cout << "Attempting to reset control manager fault..." << std::endl;
    if (!robot_->ResetFaultControlManager()) {
      throw std::runtime_error("Unable to reset control manager fault.");
    }
    std::cout << "Fault reset successfully." << std::endl;
  }

  control_manager_state = robot_->GetControlManagerState();
  if (control_manager_state.state ==
      rb::ControlManagerState::State::kEnabled) {
    std::cout << "Disabling control manager to set joint gains." << std::endl;
    if (!robot_->DisableControlManager()) {
      throw std::runtime_error("Unable to disable control manager.");
    }
  }

  robot_->SetPositionPIDGain("torso_0", 100, 1, 500);
  robot_->SetPositionPIDGain("torso_1", 100, 1, 400);
  robot_->SetPositionPIDGain("torso_2", 250, 1, 400);
  robot_->SetPositionPIDGain("torso_3", 500, 40, 400);
  robot_->SetPositionPIDGain("torso_4", 500, 20, 400);
  robot_->SetPositionPIDGain("torso_5", 500, 40, 400);

  constexpr uint16_t p_gain = 150;
  constexpr uint16_t d_gain = 750;
  // if (!robot_->SetPositionPGain("head_0", p_gain)) {
  //   throw std::runtime_error("Unable to set head_0 P gain.");
  // }
  // if (!robot_->SetPositionDGain("head_0", d_gain)) {
  //   throw std::runtime_error("Unable to set head_0 D gain.");
  // }

  std::cout << "Enabling the control manager..." << std::endl;
  if (!robot_->EnableControlManager()) {
    throw std::runtime_error("Failed to enable control manager.");
  }
  std::cout << "Control manager enabled successfully." << std::endl;

  // if (!robot_->SetToolFlangeOutputVoltage("left", 12)) {
  //   throw std::runtime_error("Failed to set left tool flange voltage.");
  // }
  // if (!robot_->SetToolFlangeOutputVoltage("right", 12)) {
  //   throw std::runtime_error("Failed to set right tool flange voltage.");
  // }
}

void RealtimeDriver::InitializeComponents() {
  AddComponent("WHEEL",
               std::vector<int>(y1_instance::kMobilityIdx.begin(),
                                y1_instance::kMobilityIdx.end()));
  state_->wheel_component = state_->component_lookup.at("WHEEL");

  AddComponent("TORSO",
               std::vector<int>(y1_instance::kTorsoIdx.begin(),
                                y1_instance::kTorsoIdx.end()));
  AddComponent("LEFT",
               std::vector<int>(y1_instance::kLeftArmIdx.begin(),
                                y1_instance::kLeftArmIdx.end()));
  AddComponent("RIGHT",
               std::vector<int>(y1_instance::kRightArmIdx.begin(),
                                y1_instance::kRightArmIdx.end()));
  AddComponent("HEAD",
               std::vector<int>(y1_instance::kHeadIdx.begin(),
                                y1_instance::kHeadIdx.end()));
}

void RealtimeDriver::AddComponent(const std::string& name,
                                  std::vector<int> dof) {
  auto component = std::make_unique<Rby1Component>(
      name, std::move(dof), config_.low_pass_freq_hz, config_.command_timeout_us);
  state_->component_lookup[name] = component.get();
  state_->components.push_back(component.get());
  state_->component_storage.push_back(std::move(component));
}

void RealtimeDriver::ResetControl() {
  state_->control_input.mode.setConstant(rb::kPositionControlMode);
  state_->control_input.feedback_gain.setConstant(4);
  state_->control_input.feedforward_torque.setConstant(0);
  state_->control_input.feedback_gain.segment<6>(4).setConstant(8);

  for (Rby1Component* component : state_->components) {
    component->ResetControl(&state_->control_input);
  }

  state_->control_input.finish = false;
}

void RealtimeDriver::HandleStateUpdate(
    const rb::RobotState<y1_instance>& state) {
  for (Rby1Component* component : state_->components) {
    component->UpdateState(state);
  }
  {
    std::lock_guard<std::mutex> guard(state_->snapshot_mutex);
    auto& snapshot = state_->snapshot;
    snapshot.timestamp_ns =
        static_cast<int64_t>(state.timestamp.tv_sec) * 1000000000LL +
        static_cast<int64_t>(state.timestamp.tv_nsec);

    const int joint_count = state.position.size();
    snapshot.joint_is_ready.resize(joint_count);
    for (int i = 0; i < joint_count; ++i) {
      snapshot.joint_is_ready[i] = state.is_ready[i];
    }

    snapshot.joint_position.assign(state.position.data(),
                                   state.position.data() + joint_count);
    snapshot.joint_velocity.assign(state.velocity.data(),
                                   state.velocity.data() + joint_count);
    snapshot.joint_current.assign(state.current.data(),
                                  state.current.data() + joint_count);
    snapshot.joint_torque.assign(state.torque.data(),
                                 state.torque.data() + joint_count);
    snapshot.joint_target_position.assign(
        state.target_position.data(),
        state.target_position.data() + joint_count);
    snapshot.joint_target_velocity.assign(
        state.target_velocity.data(),
        state.target_velocity.data() + joint_count);
    snapshot.joint_feedback_gain.assign(
        state.target_feedback_gain.data(),
        state.target_feedback_gain.data() + joint_count);
    snapshot.joint_feedforward_torque.assign(
        state.target_feedforward_torque.data(),
        state.target_feedforward_torque.data() + joint_count);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        snapshot.odom_SE2(r, c) = state.odometry(r, c);
      }
    }
    auto sanitize_sensor =
        [](const rb::FTSensorData& sensor,
           Eigen::Matrix<double, 6, 1>& wrench_out) -> bool {
      wrench_out.head<3>() = sensor.force;
      wrench_out.tail<3>() = sensor.torque;
      if (!wrench_out.allFinite()) {
        wrench_out.setZero();
        return false;
      }
      const double max_abs = wrench_out.array().abs().maxCoeff();
      if (!std::isfinite(max_abs) || max_abs > 1e6) {
        wrench_out.setZero();
        return false;
      }
      return true;
    };

    snapshot.left_ft_valid =
        sanitize_sensor(state.ft_sensor_left, snapshot.left_ee_wrench);
    snapshot.right_ft_valid =
        sanitize_sensor(state.ft_sensor_right, snapshot.right_ee_wrench);
    snapshot.is_valid = true;
  }
}

rb::ControlInput<y1_instance> RealtimeDriver::DoControl(
    const rb::ControlState<y1_instance>& state) {
  (void)state;
  const int64_t steady_now = GetTimeUs<std::chrono::steady_clock>();

  for (Rby1Component* component : state_->components) {
    component->Apply(steady_now, &state_->control_input, kControlPeriodSec);
  }

  // Periodic debug: print wheel control modes and targets every 0.5s.
  // static int64_t last_log_us = 0;
  // if (steady_now - last_log_us > 500000) {
  //   last_log_us = steady_now;
  //   std::ostringstream oss;
  //   oss << "[ctl] wheel modes/targets:";
  //   for (int i = 0; i < static_cast<int>(y1_instance::kMobilityIdx.size()); ++i) {
  //     const int idx = y1_instance::kMobilityIdx[i];
  //     const auto mode = state_->control_input.mode[idx];
  //     const double tgt = state_->control_input.target[idx];
  //     oss << " [" << i << ":" << (mode == rb::kVelocityControlMode ? 'V' : 'P')
  //         << "," << tgt << "]";
  //   }
  //   std::cout << oss.str() << std::endl;
  // }

  state_->control_input.finish = stop_requested_.load();
  return state_->control_input;
}

bool RealtimeDriver::AllComponentsReady() const {
  for (Rby1Component* component : state_->components) {
    if (!component->is_ready()) {
      return false;
    }
  }
  return true;
}

std::optional<RealtimeDriver::RobotSnapshot>
RealtimeDriver::GetLatestRobotState() const {
  std::lock_guard<std::mutex> guard(state_->snapshot_mutex);
  if (!state_->snapshot.is_valid) {
    return std::nullopt;
  }
  return state_->snapshot;
}

}  // namespace rby1::control
