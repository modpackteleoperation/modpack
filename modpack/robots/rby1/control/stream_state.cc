#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "rby1-sdk/model.h"
#include "rby1-sdk/net/real_time_control_protocol.h"
#include "rby1-sdk/robot.h"

using y1_instance = rb::y1_model::M;

namespace {

void InitializeRobot(const std::shared_ptr<rb::Robot<y1_instance>>& robot) {
  std::cout << "Checking power status..." << std::endl;
  if (!robot->IsPowerOn(".*")) {
    std::cout << "Power is currently OFF. Attempting to power on..." << std::endl;
    if (!robot->PowerOn(".*")) {
      throw std::runtime_error("Failed to power on the robot.");
    }
    std::cout << "Robot powered on successfully." << std::endl;
  } else {
    std::cout << "Power is already ON." << std::endl;
  }

  std::cout << "Checking servo status..." << std::endl;
  if (!robot->IsServoOn(".*")) {
    std::cout << "Servo is currently OFF. Attempting to activate servo..." << std::endl;
    if (!robot->ServoOn(".*")) {
      throw std::runtime_error("Failed to activate servo.");
    }
    std::cout << "Servo activated successfully." << std::endl;
  } else {
    std::cout << "Servo is already ON." << std::endl;
  }

  rb::ControlManagerState control_manager_state = robot->GetControlManagerState();
  if (control_manager_state.state == rb::ControlManagerState::State::kMajorFault ||
      control_manager_state.state == rb::ControlManagerState::State::kMinorFault) {
    std::cout << "Detected control manager fault. Attempting reset..." << std::endl;
    if (!robot->ResetFaultControlManager()) {
      throw std::runtime_error("Unable to reset control manager fault.");
    }
    std::cout << "Fault reset successfully." << std::endl;
  }

  control_manager_state = robot->GetControlManagerState();
  if (control_manager_state.state == rb::ControlManagerState::State::kEnabled) {
    std::cout << "Disabling control manager to configure gains..." << std::endl;
    if (!robot->DisableControlManager()) {
      throw std::runtime_error("Unable to disable control manager.");
    }
  }

  std::cout << "Configuring joint gains..." << std::endl;
  robot->SetPositionPIDGain("torso_0", 100, 1, 500);
  robot->SetPositionPIDGain("torso_1", 100, 1, 400);
  robot->SetPositionPIDGain("torso_2", 250, 1, 400);
  robot->SetPositionPIDGain("torso_3", 500, 40, 400);
  robot->SetPositionPIDGain("torso_4", 500, 20, 400);
  robot->SetPositionPIDGain("torso_5", 500, 40, 400);
  robot->SetPositionPGain("head_0", 150);
  robot->SetPositionDGain("head_0", 750);

  std::cout << "Enabling control manager..." << std::endl;
  if (!robot->EnableControlManager()) {
    throw std::runtime_error("Failed to enable control manager.");
  }

//   std::cout << "Setting tool flange voltages..." << std::endl;
//   if (!robot->SetToolFlangeOutputVoltage("left", 12)) {
//     throw std::runtime_error("Failed to set left tool flange voltage.");
//   }
//   if (!robot->SetToolFlangeOutputVoltage("right", 12)) {
//     throw std::runtime_error("Failed to set right tool flange voltage.");
//   }
//   std::cout << "Initialization complete." << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = (argc > 1) ? argv[1] : "localhost:50051";
  std::cout << "Connecting to robot at " << address << " ..." << std::endl;

  auto robot = rb::Robot<y1_instance>::Create(address);
  if (!robot) {
    std::cerr << "Failed to create robot instance." << std::endl;
    return 1;
  }
  if (!robot->Connect()) {
    std::cerr << "Unable to establish connection to the robot." << std::endl;
    return 1;
  }
  std::cout << "Successfully connected." << std::endl;

  try {
    InitializeRobot(robot);
  } catch (const std::exception& e) {
    std::cerr << "Initialization error: " << e.what() << std::endl;
    return 1;
  }

  std::atomic<bool> running{true};
  robot->StartStateUpdate(
      [&running](const rb::RobotState<y1_instance>& state, const auto& manager_state) {
        if (!running.load()) {
          return;
        }
        const auto& timestamp = state.timestamp;
        std::cout << "State timestamp: " << timestamp.tv_sec << "."
                  << std::setw(9) << std::setfill('0') << timestamp.tv_nsec << std::endl;
        std::cout << "  Control manager: " << rb::to_string(manager_state.state) << std::endl;
        if (state.position.size() > 0) {
          std::cout << "  First joint position: " << state.position[0] << std::endl;
        }
      },
      100);

  std::cout << "Streaming state for 5 seconds..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(5));
  running.store(false);
  robot->StopStateUpdate();
  std::cout << "State streaming finished." << std::endl;

  return 0;
}
