#include <atomic>
#include <chrono>
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
    std::cout << "Power is OFF. Powering on..." << std::endl;
    if (!robot->PowerOn(".*")) {
      throw std::runtime_error("Failed to power on robot.");
    }
  }

  std::cout << "Checking servo status..." << std::endl;
  if (!robot->IsServoOn(".*")) {
    std::cout << "Servo is OFF. Turning on..." << std::endl;
    if (!robot->ServoOn(".*")) {
      throw std::runtime_error("Failed to enable servos.");
    }
  }

  rb::ControlManagerState manager_state = robot->GetControlManagerState();
  if (manager_state.state == rb::ControlManagerState::State::kMajorFault ||
      manager_state.state == rb::ControlManagerState::State::kMinorFault) {
    std::cout << "Resetting control manager fault..." << std::endl;
    if (!robot->ResetFaultControlManager()) {
      throw std::runtime_error("Unable to reset control manager fault.");
    }
  }

  manager_state = robot->GetControlManagerState();
  if (manager_state.state == rb::ControlManagerState::State::kEnabled) {
    std::cout << "Disabling control manager to configure gains..." << std::endl;
    if (!robot->DisableControlManager()) {
      throw std::runtime_error("Failed to disable control manager.");
    }
  }

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

  robot->SetToolFlangeOutputVoltage("left", 12);
  robot->SetToolFlangeOutputVoltage("right", 12);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = (argc > 1) ? argv[1] : "localhost:50051";
  std::cout << "Connecting to robot at " << address << " ..." << std::endl;

  auto robot = rb::Robot<y1_instance>::Create(address);
  if (!robot || !robot->Connect()) {
    std::cerr << "Failed to connect to robot." << std::endl;
    return 1;
  }
  std::cout << "Connected." << std::endl;

  try {
    InitializeRobot(robot);
  } catch (const std::exception& e) {
    std::cerr << "Initialization error: " << e.what() << std::endl;
    return 1;
  }

  rb::RobotState<y1_instance> last_state{};
  std::atomic<bool> have_state{false};
  std::atomic<bool> running{true};

  robot->StartStateUpdate(
      [&](const rb::RobotState<y1_instance>& state, const auto&) {
        if (!running.load()) {
          return;
        }
        last_state = state;
        have_state.store(true);
      },
      100);

  std::cout << "Waiting for first state sample..." << std::endl;
  for (int i = 0; i < 200 && !have_state.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!have_state.load()) {
    std::cerr << "No state samples received." << std::endl;
    running.store(false);
    robot->StopStateUpdate();
    return 1;
  }

  rb::ControlInput<y1_instance> hold_input;
  hold_input.mode.setConstant(rb::kPositionControlMode);
  hold_input.feedback_gain.setConstant(4);
  hold_input.feedforward_torque.setConstant(0);
  hold_input.target = last_state.position;
  hold_input.finish = false;

  std::atomic<bool> request_stop{false};
  std::thread control_thread([&]() {
    robot->Control(
        [&](const rb::ControlState<y1_instance>&) {
          auto cmd = hold_input;
          if (request_stop.load()) {
            cmd.finish = true;
          }
          return cmd;
        },
        0, 10);
  });

  std::cout << "Control loop running for 5 seconds with zero updates..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(5));
  request_stop.store(true);
  control_thread.join();

  running.store(false);
  robot->StopStateUpdate();
  std::cout << "Control loop test complete." << std::endl;

  return 0;
}
