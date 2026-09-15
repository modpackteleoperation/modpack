#include <csignal>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "realtime_driver.h"

namespace {

std::atomic<bool> g_stop{false};

void HandleSignal(int) {
  g_stop.store(true);
}

}  // namespace

int main(int argc, char** argv) {
  if (std::signal(SIGINT, HandleSignal) == SIG_ERR ||
      std::signal(SIGTERM, HandleSignal) == SIG_ERR) {
    std::perror("signal");
  }

  rby1::control::RealtimeDriver::Config config;
  if (argc > 1) {
    config.robot_address = argv[1];
  }

  try {
    rby1::control::RealtimeDriver driver(config);
    driver.Start();
    driver.WaitUntilReady(10.0);

    std::cout << "Realtime controller running. Press Ctrl-C to exit."
              << std::endl;
    while (!g_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    driver.Stop();
  } catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
