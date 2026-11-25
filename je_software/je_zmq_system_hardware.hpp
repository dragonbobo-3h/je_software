#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "common/msg/oculus_init_joint_state.hpp"
#include "je_software/msg/end_effector_command.hpp"
#include "je_software/msg/end_effector_command_lr.hpp"
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include "mk2_guarded_command.hpp"

namespace je_software
{

class JeZmqSystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(JeZmqSystemHardware)
  ~JeZmqSystemHardware() override;

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_error(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  struct GripperCommand
  {
    int mode{je_software::msg::EndEffectorCommand::MODE_POSITION};
    double position{0.0};
    int preset{0};
    std::string command;
    double torque{0.0};
    bool received{false};
  };

  void state_receive_loop();
  bool parse_joint_state_from_json(const nlohmann::json & state_json, std::vector<double> & positions,
                                   std::vector<double> & velocities, std::vector<double> & efforts) const;
  void robot_switch(bool value);
  void gripper_spin_loop();
  void gripper_cmd_callback(const je_software::msg::EndEffectorCommandLR::SharedPtr msg);
  void handle_gripper_cmd(const je_software::msg::EndEffectorCommand & msg, int robot_index);
  void publish_oculus_feedback_from_json(const nlohmann::json & state_json) const;
  bool map_joint_to_robot_slot(
    const std::string & joint_name,
    int & robot_index,
    std::size_t & slot_index) const;
  const GripperCommand & get_gripper_command(int robot_index) const;
  nlohmann::json make_gripper_payload(const GripperCommand & gripper_cmd) const;
  void append_gripper_from_command(nlohmann::json & payload, int robot_index) const;
  bool guarded_state_ok_locked();

  std::vector<std::string> joint_names_;
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_efforts_;
  std::vector<double> hw_commands_;
  std::vector<double> hw_velocity_commands_;

  std::unordered_map<std::string, std::size_t> joint_name_to_index_;

  std::mutex state_mutex_;
  std::vector<double> latest_positions_;
  std::vector<double> latest_velocities_;
  std::vector<double> latest_efforts_;
  std::atomic_bool has_valid_state_{false};

  // Disabled for all existing arms unless explicitly enabled by MK2 description.
  bool mk2_guarded_control_{false};
  bool mk2_shared_ownership_{false};
  std::atomic_bool mk2_session_active_{false};
  std::string mk2_expected_backend_{"sdk_simulation"};
  std::string mk2_device_config_sha256_;
  double mk2_feedback_timeout_{0.15};
  double mk2_max_velocity_{Mk2GuardedCommand::default_max_velocity};
  double mk2_max_acceleration_{Mk2GuardedCommand::default_max_acceleration};
  std::unique_ptr<Mk2GuardedCommand> mk2_command_;
  Mk2GuardedCommand::Axes mk2_lower_{}, mk2_upper_{};
  // Per-axis Lead-Lag compensator for physical MK2 CSP drives:
  // C_i(s) = (1 + tau_p[i]*s) / (1 + tau_target*s) aligns all 7 joints to
  // identical first-order response tau_target without raising drive closed-loop gains.
  bool mk2_lead_lag_enabled_{false};
  double mk2_lead_lag_tau_target_{0.095};
  double mk2_lead_lag_max_lead_{0.12};
  Mk2GuardedCommand::Axes mk2_lead_lag_tau_p_{0.128, 0.128, 0.128, 0.156, 0.068, 0.070, 0.068};
  Mk2GuardedCommand::Axes mk2_lead_lag_state_{};
  bool mk2_lead_lag_initialized_{false};
  std::condition_variable state_condition_;
  std::chrono::steady_clock::time_point mk2_last_feedback_{};
  uint64_t mk2_last_sequence_{0};
  bool mk2_seen_feedback_{false};
  bool mk2_local_fault_{false};
  bool mk2_fault_reported_{false};
  bool mk2_switch_sent_{false};
  bool mk2_clean_release_supported_{false};
  bool mk2_handover_supported_{false};
  bool mk2_releasing_{false};
  std::optional<rclcpp::Time> mk2_last_write_time_;
  std::optional<double> mk2_first_mono_raw_sec_;
  std::optional<double> mk2_last_mono_raw_sec_;
  std::string mk2_fault_reason_, mk2_phase_;

  std::string robot_ip_{"192.168.0.99"};
  int pub_port_{8001};
  int sub_port_{8000};
  // MK2 only: gripper commands go on their own PUB port (the arm plugin rejects them).
  int mk2_gripper_port_{0};
  int mk2_gripper_feedback_port_{0};
  double command_dt_{0.02};
  double global_time_{0.0};
  std::string controlled_arm_{"both"};

  std::atomic_bool state_thread_running_{false};
  std::thread state_thread_;

  mutable std::mutex gripper_mutex_;
  std::string gripper_sub_topic_{"/end_effector_cmd_lr"};
  GripperCommand gripper_cmd_left_{};
  GripperCommand gripper_cmd_right_{};
  std::shared_ptr<rclcpp::Node> gripper_node_;
  rclcpp::Subscription<je_software::msg::EndEffectorCommandLR>::SharedPtr sub_gripper_cmd_;
  rclcpp::Publisher<common::msg::OculusInitJointState>::SharedPtr pub_oculus_feedback_;
  std::unique_ptr<zmq::socket_t> gripper_feedback_subscriber_;
  mutable std::mutex gripper_feedback_mutex_;
  float left_gripper_feedback_{0.0F};
  float right_gripper_feedback_{0.0F};
  bool left_gripper_feedback_valid_{false};
  bool right_gripper_feedback_valid_{false};
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> gripper_executor_;
  std::atomic_bool gripper_thread_running_{false};
  std::thread gripper_thread_;

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t> publisher_;
  std::unique_ptr<zmq::socket_t> subscriber_;
  // Used only from the gripper executor thread.
  std::unique_ptr<zmq::socket_t> gripper_publisher_;
};

}  // namespace je_software
