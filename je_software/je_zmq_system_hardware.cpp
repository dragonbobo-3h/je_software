#include "je_zmq_system_hardware.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <string>
#include <thread>
#include <utility>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace je_software
{

using json = nlohmann::json;
using namespace std::chrono_literals;

namespace
{
double monotonic_raw_seconds()
{
  struct timespec ts {};
#ifdef CLOCK_MONOTONIC_RAW
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
  }
#endif
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}
}  // namespace

JeZmqSystemHardware::~JeZmqSystemHardware()
{
  if (mk2_guarded_control_) on_deactivate(rclcpp_lifecycle::State());
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_shutdown(
  const rclcpp_lifecycle::State & state)
{
  return mk2_guarded_control_ ? on_deactivate(state) : hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_error(
  const rclcpp_lifecycle::State & state)
{
  if (mk2_guarded_control_) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    mk2_local_fault_ = true;  // Lifecycle errors must NEVER take the clean-release path.
    if (mk2_fault_reason_.empty()) mk2_fault_reason_ = "hardware lifecycle error";
  }
  return mk2_guarded_control_ ? on_deactivate(state) : hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  const auto guarded = info_.hardware_parameters.find("mk2_guarded_control");
  mk2_guarded_control_ = guarded != info_.hardware_parameters.end() && guarded->second == "true";
  const auto ownership = info_.hardware_parameters.find("mk2_shared_ownership");
  mk2_shared_ownership_ = mk2_guarded_control_ &&
    ownership != info_.hardware_parameters.end() && ownership->second == "true";
  joint_names_.clear();
  joint_names_.reserve(info_.joints.size());
  joint_name_to_index_.clear();

  for (const auto & joint : info_.joints)
  {
    if (joint.command_interfaces.size() != (mk2_guarded_control_ ? 2u : 1u) ||
      joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION ||
      (mk2_guarded_control_ && joint.command_interfaces[1].name != hardware_interface::HW_IF_VELOCITY))
    {
      RCLCPP_FATAL(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Joint '%s' requires command interfaces [%s] in order; rebuild matching MK2 description/controller",
        joint.name.c_str(), mk2_guarded_control_ ? "position, velocity" : "position");
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.state_interfaces.empty())
    {
      RCLCPP_FATAL(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Joint '%s' must expose at least one state interface",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }

    joint_name_to_index_[joint.name] = joint_names_.size();
    joint_names_.push_back(joint.name);
  }

  const auto n = joint_names_.size();
  hw_positions_.assign(n, std::numeric_limits<double>::quiet_NaN());
  hw_velocities_.assign(n, 0.0);
  hw_efforts_.assign(n, 0.0);
  hw_commands_.assign(n, std::numeric_limits<double>::quiet_NaN());
  hw_velocity_commands_.assign(n, 0.0);

  latest_positions_.assign(n, 0.0);
  latest_velocities_.assign(n, 0.0);
  latest_efforts_.assign(n, 0.0);

  auto get_string_param = [&](const std::string & key, std::string & target)
  {
    auto it = info_.hardware_parameters.find(key);
    if (it != info_.hardware_parameters.end())
    {
      target = it->second;
    }
  };

  auto get_int_param = [&](const std::string & key, int & target)
  {
    auto it = info_.hardware_parameters.find(key);
    if (it != info_.hardware_parameters.end())
    {
      try
      {
        target = std::stoi(it->second);
      }
      catch (...) {}
    }
  };

  auto get_double_param = [&](const std::string & key, double & target)
  {
    auto it = info_.hardware_parameters.find(key);
    if (it != info_.hardware_parameters.end())
    {
      try
      {
        target = std::stod(it->second);
      }
      catch (...) {}
    }
  };

  get_string_param("robot_ip", robot_ip_);
  get_int_param("pub_port", pub_port_);
  get_int_param("sub_port", sub_port_);
  get_int_param("mk2_gripper_port", mk2_gripper_port_);
  get_int_param("mk2_gripper_feedback_port", mk2_gripper_feedback_port_);
  get_double_param("command_dt", command_dt_);
  get_string_param("gripper_sub_topic", gripper_sub_topic_);
  get_double_param("mk2_feedback_timeout", mk2_feedback_timeout_);
  get_string_param("mk2_expected_backend", mk2_expected_backend_);
  get_string_param("mk2_device_config_sha256", mk2_device_config_sha256_);
  get_double_param("mk2_max_velocity", mk2_max_velocity_);
  get_double_param("mk2_max_acceleration", mk2_max_acceleration_);

  if (const char * controlled_arm = std::getenv("JE_ARM_CONTROLLED_ARM"))
  {
    controlled_arm_ = controlled_arm;
  }
  if (controlled_arm_ != "both" && controlled_arm_ != "left" && controlled_arm_ != "right")
  {
    RCLCPP_WARN(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Invalid JE_ARM_CONTROLLED_ARM='%s'; falling back to 'both'", controlled_arm_.c_str());
    controlled_arm_ = "both";
  }

  if (mk2_guarded_control_) {
    const auto physical_confirmation = info_.hardware_parameters.find("mk2_confirm_physical_control");
    if ((mk2_expected_backend_ != "sdk_simulation" && mk2_expected_backend_ != "physical") ||
        (mk2_expected_backend_ == "physical" &&
         (physical_confirmation == info_.hardware_parameters.end() || physical_confirmation->second != "true" ||
          mk2_device_config_sha256_.size() != 64 ||
          mk2_device_config_sha256_.find_first_not_of("0123456789abcdef") != std::string::npos))) {
      RCLCPP_ERROR(rclcpp::get_logger("JeZmqSystemHardware"),
        "MK2 physical backend requires separate confirmation and verified DeviceConfig SHA-256");
      return hardware_interface::CallbackReturn::ERROR;
    }
    if (joint_names_.size() != 7 || controlled_arm_ != "left" ||
        !std::isfinite(command_dt_) || command_dt_ < 0.005 || command_dt_ > 0.05 ||
        !std::isfinite(mk2_feedback_timeout_) || mk2_feedback_timeout_ <= 0.0 ||
        mk2_feedback_timeout_ > 0.2 ||
        (mk2_gripper_port_ != 0 && (mk2_gripper_port_ < 1024 || mk2_gripper_port_ > 65535 ||
                                    mk2_gripper_port_ == pub_port_ || mk2_gripper_port_ == sub_port_)) ||
        (mk2_gripper_feedback_port_ != 0 && (mk2_gripper_feedback_port_ < 1024 ||
                                    mk2_gripper_feedback_port_ > 65534 ||
                                    mk2_gripper_feedback_port_ + 1 == pub_port_ ||
                                    mk2_gripper_feedback_port_ + 1 == sub_port_))) {
      RCLCPP_ERROR(rclcpp::get_logger("JeZmqSystemHardware"), "Invalid MK2 guarded transport settings");
      return hardware_interface::CallbackReturn::ERROR;
    }
    // Physical caps are bounded by physical_max_*; only isolated SDK simulation
    // may go further (native builds still enforce their own policy caps).
    const bool caps_physical = mk2_max_velocity_ <= Mk2GuardedCommand::physical_max_velocity &&
      mk2_max_acceleration_ <= Mk2GuardedCommand::physical_max_acceleration;
    if (!std::isfinite(mk2_max_velocity_) || !std::isfinite(mk2_max_acceleration_) ||
        mk2_max_velocity_ <= 0.0 || mk2_max_acceleration_ <= 0.0 ||
        mk2_max_velocity_ > 4.0 || mk2_max_acceleration_ > 40.0 ||
        (mk2_expected_backend_ != "sdk_simulation" && !caps_physical)) {
      RCLCPP_ERROR(rclcpp::get_logger("JeZmqSystemHardware"),
        "Invalid MK2 caps v=%.3f a=%.3f for backend %s (physical max v=%.1f a=%.1f)",
        mk2_max_velocity_, mk2_max_acceleration_, mk2_expected_backend_.c_str(),
        Mk2GuardedCommand::physical_max_velocity, Mk2GuardedCommand::physical_max_acceleration);
      return hardware_interface::CallbackReturn::ERROR;
    }
    try {
      for (size_t i = 0; i < 7; ++i) {
        if (joint_names_[i] != "joint" + std::to_string(i + 1)) throw std::runtime_error("joint order");
        const auto & limits = info_.joints[i].command_interfaces[0];
        mk2_lower_[i] = std::stod(limits.min);
        mk2_upper_[i] = std::stod(limits.max);
        if (!std::isfinite(mk2_lower_[i]) || !std::isfinite(mk2_upper_[i]) ||
            mk2_lower_[i] >= mk2_upper_[i]) throw std::runtime_error("joint bounds");
      }
      mk2_command_ = std::make_unique<Mk2GuardedCommand>(
        command_dt_, mk2_max_velocity_, mk2_max_acceleration_);
      mk2_lead_lag_enabled_ = (mk2_expected_backend_ == "physical");
      if (const char * env_enable = std::getenv("JE_MK2_LEAD_LAG_ENABLE")) {
        const std::string val(env_enable);
        mk2_lead_lag_enabled_ = (val != "0" && val != "false" && val != "FALSE");
      }
      if (const char * env_tau = std::getenv("JE_MK2_LEAD_LAG_TAU_TARGET")) {
        try {
          const double parsed = std::stod(env_tau);
          if (std::isfinite(parsed) && parsed >= 0.060 && parsed <= 0.160) {
            mk2_lead_lag_tau_target_ = parsed;
          }
        } catch (...) {}
      }
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(rclcpp::get_logger("JeZmqSystemHardware"), "Invalid MK2 limits: %s", ex.what());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "Configured JE hardware transport: robot_ip=%s, pub_port=%d, sub_port=%d, command_dt=%.4f, gripper_sub_topic=%s, controlled_arm=%s",
    robot_ip_.c_str(), pub_port_, sub_port_, command_dt_, gripper_sub_topic_.c_str(),
    controlled_arm_.c_str());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_configure(
  const rclcpp_lifecycle::State &)
{
  std::fill(hw_positions_.begin(), hw_positions_.end(), 0.0);
  std::fill(hw_velocities_.begin(), hw_velocities_.end(), 0.0);
  std::fill(hw_efforts_.begin(), hw_efforts_.end(), 0.0);
  std::fill(hw_commands_.begin(), hw_commands_.end(), 0.0);
  std::fill(hw_velocity_commands_.begin(), hw_velocity_commands_.end(), 0.0);

  std::fill(latest_positions_.begin(), latest_positions_.end(), 0.0);
  std::fill(latest_velocities_.begin(), latest_velocities_.end(), 0.0);
  std::fill(latest_efforts_.begin(), latest_efforts_.end(), 0.0);
  has_valid_state_ = false;

  try
  {
    context_ = std::make_unique<zmq::context_t>(1);
    publisher_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::pub);
    subscriber_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::sub);

    // Hardware plugin as publisher: binds on pub_port to send commands
    // Hardware plugin as subscriber: connects to robot's state port to receive state
    const std::string pub_bind_addr = "tcp://*:" + std::to_string(pub_port_);
    const std::string sub_connect_addr = "tcp://" + robot_ip_ + ":" + std::to_string(sub_port_);

    publisher_->set(zmq::sockopt::sndhwm, mk2_guarded_control_ ? 4 : 0);
    if (mk2_guarded_control_) publisher_->set(zmq::sockopt::linger, 0);
    publisher_->set(zmq::sockopt::immediate, 1);
    publisher_->bind(pub_bind_addr);

    subscriber_->set(zmq::sockopt::subscribe, "State ");
    subscriber_->set(zmq::sockopt::conflate, 1);
    subscriber_->connect(sub_connect_addr);

    if (mk2_guarded_control_ && mk2_gripper_port_ != 0) {
      gripper_publisher_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::pub);
      gripper_publisher_->set(zmq::sockopt::sndhwm, 16);
      gripper_publisher_->set(zmq::sockopt::linger, 0);
      gripper_publisher_->bind("tcp://*:" + std::to_string(mk2_gripper_port_));
    }

    if (mk2_guarded_control_ && mk2_gripper_feedback_port_ != 0) {
      gripper_feedback_subscriber_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::sub);
      gripper_feedback_subscriber_->set(zmq::sockopt::subscribe, "GripperFeedback ");
      gripper_feedback_subscriber_->set(zmq::sockopt::conflate, 1);
      gripper_feedback_subscriber_->connect("tcp://" + robot_ip_ + ":" + std::to_string(mk2_gripper_feedback_port_));
      gripper_feedback_subscriber_->connect("tcp://" + robot_ip_ + ":" + std::to_string(mk2_gripper_feedback_port_ + 1));
    }

    RCLCPP_INFO(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "ZMQ configured: pub_bind=%s, sub_connect=%s, mk2_gripper_port=%d",
      pub_bind_addr.c_str(), sub_connect_addr.c_str(), gripper_publisher_ ? mk2_gripper_port_ : 0);
  }
  catch (const std::exception & ex)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to configure ZMQ transport: %s", ex.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_activate(
  const rclcpp_lifecycle::State & previous_state)
{
  mk2_session_active_.store(false);
  // Deactivation closes the old transport. Never reuse its samples or queues.
  if (mk2_shared_ownership_ && !context_ &&
      on_configure(previous_state) != hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  global_time_ = 0.0;
  mk2_last_write_time_.reset();
  mk2_first_mono_raw_sec_.reset();
  mk2_last_mono_raw_sec_.reset();
  mk2_switch_sent_ = false;
  if (mk2_guarded_control_) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    has_valid_state_ = false;
    mk2_seen_feedback_ = false;
    mk2_last_sequence_ = 0;
    mk2_local_fault_ = false;
    mk2_fault_reported_ = false;
    mk2_clean_release_supported_ = false;
    mk2_handover_supported_ = false;
    mk2_releasing_ = false;
    mk2_fault_reason_.clear();
    mk2_phase_.clear();
  }

  try
  {
    gripper_node_ = std::make_shared<rclcpp::Node>("je_zmq_system_hardware_gripper_bridge");
    sub_gripper_cmd_ = gripper_node_->create_subscription<je_software::msg::EndEffectorCommandLR>(
      gripper_sub_topic_,
      rclcpp::QoS(10).reliable(),
      std::bind(&JeZmqSystemHardware::gripper_cmd_callback, this, std::placeholders::_1));
    pub_oculus_feedback_ = gripper_node_->create_publisher<common::msg::OculusInitJointState>(
      "/joint_states_double_arm", rclcpp::QoS(10).reliable());

    gripper_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    gripper_executor_->add_node(gripper_node_);
    gripper_thread_running_.store(true);
    gripper_thread_ = std::thread(&JeZmqSystemHardware::gripper_spin_loop, this);

    RCLCPP_INFO(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Subscribed gripper command topic: %s",
      gripper_sub_topic_.c_str());
  }
  catch (const std::exception & ex)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to initialize gripper subscriber: %s", ex.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Initialize ZMQ connection before starting state thread
  // Add a small delay to allow ZMQ sockets to fully bind/connect
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  if (mk2_guarded_control_) {
    state_thread_running_.store(true);
    state_thread_ = std::thread(&JeZmqSystemHardware::state_receive_loop, this);
    bool ready = false;
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      state_condition_.wait_for(lock, 2s, [this] { return mk2_seen_feedback_ || mk2_local_fault_; });
      ready = guarded_state_ok_locked() && mk2_phase_ == "ready" &&
        (!mk2_shared_ownership_ || mk2_handover_supported_);
    }
    if (!ready) {
      RCLCPP_ERROR(rclcpp::get_logger("JeZmqSystemHardware"), "MK2 native bridge not ready; no Switch=true sent");
      on_deactivate(previous_state);
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  try
  {
    robot_switch(true);
    mk2_switch_sent_ = true;
  }
  catch (const std::exception & ex)
  {
    RCLCPP_WARN(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to send switch command on activate: %s", ex.what());
  }

  if (mk2_guarded_control_) {
    bool acknowledged = false;
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      state_condition_.wait_for(lock, 500ms, [this] {
        return mk2_phase_ == "waiting_first_joint" || mk2_local_fault_;
      });
      acknowledged = guarded_state_ok_locked() && mk2_phase_ == "waiting_first_joint";
    }
    if (!acknowledged) {
      RCLCPP_ERROR(rclcpp::get_logger("JeZmqSystemHardware"), "MK2 Switch acknowledgement failed");
      on_deactivate(previous_state);
      return hardware_interface::CallbackReturn::ERROR;
    }
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    state_thread_running_.store(true);
    state_thread_ = std::thread(&JeZmqSystemHardware::state_receive_loop, this);
  }

  // Wait for first valid state before considering hardware ready
  // Increased timeout to 15 seconds to allow robot backend to establish connection
  int wait_count = 0;
  int max_wait = 1500;  // 15 seconds at 10ms intervals
  while (!has_valid_state_ && wait_count < max_wait)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    wait_count++;
    
    // Log progress every 100 iterations (1 second)
    if (wait_count % 100 == 0)
    {
      RCLCPP_DEBUG(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Waiting for initial state from ZMQ hardware... (%d/%d seconds)",
        wait_count / 100, max_wait / 100);
    }
  }

  if (!has_valid_state_)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Timeout waiting for initial state from ZMQ hardware after %d seconds",
      max_wait / 100);
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Now initialize hw_positions and hw_commands with the actual state
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    hw_positions_ = latest_positions_;
  }
  hw_commands_ = hw_positions_;
  std::fill(hw_velocity_commands_.begin(), hw_velocity_commands_.end(), 0.0);
  if (mk2_guarded_control_) {
    Mk2GuardedCommand::Axes initial;
    std::copy_n(hw_positions_.begin(), 7, initial.begin());
    mk2_command_->reset(initial, mk2_lower_, mk2_upper_);
    mk2_lead_lag_state_ = initial;
    mk2_lead_lag_initialized_ = true;
    RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"),
      "MK2 timed position+velocity passthrough enabled; backend=%s caps v<=%.3f a<=%.3f; "
      "lead_lag=%s tau_target=%.3fs max_lead=%.3frad",
      mk2_expected_backend_.c_str(), mk2_command_->max_velocity(), mk2_command_->max_acceleration(),
      mk2_lead_lag_enabled_ ? "on" : "off", mk2_lead_lag_tau_target_, mk2_lead_lag_max_lead_);
  }

  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "Hardware activated. Initial positions: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
    hw_positions_[0], hw_positions_[1], hw_positions_[2], hw_positions_[3],
    hw_positions_[4], hw_positions_[5], hw_positions_[6]);

  mk2_session_active_.store(true);
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  mk2_session_active_.store(false);
  bool handover_ok = !mk2_switch_sent_;
  if (mk2_guarded_control_ && mk2_switch_sent_) {
    // Explicit clean release is capability-negotiated and only for a healthy
    // local session. Faults/old plugins retain the original hard-stop protocol.
    bool release = false;
    uint64_t before_sequence = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      release = (mk2_expected_backend_ == "sdk_simulation" || mk2_shared_ownership_) &&
        mk2_clean_release_supported_ && !mk2_local_fault_ && mk2_seen_feedback_ &&
        (mk2_phase_ == "active" || mk2_phase_ == "waiting_first_joint") &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() - mk2_last_feedback_).count() <=
          mk2_feedback_timeout_;
      before_sequence = mk2_last_sequence_;
      mk2_releasing_ = release;
    }
    try {
      if (release) {
        const std::string frame = "Release {\"Release\":true}";
        publisher_->send(zmq::buffer(frame), zmq::send_flags::dontwait);
      } else robot_switch(false);
      std::unique_lock<std::mutex> lock(state_mutex_);
      const bool acknowledged = state_condition_.wait_for(lock, release ? 2500ms : 200ms,
        [this, release, before_sequence] {
        return mk2_phase_ == "stopped" || mk2_phase_ == "fault" ||
          (release && mk2_phase_ == "ready" && mk2_last_sequence_ > before_sequence);
      });
      if (release && acknowledged && mk2_phase_ == "ready") {
        handover_ok = true;
        RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"),
          "MK2 clean release observed: native ready; next launch still requires fresh preflight");
      }
      if (!acknowledged) RCLCPP_WARN(rclcpp::get_logger("JeZmqSystemHardware"),
        "MK2 stop/release not acknowledged; do not assume SDK stop or restart readiness");
    } catch (const std::exception & ex) {
      RCLCPP_WARN(rclcpp::get_logger("JeZmqSystemHardware"), "MK2 stop send failed: %s", ex.what());
    }
    mk2_switch_sent_ = false;
  }
  gripper_thread_running_.store(false);
  if (gripper_executor_)
  {
    gripper_executor_->cancel();
  }
  if (gripper_thread_.joinable())
  {
    gripper_thread_.join();
  }
  if (gripper_executor_ && gripper_node_)
  {
    gripper_executor_->remove_node(gripper_node_);
  }
  state_thread_running_.store(false);

  if (context_)
  {
    try
    {
      context_->shutdown();
    }
    catch (...) {}
  }

  if (state_thread_.joinable())
  {
    state_thread_.join();
  }

  // The receiver also uses this publisher: release it only after joining.
  sub_gripper_cmd_.reset();
  pub_oculus_feedback_.reset();
  gripper_executor_.reset();
  gripper_node_.reset();

  try
  {
    if (!mk2_guarded_control_) robot_switch(false);
  }
  catch (...) {}

  subscriber_.reset();
  publisher_.reset();
  gripper_publisher_.reset();
  gripper_feedback_subscriber_.reset();
  context_.reset();

  if (mk2_shared_ownership_ && !handover_ok) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> JeZmqSystemHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(joint_names_.size() * 3);

  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]);
  }

  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "[export_state_interfaces] Exported %zu state interfaces for %zu joints",
    state_interfaces.size(), joint_names_.size());

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> JeZmqSystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(joint_names_.size() * (mk2_guarded_control_ ? 2 : 1));

  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    command_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
    if (mk2_guarded_control_) {
      command_interfaces.emplace_back(
        joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_velocity_commands_[i]);
    }
  }

  return command_interfaces;
}

hardware_interface::return_type JeZmqSystemHardware::read(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  // Shared standby has no controller claims and no PC command session. Its
  // diagnostic interfaces are NOT the global measured /joint_states source.
  if (mk2_shared_ownership_ && !mk2_session_active_.load()) {
    return hardware_interface::return_type::OK;
  }
  if (mk2_guarded_control_ && !guarded_state_ok_locked()) {
    return hardware_interface::return_type::ERROR;
  }
  static int read_count = 0;
  static bool first_valid = true;
  
  if (has_valid_state_)
  {
    // Copy from ZMQ state to hardware interfaces
    for (size_t i = 0; i < hw_positions_.size(); ++i)
    {
      hw_positions_[i] = latest_positions_[i];
      hw_velocities_[i] = latest_velocities_[i];
      hw_efforts_[i] = latest_efforts_[i];
    }
    
    if (first_valid)
    {
      RCLCPP_INFO(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[read] First valid state received - hardware interfaces active");
      first_valid = false;
    }
    
    // Log less frequently (every 2000 reads ≈ 20 seconds at 100Hz)
    if (++read_count % 2000 == 0)
    {
      RCLCPP_DEBUG(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[read] Read #%d: positions=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
        read_count,
        hw_positions_[0], hw_positions_[1], hw_positions_[2], hw_positions_[3],
        hw_positions_[4], hw_positions_[5], hw_positions_[6]);
    }
  }
  else
  {
    static bool warned = false;
    if (!warned)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[read] Waiting for valid ZMQ state from hardware...");
      warned = true;
    }
  }
  
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type JeZmqSystemHardware::write(
  const rclcpp::Time & time,
  const rclcpp::Duration & period)
{
  if (mk2_shared_ownership_ && !mk2_session_active_.load()) {
    return hardware_interface::return_type::OK;
  }
  if (!publisher_)
  {
    return hardware_interface::return_type::ERROR;
  }

  std::vector<double> commands = hw_commands_;
  double sample_dt = command_dt_;
  if (mk2_guarded_control_) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (mk2_releasing_) return hardware_interface::return_type::ERROR;
      if (!guarded_state_ok_locked()) return hardware_interface::return_type::ERROR;
    }
    if (mk2_last_write_time_) {
      if (time.get_clock_type() != mk2_last_write_time_->get_clock_type()) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        mk2_local_fault_ = true;
        mk2_fault_reason_ = "trajectory command time source changed";
        return hardware_interface::return_type::ERROR;
      }
      sample_dt = (time - *mk2_last_write_time_).seconds();
    }
    Mk2GuardedCommand::Axes target, target_velocity, bounded;
    std::copy_n(hw_commands_.begin(), 7, target.begin());
    std::copy_n(hw_velocity_commands_.begin(), 7, target_velocity.begin());
    std::string error;
    const bool was_smoothing = mk2_command_->smoothing();
    const bool was_blending = mk2_command_->blending();
    if (!mk2_command_->step(target, target_velocity, bounded, error, sample_dt)) {
      // Record the actual requested sample and both timing inputs BEFORE the
      // existing fail-closed path. Do not silently substitute period for dt:
      // JTC samples at update(time), which is not the later write(time).
      json sample;
      sample["requested_q"] = target;
      sample["requested_velocity"] = target_velocity;
      sample["sample_dt"] = sample_dt;
      sample["manager_period"] = period.seconds();
      sample["write_time_ns"] = time.nanoseconds();
      sample["last_write_time_ns"] = mk2_last_write_time_ ?
        mk2_last_write_time_->nanoseconds() : int64_t{0};
      sample["native_time_before_sample"] = global_time_;
      error += "; rejected_sample=" + sample.dump();
      std::lock_guard<std::mutex> lock(state_mutex_);
      mk2_local_fault_ = true;
      mk2_fault_reason_ = error;
      guarded_state_ok_locked();
      return hardware_interface::return_type::ERROR;
    }
    mk2_last_write_time_ = time;
    if (!was_smoothing && mk2_command_->smoothing()) {
      RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"),
        "MK2 zero-velocity hold discontinuity: bounded transition, not instantaneous hold");
    }
    if (mk2_lead_lag_enabled_) {
      if (!mk2_lead_lag_initialized_) {
        mk2_lead_lag_state_ = bounded;
        mk2_lead_lag_initialized_ = true;
      } else {
        const double alpha_f = 1.0 - std::exp(-sample_dt / mk2_lead_lag_tau_target_);
        for (size_t i = 0; i < 7; ++i) {
          mk2_lead_lag_state_[i] += alpha_f * (bounded[i] - mk2_lead_lag_state_[i]);
          const double k_lead = (mk2_lead_lag_tau_p_[i] / mk2_lead_lag_tau_target_) - 1.0;
          const double lead = std::clamp(
            k_lead * (bounded[i] - mk2_lead_lag_state_[i]),
            -mk2_lead_lag_max_lead_, mk2_lead_lag_max_lead_);
          bounded[i] = std::clamp(bounded[i] + lead, mk2_lower_[i], mk2_upper_[i]);
        }
      }
    }
    if (!was_blending && mk2_command_->blending()) {
      RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"),
        "MK2 settled start offset (<=0.02 rad): fading into trajectory, caps unchanged");
    }
    commands.assign(bounded.begin(), bounded.end());
  }

  if (mk2_guarded_control_) {
    // Use CLOCK_MONOTONIC_RAW anchored at the first sent frame so NTP slewing/stepping
    // on CLOCK_REALTIME cannot drift global_time_ past the bridge's clock_skew window.
    const double now_mono = monotonic_raw_seconds();
    if (!mk2_first_mono_raw_sec_) {
      mk2_first_mono_raw_sec_ = now_mono;
      mk2_last_mono_raw_sec_ = now_mono;
      global_time_ = command_dt_;
    } else {
      const double mono_dt = now_mono - *mk2_last_mono_raw_sec_;
      if (mono_dt > 0.025) {
        RCLCPP_WARN(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "MK2 write monotonic interval jitter: mono_dt=%.4fs sample_dt=%.4fs",
          mono_dt, sample_dt);
      }
      mk2_last_mono_raw_sec_ = now_mono;
      const double target_time = command_dt_ + (now_mono - *mk2_first_mono_raw_sec_);
      // Preserve bridge min_dt (5 ms) even if an update() stall is followed by a fast cycle.
      global_time_ = std::max(global_time_ + 0.0051, target_time);
    }
  } else {
    // Keep the native data clock aligned with the actual JTC sample clock.
    // Otherwise scheduling jitter is misinterpreted as target acceleration.
    global_time_ += sample_dt;
  }

  json payload;
  std::vector<double> robot0_joint(7, 0.0);
  std::vector<double> robot1_joint(7, 0.0);
  bool has_robot0 = false;
  bool has_robot1 = false;

  for (std::size_t i = 0; i < hw_commands_.size() && i < joint_names_.size(); ++i)
  {
    int robot_index = 0;
    std::size_t slot_index = 0;
    if (!map_joint_to_robot_slot(joint_names_[i], robot_index, slot_index))
    {
      if (i < 7)
      {
        robot_index = 0;
        slot_index = i;
      }
      else
      {
        robot_index = 1;
        slot_index = i - 7;
      }
    }

    if (slot_index >= 7)
    {
      continue;
    }

    if (robot_index == 1)
    {
      robot1_joint[slot_index] = commands[i];
      has_robot1 = true;
    }
    else
    {
      robot0_joint[slot_index] = commands[i];
      has_robot0 = true;
    }
  }

  if (has_robot0 && controlled_arm_ != "right")
  {
    payload["Robot0"]["time"] = global_time_;
    payload["Robot0"]["joint"] = robot0_joint;
    if (!mk2_guarded_control_) append_gripper_from_command(payload, 0);
  }

  if (has_robot1 && controlled_arm_ != "left")
  {
    payload["Robot1"]["time"] = global_time_;
    payload["Robot1"]["joint"] = robot1_joint;
    if (!mk2_guarded_control_) append_gripper_from_command(payload, 1);
  }

  try
  {
    // Match je_robot_node.cpp format: prefix "Joint " + JSON payload
    const std::string message = "Joint " + payload.dump();
    // Send with default blocking behavior to ensure delivery
    auto result = publisher_->send(zmq::buffer(message));
    if (!result)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[write] Failed to send message (result is false)");
      if (mk2_guarded_control_) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        mk2_local_fault_ = true;
        mk2_fault_reason_ = "joint command transport send failed";
        return hardware_interface::return_type::ERROR;
      }
    }
    
    // Log sent commands less frequently (every 500 writes to avoid log spam)
    static int write_count = 0;
    if (++write_count % 500 == 0)
    {
      RCLCPP_DEBUG(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[write] Write #%d: Sent message (length=%zu): %s",
        write_count, message.length(), message.c_str());
    }

    json gripper_payload;
    {
      std::lock_guard<std::mutex> lock(gripper_mutex_);
      if (gripper_cmd_left_.received && controlled_arm_ != "right")
      {
        gripper_payload["Robot0"] = make_gripper_payload(gripper_cmd_left_);
      }
      if (gripper_cmd_right_.received && controlled_arm_ != "left")
      {
        gripper_payload["Robot1"] = make_gripper_payload(gripper_cmd_right_);
      }
    }

    if (!mk2_guarded_control_ && !gripper_payload.empty())
    {
      const std::string gripper_message = "Gripper " + gripper_payload.dump();
      auto gripper_result = publisher_->send(zmq::buffer(gripper_message));
      if (!gripper_result)
      {
        RCLCPP_WARN(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[write] Failed to send gripper message (result is false)");
      }
      if (write_count % 500 == 0)
      {
        RCLCPP_DEBUG(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[write] Write #%d: Sent gripper message (length=%zu): %s",
          write_count, gripper_message.length(), gripper_message.c_str());
      }
    }
  }
  catch (const std::exception & ex)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to send joint command: %s", ex.what());
    if (mk2_guarded_control_) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      mk2_local_fault_ = true;
      mk2_fault_reason_ = "joint command transport exception";
    }
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

bool JeZmqSystemHardware::guarded_state_ok_locked()
{
  if (!mk2_local_fault_ && (!mk2_seen_feedback_ ||
      std::chrono::duration<double>(std::chrono::steady_clock::now() - mk2_last_feedback_).count() >
      mk2_feedback_timeout_)) {
    mk2_local_fault_ = true;
    mk2_fault_reason_ = "native feedback missing/stale";
  }
  if (mk2_local_fault_ && !mk2_fault_reported_) {
    RCLCPP_ERROR(rclcpp::get_logger("JeZmqSystemHardware"),
      "MK2 control fault: %s; rejecting writes, no automatic recovery", mk2_fault_reason_.c_str());
    mk2_fault_reported_ = true;
  }
  return !mk2_local_fault_;
}

void JeZmqSystemHardware::state_receive_loop()
{
  RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"), "state_receive_loop started");
  int msg_count = 0;
  bool first_message_logged = false;
  
  while (state_thread_running_.load())
  {
    if (!subscriber_)
    {
      break;
    }

    zmq::message_t msg;
    try
    {
      auto res = subscriber_->recv(msg, zmq::recv_flags::none);
      if (!res)
      {
        continue;
      }
    }
    catch (const zmq::error_t & ex)
    {
      if (ex.num() == ETERM || ex.num() == EAGAIN)
      {
        continue;
      }
      RCLCPP_ERROR(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "ZMQ receive error: %s", ex.what());
      continue;
    }
    catch (const std::exception & ex)
    {
      RCLCPP_ERROR(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "State receive exception: %s", ex.what());
      continue;
    }

    try
    {
      const std::string state_message(static_cast<char *>(msg.data()), msg.size());
      const auto pos = state_message.find(' ');
      if (pos == std::string::npos)
      {
        continue;
      }

      const std::string topic = state_message.substr(0, pos);
      if (topic != "State")
      {
        continue;
      }

      const auto state_json = json::parse(state_message.substr(pos + 1));
      if (mk2_guarded_control_) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        const auto error = mk2_guarded_state_error(state_json, mk2_releasing_,
          mk2_expected_backend_, mk2_device_config_sha256_, mk2_shared_ownership_);
        if (!error.empty()) {
          mk2_local_fault_ = true;
          mk2_fault_reason_ = error;
          if (state_json.contains("Bridge") && state_json["Bridge"].is_object() &&
              state_json["Bridge"].contains("control_state") &&
              state_json["Bridge"]["control_state"].is_string()) {
            mk2_phase_ = state_json["Bridge"]["control_state"].get<std::string>();
          }
          state_condition_.notify_all();
          continue;
        }
      }

  // Log first message received for diagnostics
      if (!first_message_logged)
      {
        RCLCPP_DEBUG(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[state_receive_loop] FIRST MESSAGE RECEIVED. Message size: %zu bytes. Full JSON:\n%s",
          state_message.size(),
          state_json.dump(2).c_str());
        first_message_logged = true;
      }

      // Log every 500 messages with message size and raw data to detect if messages are changing
      static int size_log_count = 0;
      if (++size_log_count % 500 == 0)
      {
        RCLCPP_DEBUG(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[state_receive_loop] Message #%d: size=%zu bytes. Payload: %s",
          size_log_count,
          state_message.size(),
          state_message.substr(0, std::min(size_t(200), state_message.size())).c_str());
      }

      std::vector<double> positions(joint_names_.size(), 0.0);
      std::vector<double> velocities(joint_names_.size(), 0.0);
      std::vector<double> efforts(joint_names_.size(), 0.0);

      if (!parse_joint_state_from_json(state_json, positions, velocities, efforts))
      {
  // Log failed parsing every 500 messages to help diagnose format issues
        static int parse_fail_count = 0;
  if (++parse_fail_count % 500 == 0)
        {
          // Dump the entire JSON to see what we're actually receiving
          RCLCPP_WARN(
            rclcpp::get_logger("JeZmqSystemHardware"),
            "[state_receive_loop] Failed to parse state JSON (fail count=%d). Full message:\n%s",
            parse_fail_count,
            state_message.c_str());
        }
        continue;
      }

      publish_oculus_feedback_from_json(state_json);

      msg_count++;
      if (msg_count % 500 == 0)  // Log every 500 messages to reduce spam
      {
        RCLCPP_DEBUG(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[state_receive_loop] Received %d messages. Parsed positions: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
          msg_count,
          positions[0], positions[1], positions[2], positions[3], 
          positions[4], positions[5], positions[6]);
      }

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (mk2_guarded_control_) {
          const auto sequence = state_json["Bridge"]["sequence"].get<uint64_t>();
          if (mk2_seen_feedback_ && sequence <= mk2_last_sequence_) {
            if (sequence < mk2_last_sequence_) {
              mk2_local_fault_ = true;
              mk2_fault_reason_ = "native sequence moved backwards/session changed";
              state_condition_.notify_all();
            }
            continue;  // Duplicate frames never renew freshness.
          }
          if (!std::all_of(positions.begin(), positions.end(), [](double q) { return std::isfinite(q); }) ||
              !std::all_of(velocities.begin(), velocities.end(), [](double q) { return std::isfinite(q); }) ||
              !std::all_of(efforts.begin(), efforts.end(), [](double q) { return std::isfinite(q); })) {
            mk2_local_fault_ = true;
            mk2_fault_reason_ = "nonfinite native joint feedback";
            state_condition_.notify_all();
            continue;
          }
          mk2_last_sequence_ = sequence;
          mk2_last_feedback_ = std::chrono::steady_clock::now();
          mk2_seen_feedback_ = true;
          mk2_phase_ = state_json["Bridge"]["control_state"].get<std::string>();
          const auto & bridge = state_json["Bridge"];
          mk2_clean_release_supported_ = bridge.contains("clean_release_supported") &&
            bridge["clean_release_supported"].is_boolean() &&
            bridge["clean_release_supported"].get<bool>();
          mk2_handover_supported_ = mk2_clean_release_supported_ &&
            bridge.contains("ownership_handover_supported") &&
            bridge["ownership_handover_supported"].is_boolean() &&
            bridge["ownership_handover_supported"].get<bool>();
        }
  // Log before updating
  if (msg_count % 500 == 0)
        {
          RCLCPP_DEBUG(
            rclcpp::get_logger("JeZmqSystemHardware"),
            "[state_receive_loop] Updating latest_positions. Before: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
            latest_positions_[0], latest_positions_[1], latest_positions_[2], latest_positions_[3],
            latest_positions_[4], latest_positions_[5], latest_positions_[6]);
        }
        
        latest_positions_ = std::move(positions);
        latest_velocities_ = std::move(velocities);
        latest_efforts_ = std::move(efforts);
        has_valid_state_ = true;
        state_condition_.notify_all();
        
  // Log after updating
  if (msg_count % 500 == 0)
        {
          RCLCPP_DEBUG(
            rclcpp::get_logger("JeZmqSystemHardware"),
            "[state_receive_loop] Updated latest_positions. After: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
            latest_positions_[0], latest_positions_[1], latest_positions_[2], latest_positions_[3],
            latest_positions_[4], latest_positions_[5], latest_positions_[6]);
        }
      }
    }
    catch (const std::exception & ex)
    {
      RCLCPP_ERROR(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Failed parsing state JSON: %s", ex.what());
      if (mk2_guarded_control_) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        mk2_local_fault_ = true;
        mk2_fault_reason_ = "native state parse/publish failure";
        state_condition_.notify_all();
      }
    }
  }
  
  RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"), "state_receive_loop stopped (total messages: %d)", msg_count);
}

void JeZmqSystemHardware::gripper_spin_loop()
{
  while (gripper_thread_running_.load())
  {
    if (gripper_feedback_subscriber_ && pub_oculus_feedback_) {
      zmq::pollitem_t items[] = {{*gripper_feedback_subscriber_, 0, ZMQ_POLLIN, 0}};
      if (zmq::poll(items, 1, std::chrono::milliseconds(20)) > 0 && (items[0].revents & ZMQ_POLLIN)) {
        zmq::message_t msg;
        if (gripper_feedback_subscriber_->recv(msg, zmq::recv_flags::dontwait)) {
          const std::string frame(static_cast<const char *>(msg.data()), msg.size());
          constexpr const char * prefix = "GripperFeedback ";
          if (frame.rfind(prefix, 0) == 0) {
            try {
              const auto j = nlohmann::json::parse(frame.substr(std::char_traits<char>::length(prefix)));
              {
                std::lock_guard<std::mutex> lock(gripper_feedback_mutex_);
                for (const char * key : {"Robot0", "Robot1"}) {
                  if (!j.contains(key) || !j[key].is_object()) continue;
                  const auto & r = j[key];
                  if (!r.contains("feedback_valid") || !r.contains("position") ||
                      !r["feedback_valid"].is_boolean() || !r["position"].is_number()) continue;
                  if (std::string(key) == "Robot0") {
                    left_gripper_feedback_valid_ = r["feedback_valid"].get<bool>();
                    left_gripper_feedback_ = static_cast<float>(r["position"].get<double>());
                  } else {
                    right_gripper_feedback_valid_ = r["feedback_valid"].get<bool>();
                    right_gripper_feedback_ = static_cast<float>(r["position"].get<double>());
                  }
                }
              }
              common::msg::OculusInitJointState feedback;
              feedback.header.stamp = gripper_node_->now();
              feedback.init = true;
              {
                std::lock_guard<std::mutex> lock(gripper_feedback_mutex_);
                if (left_gripper_feedback_valid_) {
                  feedback.left_valid = true;
                  feedback.left_gripper = left_gripper_feedback_;
                }
                if (right_gripper_feedback_valid_) {
                  feedback.right_valid = true;
                  feedback.right_gripper = right_gripper_feedback_;
                }
              }
              pub_oculus_feedback_->publish(feedback);
            } catch (const std::exception & ex) {
              RCLCPP_WARN(rclcpp::get_logger("JeZmqSystemHardware"),
                          "Bad gripper feedback frame: %s", ex.what());
            }
          }
        }
      }
    }

    if (!gripper_executor_) break;
    try {
      gripper_executor_->spin_once(50ms);
    } catch (const std::exception & ex) {
      RCLCPP_WARN(rclcpp::get_logger("JeZmqSystemHardware"), "Gripper bridge executor stopped: %s", ex.what());
      break;
    }
  }
}

void JeZmqSystemHardware::handle_gripper_cmd(
  const je_software::msg::EndEffectorCommand & msg,
  int robot_index)
{
  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "[gripper_cmd] received robot_index=%d mode=%d position=%.4f preset=%d command=%s torque=%.4f",
    robot_index, msg.mode, msg.position, msg.preset, msg.command.c_str(), msg.torque);

  if (msg.mode != je_software::msg::EndEffectorCommand::MODE_POSITION &&
    msg.mode != je_software::msg::EndEffectorCommand::MODE_PRESET &&
    msg.mode != je_software::msg::EndEffectorCommand::MODE_TORQUE)
  {
    RCLCPP_WARN(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Invalid end effector mode: %d", msg.mode);
    return;
  }

  std::string command = msg.command;
  std::transform(command.begin(), command.end(), command.begin(), ::tolower);

  if (msg.mode == je_software::msg::EndEffectorCommand::MODE_TORQUE)
  {
    if (
      command != je_software::msg::EndEffectorCommand::CMD_OPEN &&
      command != je_software::msg::EndEffectorCommand::CMD_CLOSE)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Invalid torque gripper command: '%s'", command.c_str());
      return;
    }

    if (!std::isfinite(msg.torque) || msg.torque <= 0.0)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Invalid torque value: %.4f", msg.torque);
      return;
    }
  }

  nlohmann::json mk2_payload;
  {
    std::lock_guard<std::mutex> lock(gripper_mutex_);
    auto & target = (robot_index == 1) ? gripper_cmd_right_ : gripper_cmd_left_;
    target.mode = msg.mode;
    target.position = msg.position;
    target.preset = msg.preset;
    target.command = command;
    target.torque = msg.torque;
    target.received = true;
    if (mk2_guarded_control_) {
      mk2_payload[(robot_index == 1) ? "Robot1" : "Robot0"] = make_gripper_payload(target);
    }
  }

  // MK2: send once per command on the dedicated gripper port (never on the arm port).
  if (mk2_guarded_control_) {
    if (!gripper_publisher_) {
      RCLCPP_WARN(rclcpp::get_logger("JeZmqSystemHardware"),
        "MK2 gripper command dropped: mk2_gripper_port not configured");
      return;
    }
    const std::string frame = "Gripper " + mk2_payload.dump();
    try {
      if (!gripper_publisher_->send(zmq::buffer(frame), zmq::send_flags::dontwait)) {
        RCLCPP_WARN(rclcpp::get_logger("JeZmqSystemHardware"), "MK2 gripper send would block");
      } else {
        RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"), "MK2 gripper sent: %s", frame.c_str());
      }
    } catch (const std::exception & ex) {
      RCLCPP_WARN(rclcpp::get_logger("JeZmqSystemHardware"), "MK2 gripper send failed: %s", ex.what());
    }
  }
}

void JeZmqSystemHardware::gripper_cmd_callback(
  const je_software::msg::EndEffectorCommandLR::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }

  if (msg->left_valid && (mk2_guarded_control_ || controlled_arm_ != "right"))
  {
    handle_gripper_cmd(msg->left, 0);
  }
  if (msg->right_valid && (mk2_guarded_control_ || controlled_arm_ != "left"))
  {
    handle_gripper_cmd(msg->right, 1);
  }
}

void JeZmqSystemHardware::publish_oculus_feedback_from_json(const nlohmann::json & state_json) const
{
  if (!pub_oculus_feedback_)
  {
    return;
  }

  common::msg::OculusInitJointState msg;
  msg.header.stamp = gripper_node_->now();
  msg.init = true;

  auto fill_side = [](const nlohmann::json & robot, sensor_msgs::msg::JointState & joint_state,
      float & gripper, bool & valid)
    {
      const auto fetch_array = [&robot](const char * lower_key, const char * upper_key,
          std::vector<double> & out) -> bool
        {
          if (robot.contains(lower_key) && robot[lower_key].is_array())
          {
            out = robot[lower_key].get<std::vector<double>>();
            return true;
          }
          if (robot.contains(upper_key) && robot[upper_key].is_array())
          {
            out = robot[upper_key].get<std::vector<double>>();
            return true;
          }
          return false;
        };

      fetch_array("joint", "Joint", joint_state.position);
      fetch_array("jointVelocity", "JointVelocity", joint_state.velocity);
      fetch_array("jointSensorTorque", "JointSensorTorque", joint_state.effort);

      valid = false;
      gripper = 0.0F;
      if (!robot.contains("EndEffector") || !robot["EndEffector"].is_object())
      {
        return;
      }

      const auto & end_effector = robot["EndEffector"];
      if (end_effector.contains("feedback_valid") && end_effector["feedback_valid"].is_boolean())
      {
        valid = end_effector["feedback_valid"].get<bool>();
      }
      else
      {
        valid = end_effector.contains("CurrentPosition") && end_effector["CurrentPosition"].is_number();
      }

      if (end_effector.contains("CurrentPosition") && end_effector["CurrentPosition"].is_number())
      {
        gripper = static_cast<float>(end_effector["CurrentPosition"].get<double>());
      }
    };

  if (state_json.contains("Robot0") && state_json["Robot0"].is_object())
  {
    fill_side(state_json["Robot0"], msg.left, msg.left_gripper, msg.left_valid);
  }
  if (state_json.contains("Robot1") && state_json["Robot1"].is_object())
  {
    fill_side(state_json["Robot1"], msg.right, msg.right_gripper, msg.right_valid);
  }

  {
    std::lock_guard<std::mutex> lock(gripper_feedback_mutex_);
    if (left_gripper_feedback_valid_) {
      msg.left_valid = true;
      msg.left_gripper = left_gripper_feedback_;
    }
    if (right_gripper_feedback_valid_) {
      msg.right_valid = true;
      msg.right_gripper = right_gripper_feedback_;
    }
  }

  pub_oculus_feedback_->publish(msg);
}

bool JeZmqSystemHardware::map_joint_to_robot_slot(
  const std::string & joint_name,
  int & robot_index,
  std::size_t & slot_index) const
{
  auto parse_number = [](const std::string & name, const std::string & prefix, int & out_value) -> bool
    {
      if (name.rfind(prefix, 0) != 0)
      {
        return false;
      }
      const auto suffix = name.substr(prefix.size());
      if (suffix.empty())
      {
        return false;
      }
      if (!std::all_of(suffix.begin(), suffix.end(), [](unsigned char ch) { return std::isdigit(ch); }))
      {
        return false;
      }
      try
      {
        out_value = std::stoi(suffix);
      }
      catch (...)
      {
        return false;
      }
      return true;
    };

  int value = 0;

  if (parse_number(joint_name, "joint", value))
  {
    if (value >= 1 && value <= 7)
    {
      robot_index = 0;
      slot_index = static_cast<std::size_t>(value - 1);
      return true;
    }
    if (value >= 11 && value <= 17)
    {
      robot_index = 0;
      slot_index = static_cast<std::size_t>(value - 11);
      return true;
    }
    if (value >= 21 && value <= 27)
    {
      robot_index = 1;
      slot_index = static_cast<std::size_t>(value - 21);
      return true;
    }
  }

  if (parse_number(joint_name, "l_joint", value) && value >= 1 && value <= 7)
  {
    robot_index = 0;
    slot_index = static_cast<std::size_t>(value - 1);
    return true;
  }

  if (parse_number(joint_name, "r_joint", value) && value >= 1 && value <= 7)
  {
    robot_index = 1;
    slot_index = static_cast<std::size_t>(value - 1);
    return true;
  }

  return false;
}

const JeZmqSystemHardware::GripperCommand & JeZmqSystemHardware::get_gripper_command(int robot_index) const
{
  return (robot_index == 1) ? gripper_cmd_right_ : gripper_cmd_left_;
}

nlohmann::json JeZmqSystemHardware::make_gripper_payload(const GripperCommand & gripper_cmd) const
{
  nlohmann::json ee;
  ee["mode"] = gripper_cmd.mode;
  if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_POSITION)
  {
    ee["position"] = gripper_cmd.position;
  }
  else if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_PRESET)
  {
    ee["preset"] = gripper_cmd.preset;
  }
  else if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_TORQUE)
  {
    const double target_position = std::clamp(gripper_cmd.position, 0.0, 1.0);
    ee["torque"] = gripper_cmd.torque;
    ee["position"] = target_position;
    ee["angle_0p001mm"] = static_cast<int>(std::lround(target_position * 70000.0));
    ee["effort_0p001Nm"] = static_cast<int>(std::lround(gripper_cmd.torque * 1000.0));
  }
  return ee;
}

void JeZmqSystemHardware::append_gripper_from_command(nlohmann::json & payload, int robot_index) const
{
  std::lock_guard<std::mutex> lock(gripper_mutex_);
  const auto & gripper_cmd = get_gripper_command(robot_index);
  if (!gripper_cmd.received)
  {
    return;
  }

  auto & robot = payload[(robot_index == 1) ? "Robot1" : "Robot0"];
  
  robot["EndEffector"] = make_gripper_payload(gripper_cmd);
}

bool JeZmqSystemHardware::parse_joint_state_from_json(
  const nlohmann::json & state_json,
  std::vector<double> & positions,
  std::vector<double> & velocities,
  std::vector<double> & efforts) const
{
  if (!state_json.contains("Robot0") || !state_json["Robot0"].is_object())
  {
    return false;
  }

  const auto fetch_array = [](const nlohmann::json & robot, const char * lower_key, const char * upper_key,
      std::vector<double> & out) -> bool
    {
      if (robot.contains(lower_key))
      {
        out = robot[lower_key].get<std::vector<double>>();
        return true;
      }
      if (robot.contains(upper_key))
      {
        out = robot[upper_key].get<std::vector<double>>();
        return true;
      }
      return false;
    };

  std::vector<double> joint0;
  std::vector<double> joint1;
  std::vector<double> vel0;
  std::vector<double> vel1;
  std::vector<double> effort0;
  std::vector<double> effort1;

  const auto & robot0 = state_json["Robot0"];
  if (!fetch_array(robot0, "joint", "Joint", joint0))
  {
    return false;
  }
  fetch_array(robot0, "jointVelocity", "JointVelocity", vel0);
  fetch_array(robot0, "jointSensorTorque", "JointSensorTorque", effort0);

  const bool has_robot1 = state_json.contains("Robot1") && state_json["Robot1"].is_object();
  if (has_robot1)
  {
    const auto & robot1 = state_json["Robot1"];
    fetch_array(robot1, "joint", "Joint", joint1);
    fetch_array(robot1, "jointVelocity", "JointVelocity", vel1);
    fetch_array(robot1, "jointSensorTorque", "JointSensorTorque", effort1);
  }

  for (std::size_t i = 0; i < positions.size() && i < joint_names_.size(); ++i)
  {
    int robot_index = 0;
    std::size_t slot_index = 0;
    if (!map_joint_to_robot_slot(joint_names_[i], robot_index, slot_index))
    {
      if (positions.size() <= 7 || i < 7)
      {
        robot_index = 0;
        slot_index = i;
      }
      else
      {
        robot_index = 1;
        slot_index = i - 7;
      }
    }

    const auto * joint_source = (robot_index == 1) ? &joint1 : &joint0;
    const auto * vel_source = (robot_index == 1) ? &vel1 : &vel0;
    const auto * effort_source = (robot_index == 1) ? &effort1 : &effort0;

    if (joint_source->empty() || slot_index >= joint_source->size())
    {
      return false;
    }

    positions[i] = (*joint_source)[slot_index];
    if (slot_index < vel_source->size())
    {
      velocities[i] = (*vel_source)[slot_index];
    }
    if (slot_index < effort_source->size())
    {
      efforts[i] = (*effort_source)[slot_index];
    }
  }

  return true;
}

void JeZmqSystemHardware::robot_switch(bool value)
{
  if (!publisher_)
  {
    return;
  }

  json cmd_switch;
  cmd_switch["Switch"] = value;
  const std::string payload = "Switch " + cmd_switch.dump();
  publisher_->send(zmq::buffer(payload), zmq::send_flags::none);
}

}  // namespace je_software

PLUGINLIB_EXPORT_CLASS(je_software::JeZmqSystemHardware, hardware_interface::SystemInterface)
