#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <ruckig/ruckig.hpp>

namespace je_software
{
// Opt-in MK2 command adapter. Native position/following/watchdog guards remain
// authoritative; infeasible motion is rejected, never zero-filled. Only
// sub-picoradian floating-point boundary roundoff is snapped back into bounds.
class Mk2GuardedCommand
{
public:
  using Axes = std::array<double, 7>;
  // Default caps. Physical may raise them up to physical_max_* (inside the
  // native physical policy 2.7 / 15); only sdk_simulation may go beyond.
  static constexpr double default_max_velocity = 0.24;
  static constexpr double default_max_acceleration = 0.4;
  static constexpr double physical_max_velocity = 2.4;
  static constexpr double physical_max_acceleration = 12.0;
  // A plan starts from the latest measured /joint_states sample, while a
  // settled hold keeps emitting the earlier commanded pose. Their difference
  // can be up to JTC's goal tolerance / MoveIt's allowed_start_tolerance (0.02
  // rad) when consecutive trajectories run back-to-back, yet even a 4e-5 rad
  // step from rest exceeds a_cap at 10 ms. Absorb settled start offsets up to
  // 0.02 rad by fading them out under JTC's own clock (all emitted samples
  // remain validated against v_cap / a_cap); larger offsets remain rejected.
  static constexpr double max_start_offset = 0.02;
  static constexpr double start_offset_duration = 0.5;
  static constexpr double settled_velocity = 1e-6;
  explicit Mk2GuardedCommand(double dt, double v_cap = default_max_velocity,
    double a_cap = default_max_acceleration)
  : max_velocity_(v_cap), max_acceleration_(a_cap), generator_(dt), dt_(dt) {}
  double max_velocity() const { return max_velocity_; }
  double max_acceleration() const { return max_acceleration_; }
  bool smoothing() const { return smoothing_; }
  bool blending() const { return blending_; }

  void reset(const Axes & position, const Axes & lower, const Axes & upper)
  {
    input_ = ruckig::InputParameter<7>();
    input_.current_position = position;
    input_.current_velocity.fill(0.0);
    input_.current_acceleration.fill(0.0);
    input_.target_velocity.fill(0.0);
    input_.target_acceleration.fill(0.0);
    // Leave margin below the independent native policy limits.
    input_.max_velocity.fill(max_velocity_);
    // OTG bounds instantaneous acceleration; native bounds differences of
    // interval-average velocity. Leave margin between them when dt jitters
    // (e.g. alternating 8/12 ms), without relaxing emitted/native limits.
    // Defaults: 0.75 * 0.4 = 0.3 rad/s^2, jerk 5 * 0.4 = 2.0 rad/s^3.
    input_.max_acceleration.fill(0.75 * max_acceleration_);
    input_.max_jerk.fill(5.0 * max_acceleration_);
    input_.synchronization = ruckig::Synchronization::None;
    lower_ = lower;
    upper_ = upper;
    previous_position_ = position;
    previous_velocity_.fill(0.0);
    previous_acceleration_.fill(0.0);
    previous_target_ = position;
    previous_target_velocity_.fill(0.0);
    previous_dt_ = 0.0;
    smoothing_ = false;
    braking_ = false;
    brake_held_ = false;
    brake_hold_target_ = position;
    blending_ = false;
    first_ = true;
    generator_.reset();
  }

  bool step(const Axes & target, const Axes & target_velocity, Axes & result, std::string & error,
    double sample_dt)
  {
    if (!std::isfinite(sample_dt) || sample_dt < 0.005 || sample_dt > 0.05) {
      error = "invalid trajectory sample interval";
      return false;
    }
    dt_ = sample_dt;
    generator_.delta_time = dt_;
    bool holding = true;
    for (size_t i = 0; i < 7; ++i) {
      if (!std::isfinite(target[i]) || target[i] < lower_[i] || target[i] > upper_[i]) {
        error = "invalid/out-of-bounds requested target";
        return false;
      }
      if (!std::isfinite(target_velocity[i]) || std::abs(target_velocity[i]) > max_velocity_ + 1e-8) {
        error = "invalid/over-speed trajectory velocity";
        return false;
      }
      holding = holding && std::abs(target_velocity[i]) <= 1e-10;
      if (first_ && (std::abs(target[i] - previous_position_[i]) > 1e-10 ||
          std::abs(target_velocity[i]) > 1e-10)) {
        error = "first command must hold the measured activation pose";
        return false;
      }
    }
    // JTC (humble 2.52) skips its command write in the cycle it aborts on a
    // path/goal tolerance, so the interfaces repeat the previous position AND
    // the previous non-zero velocity bit-for-bit. Such a sample cannot belong
    // to a timed trajectory; treat it as the stop request it is.
    bool stale = !first_;
    for (size_t i = 0; i < 7 && stale; ++i) {
      stale = target[i] == previous_target_[i] && target_velocity[i] == previous_target_velocity_[i];
    }
    holding = holding || stale;
    // After a smooth velocity brake (e.g. pause/cancel), JTC keeps sending the
    // frozen cancel-time hold pose while the arm is already safely parked at
    // its natural stopping position. Stay parked without reversing back until a
    // new motion or new target arrives.
    if (brake_held_) {
      bool same_hold_target = holding;
      for (size_t i = 0; i < 7 && same_hold_target; ++i) {
        same_hold_target = std::abs(target[i] - brake_hold_target_[i]) <= 1e-10;
      }
      if (same_hold_target) {
        result = previous_position_;
        Axes zero_v{}, zero_a{};
        commit(result, zero_v, zero_a);
        return true;
      }
      brake_held_ = false;
    }

    Axes velocity, acceleration;
    std::string violation;
    // Engage only when the raw sample would be rejected: a feasible first
    // step of the trajectory itself must pass through unmodified.
    if (!first_ && !smoothing_ && !blending_ && !validate(target, velocity, acceleration)) {
      bool settled = true;
      double offset = 0.0;
      for (size_t i = 0; i < 7; ++i) {
        settled = settled && std::abs(previous_velocity_[i]) <= settled_velocity;
        offset = std::max(offset, std::abs(previous_position_[i] - target[i]));
      }
      if (settled && offset > 0.0 && offset <= max_start_offset) {
        for (size_t i = 0; i < 7; ++i) start_offset_[i] = previous_position_[i] - target[i];
        blend_elapsed_ = 0.0;
        blending_ = true;
      }
    }
    Axes command = target;
    if (blending_) {
      // Quintic fade 1 -> 0 with zero slope/curvature at both ends. The first
      // emitted sample equals the settled pose exactly; peak added accel is
      // 5.78 * 0.02 / 0.5^2 ~= 0.46 rad/s^2 at the maximum allowed offset.
      const double u = std::min(blend_elapsed_ / start_offset_duration, 1.0);
      const double fade = 1.0 - u * u * u * (10.0 - 15.0 * u + 6.0 * u * u);
      for (size_t i = 0; i < 7; ++i) command[i] = target[i] + start_offset_[i] * fade;
    }

    if ((!smoothing_ || !holding) && validate(command, velocity, acceleration, &violation)) {
      // JTC already sampled a timed trajectory. Preserve it exactly whenever
      // the emitted stream is feasible: do NOT retime every point to v=0.
      result = command;
      commit(result, velocity, acceleration);
      smoothing_ = false;
      braking_ = false;
      first_ = false;
      if (blending_) {
        blend_elapsed_ += dt_;
        blending_ = blend_elapsed_ < start_offset_duration + 1e-9;
      }
      return true;
    }
    blending_ = false;
    if (!holding) {
      // Invalid moving trajectories are rejected, not silently slowed down
      // behind JTC's original clock. A new goal during a stop must also rejoin
      // the emitted stream within the same discrete limits.
      braking_ = false;
      error = "timed trajectory discontinuity/limit violation; " + violation +
        "; replan from settled feedback";
      return false;
    }
    bool was_moving = false;
    for (size_t i = 0; i < 7; ++i) {
      was_moving = was_moving || std::abs(previous_velocity_[i]) > settled_velocity;
    }
    if (!smoothing_) {
      input_.current_position = previous_position_;
      input_.current_velocity = previous_velocity_;
      input_.current_acceleration = previous_acceleration_;
      braking_ = was_moving;
      if (braking_) {
        // Cancel/abort while moving: brake smoothly to v=0 in velocity mode
        // instead of overshooting and reversing back to the cancel pose.
        input_.control_interface = ruckig::ControlInterface::Velocity;
        brake_hold_target_ = target;
      } else {
        input_.control_interface = ruckig::ControlInterface::Position;
      }
      generator_.reset();
    }
    if (!braking_) {
      input_.target_position = target;
    }
    input_.target_velocity.fill(0.0);
    input_.target_acceleration.fill(0.0);
    const auto status = generator_.update(input_, output_);
    if (status != ruckig::Result::Working && status != ruckig::Result::Finished) {
      error = "Ruckig could not produce a bounded command";
      return false;
    }
    for (size_t i = 0; i < 7; ++i) {
      double q = output_.new_position[i];
      // Finished Ruckig trajectories can exceed an exact target by one ULP.
      // Do not widen native limits or clip real overshoot: validate discrete
      // derivatives again after correcting only numerical boundary roundoff.
      if (q > upper_[i] && q - upper_[i] <= 1e-12) q = upper_[i];
      if (q < lower_[i] && lower_[i] - q <= 1e-12) q = lower_[i];
      output_.new_position[i] = q;
    }
    if (!validate(output_.new_position, velocity, acceleration, &violation)) {
      error = "bounded hold failed discrete position/velocity/acceleration validation; " + violation;
      return false;
    }
    result = output_.new_position;
    commit(result, velocity, acceleration);
    // Keep a stop on its continuous OTG state until it actually finishes.
    // Crossing the hold position with nonzero speed is NOT a safe handover:
    // snapping there would restart the decelerator on the following cycle.
    smoothing_ = status != ruckig::Result::Finished;
    if (!smoothing_ && braking_) {
      braking_ = false;
      brake_held_ = true;
      brake_hold_target_ = target;
    }
    first_ = false;
    output_.pass_to_input(input_);
    return true;
  }

private:
  bool validate(const Axes & position, Axes & velocity, Axes & acceleration,
    std::string * violation = nullptr) const
  {
    for (size_t i = 0; i < 7; ++i) {
      velocity[i] = (position[i] - previous_position_[i]) / dt_;
      acceleration[i] = (velocity[i] - previous_velocity_[i]) / dt_;
      if (!std::isfinite(position[i]) || !std::isfinite(velocity[i]) || !std::isfinite(acceleration[i]) ||
          position[i] < lower_[i] || position[i] > upper_[i] ||
          std::abs(velocity[i]) > max_velocity_ + 1e-8 ||
          std::abs(acceleration[i]) > max_acceleration_ + 1e-8) {
        if (violation) {
          // Construct diagnostics only on rejection. These are command-stream
          // finite differences, NOT measured mechanism velocity/acceleration.
          std::ostringstream detail;
          detail << std::setprecision(17) << "joint=" << (i + 1)
                 << " q=" << position[i] << " previous_q=" << previous_position_[i]
                 << " dt=" << dt_ << " previous_dt=" << previous_dt_
                 << " command_fd_v=" << velocity[i]
                 << " previous_command_fd_v=" << previous_velocity_[i]
                 << " command_fd_a=" << acceleration[i]
                 << " previous_command_fd_a=" << previous_acceleration_[i]
                 << " bounds=[" << lower_[i] << ',' << upper_[i] << ']'
                 << " v_cap=" << max_velocity_ << " a_cap=" << max_acceleration_;
          *violation = detail.str();
        }
        return false;
      }
    }
    return true;
  }
  void commit(const Axes & position, const Axes & velocity, const Axes & acceleration)
  {
    previous_position_ = position;
    previous_velocity_ = velocity;
    previous_acceleration_ = acceleration;
    previous_dt_ = dt_;
  }
  double max_velocity_;
  double max_acceleration_;
  ruckig::Ruckig<7> generator_;
  ruckig::InputParameter<7> input_;
  ruckig::OutputParameter<7> output_;
  double dt_;
  double previous_dt_{0.0};
  bool smoothing_{false}, braking_{false}, brake_held_{false}, blending_{false}, first_{true};
  double blend_elapsed_{0.0};
  Axes start_offset_{};
  Axes brake_hold_target_{};
  Axes lower_{}, upper_{}, previous_position_{}, previous_velocity_{}, previous_acceleration_{};
  Axes previous_target_{}, previous_target_velocity_{};
};

inline std::string mk2_guarded_state_error(const nlohmann::json & state, bool allow_release = false,
  const std::string & expected_backend = "sdk_simulation", const std::string & device_sha256 = "",
  bool allow_physical_handover = false)
{
  try {
    const auto & b = state.at("Bridge");
    const auto & r = state.at("Robot0");
    const bool physical = expected_backend == "physical";
    if ((!physical && expected_backend != "sdk_simulation") ||
        b.at("mode") != "control_guarded" || b.at("backend") != expected_backend ||
        !b.at("simulation_only").is_boolean() || b.at("simulation_only").get<bool>() == physical ||
        !b.at("simulation_isolated").is_boolean() || b.at("simulation_isolated").get<bool>() == physical) {
      return "native backend does not match the explicitly selected guarded backend";
    }
    const bool handover = allow_physical_handover &&
      b.value("ownership_handover_supported", nlohmann::json(false)) == nlohmann::json(true) &&
      b.value("clean_release_supported", nlohmann::json(false)) == nlohmann::json(true);
    if (physical) {
      if (device_sha256.size() != 64 || b.at("device_config_sha256") != device_sha256 ||
          b.at("profile") != "/home/robot/Work/system/robot_config_new_left" ||
          !b.at("configuration_verified").is_boolean() || !b.at("configuration_verified").get<bool>() ||
          !b.at("clean_release_supported").is_boolean() ||
          (b.at("clean_release_supported").get<bool>() && !handover)) {
        return "physical profile/configuration is not verified or advertises unsupported release";
      }
      for (const auto * key : {"DriverError", "MoveError"}) {
        if (!r.at(key).is_number_integer() || r.at(key).get<int>() != 0) {
          return "physical SDK reports driver/motion error";
        }
      }
    }
    const auto phase = b.at("control_state").get<std::string>();
    const bool releasing = (!physical || handover) && allow_release && phase == "releasing" &&
      b.contains("clean_release_supported") && b.at("clean_release_supported").is_boolean() &&
      b.at("clean_release_supported").get<bool>();
    if (b.at("fault") != "none" || !b.at("sdk_error").is_number_integer() ||
        b.at("sdk_error").get<int>() != 0 || !b.at("commands_enabled").is_boolean() ||
        (!b.at("commands_enabled").get<bool>() && !releasing) ||
        (phase != "ready" && phase != "waiting_first_joint" && phase != "active" && !releasing)) {
      return "native command bridge refused control: " + b.dump();
    }
    if (!r.at("PowerState").is_number_integer() || r.at("PowerState").get<int>() != 1 ||
        !r.at("MoveState").is_number_integer() || r.at("MoveState").get<int>() < 0 ||
        r.at("MoveState").get<int>() == 7) {
      return "native power/motion state is not healthy";
    }
    const auto & q = r.contains("Joint") ? r.at("Joint") : r.at("joint");
    if (!q.is_array() || q.size() != 7) return "native joint feedback is incomplete";
    for (const auto & value : q) {
      if (!value.is_number() || !std::isfinite(value.get<double>())) {
        return "native joint feedback is nonnumeric/nonfinite";
      }
    }
    if (!b.at("sequence").is_number_unsigned()) return "native sequence is missing/invalid";
    return {};
  } catch (const std::exception &) {
    return "native guarded feedback schema is missing/invalid";
  }
}
}  // namespace je_software