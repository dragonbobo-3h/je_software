#!/usr/bin/env python3
import argparse
import collections

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node

from common.msg import OculusInitJointState


class LeftArmPlotNode(Node):
    def __init__(self, window: float = 10.0, fps: float = 30.0, topic: str = "/joint_cmd_double_arm"):
        super().__init__("left_arm_plot")
        self._window = max(1.0, window)
        self._period = 1.0 / max(1.0, fps)
        self._buf_len = int(self._window * fps) + 1

        self._times = collections.deque(maxlen=self._buf_len)
        self._positions: list[collections.deque] = []
        self._names: list[str] = []
        self._gripper = collections.deque(maxlen=self._buf_len)
        self._initialized = False
        self._t0: float | None = None

        self._sub = self.create_subscription(OculusInitJointState, topic, self._on_msg, 10)
        self._timer = self.create_timer(self._period, self._refresh)

        self._fig: plt.Figure | None = None
        self._axes: list[plt.Axes] = []
        self._lines: list[plt.Line2D] = []
        self._gripper_line: plt.Line2D | None = None

        self.get_logger().info(f"Plotting left arm from '{topic}', window={self._window}s, fps={fps:.0f}")

    def _init_plot(self, names: list[str]):
        self._names = list(names)
        n_joints = len(names)
        self._positions = [collections.deque(maxlen=self._buf_len) for _ in range(n_joints)]

        plt.ion()
        self._fig, self._axes = plt.subplots(n_joints + 1, 1, sharex=True, figsize=(12, 2.0 * (n_joints + 1)))
        if n_joints + 1 == 1:
            self._axes = [self._axes]

        for i, name in enumerate(names):
            (line,) = self._axes[i].plot([], [], lw=1)
            self._lines.append(line)
            self._axes[i].set_ylabel(name)
            self._axes[i].grid(True, alpha=0.3)

        (g_line,) = self._axes[-1].plot([], [], lw=1, color="tab:red")
        self._gripper_line = g_line
        self._axes[-1].set_ylabel("gripper")
        self._axes[-1].set_xlabel("t (s)")
        self._axes[-1].grid(True, alpha=0.3)

        self._fig.suptitle("Left Arm Joint Positions")
        self._fig.tight_layout()
        self._initialized = True

    def _on_msg(self, msg: OculusInitJointState):
        if not msg.left_valid:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = now

        names = list(msg.left.name) if msg.left.name else []
        if not self._initialized:
            names = names + ["gripper"] if names else ["gripper"]
            self._init_plot(names[:7])

        t = now - self._t0
        self._times.append(t)
        for i, pos in enumerate(msg.left.position[:7]):
            if i < len(self._positions):
                self._positions[i].append(pos)
        self._gripper.append(float(msg.left_gripper))

    def _refresh(self):
        if not self._initialized or self._fig is None:
            return
        times = list(self._times)
        if len(times) < 2:
            return

        for i, line in enumerate(self._lines):
            if i < len(self._positions):
                line.set_data(times, list(self._positions[i]))

        if self._gripper_line is not None:
            self._gripper_line.set_data(times, list(self._gripper))

        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()

        self._axes[-1].set_xlim(max(0, times[-1] - self._window), max(times[-1], self._window))
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def destroy_node(self):
        if self._fig is not None:
            plt.close(self._fig)
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="Real-time left arm joint position plotter")
    parser.add_argument("--window", type=float, default=10.0, help="Rolling window in seconds (default: 10)")
    parser.add_argument("--fps", type=float, default=30.0, help="Plot refresh rate (default: 30)")
    parser.add_argument("--topic", default="/joint_cmd_double_arm", help="Topic to subscribe to")
    args = parser.parse_args()

    rclpy.init()
    node = LeftArmPlotNode(window=args.window, fps=args.fps, topic=args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
