#!/usr/bin/env python3
"""ZMQ motion backend node — serves robot TCP pose for calibration.

Subscribes to the end-effector pose topic (published by je_robot_node)
and exposes /motion_backend/get_robot_state service.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from common.msg import OculusControllers
from je_software.srv import GetRobotState


class ZmqMotionBackendNode(Node):
    """Caches the latest TCP pose and serves it via GetRobotState service."""

    def __init__(self):
        super().__init__('zmq_motion_backend_node')

        self.declare_parameter('endpose_topic', '/endpose_states_double_arm')
        endpose_topic = self.get_parameter('endpose_topic').value

        self._latest_tcp_pose: PoseStamped | None = None

        self._endpose_sub = self.create_subscription(
            OculusControllers,
            endpose_topic,
            self._on_endpose,
            10,
        )

        self._srv = self.create_service(
            GetRobotState,
            '/motion_backend/get_robot_state',
            self._on_get_robot_state,
        )

        self.get_logger().info(
            f'Subscribed to {endpose_topic}, '
            f'serving /motion_backend/get_robot_state'
        )

    def _on_endpose(self, msg: OculusControllers) -> None:
        if msg.left_valid:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = msg.left_pose
            self._latest_tcp_pose = ps
        elif msg.right_valid:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = msg.right_pose
            self._latest_tcp_pose = ps

    def _on_get_robot_state(self, request, response):
        if self._latest_tcp_pose is None:
            response.success = False
            response.reason = 'no robot pose received yet'
            response.state.cartesian_valid = False
            return response

        response.success = True
        response.reason = ''
        response.state.cartesian_valid = True
        response.state.tcp_pose = self._latest_tcp_pose
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ZmqMotionBackendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
