#!/usr/bin/env python3
"""TCP (Tool Center Point) calibration node.

Align gripper tip to 4 ArUco markers on a board to solve for the fixed
transform from robot end-effector flange to the actual gripper centre.

Workflow:
  1. Press 'd' to detect markers and record their base_link positions (via camera).
  2. Jog the robot so the gripper tip touches each marker centre (Z-axis ⟂ board).
  3. Press 's' to capture the current end-effector pose for the active marker.
  4. After 4 points, press 'c' to compute and save the TCP transform.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from common.msg import OculusControllers
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def _stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def _pose_to_transform(pose: Pose) -> np.ndarray:
    q = np.array([pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    t = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    return _make_transform(R, t)


def _orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    R2 = u @ vt
    if np.linalg.det(R2) < 0:
        u[:, -1] *= -1
        R2 = u @ vt
    return R2


def _rotation_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    R = _orthonormalize_rotation(R)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0,      -v[2],   v[1]],
        [v[2],    0,     -v[0]],
        [-v[1],   v[0],   0],
    ], dtype=np.float64)


# ── Marker layout on the paper (same as placeslot convention) ──────────────
# ID1: top-left,  ID2: top-right,  ID3: bottom-left,  ID4: bottom-right
# Origin at paper centre, Z=0 on the paper surface.
def _default_marker_positions_mm(paper_w_mm: float, paper_h_mm: float):
    hw = paper_w_mm / 2.0
    hh = paper_h_mm / 2.0
    return {
        1: np.array([-hw, -hh, 0.0]),
        2: np.array([ hw, -hh, 0.0]),
        3: np.array([-hw,  hh, 0.0]),
        4: np.array([ hw,  hh, 0.0]),
    }


class TcpCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__('tcp_calibration')

        # Parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('endpose_sub_topic', '/endpose_states_double_arm')
        self.declare_parameter('arm', 'left')
        self.declare_parameter('hand_eye_calibration_file', '')
        self.declare_parameter('custom_intrinsics_path', '')
        self.declare_parameter('aruco_dictionary', 'DICT_4X4_50')
        self.declare_parameter('output_dir', str(Path.home() / 'tcp_calibration'))
        self.declare_parameter('show_preview', True)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.endpose_topic = str(self.get_parameter('endpose_sub_topic').value).strip()
        self.arm = str(self.get_parameter('arm').value).strip().lower()
        self.hand_eye_file = str(self.get_parameter('hand_eye_calibration_file').value).strip()
        self.custom_intrinsics_path = str(self.get_parameter('custom_intrinsics_path').value).strip()
        self.aruco_dict_name = str(self.get_parameter('aruco_dictionary').value).strip()
        self.marker_size_mm = self._declare_float_param('marker_size_mm', 40.0)
        self.paper_w_mm = self._declare_float_param('paper_width_mm', 134.0)
        self.paper_h_mm = self._declare_float_param('paper_height_mm', 237.0)
        self.output_root = Path(str(self.get_parameter('output_dir').value)).expanduser()
        self.show_preview = bool(self.get_parameter('show_preview').value)

        if self.arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")

        # Marker positions
        self.marker_positions_mm = _default_marker_positions_mm(
            self.paper_w_mm, self.paper_h_mm)

        # ArUco setup
        dict_id = int(getattr(cv2.aruco, self.aruco_dict_name))
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters') \
            else cv2.aruco.DetectorParameters_create()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params) \
            if hasattr(cv2.aruco, 'ArucoDetector') else None

        # Camera intrinsics
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self._custom_intrinsics_loaded = False
        self._load_custom_intrinsics_if_needed()

        # Hand-eye calibration
        self.T_base_cam: np.ndarray | None = None  # base_T_camera (4x4)
        self._load_hand_eye_calibration()

        # State
        self.marker_positions_base: dict[int, np.ndarray] = {}  # ID → 3D pos in base_link
        self.board_normal_base: np.ndarray | None = None  # board normal in base_link
        self.captures: list[dict[str, Any]] = []  # collected flange poses per marker
        self.current_target_idx: int = 0
        self.detection_done: bool = False
        self.calibration_done: bool = False

        self.pose_buffer: list[tuple[float, str, np.ndarray]] = []  # (stamp, frame_id, T)
        self.latest_bgr: np.ndarray | None = None
        self.latest_rvec: np.ndarray | None = None
        self.latest_tvec: np.ndarray | None = None

        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_root / f'tcp_calibration_{timestamp}'
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, qos_profile_sensor_data)
        self.endpose_sub = self.create_subscription(
            OculusControllers, self.endpose_topic, self._endpose_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f'TCP calibrator started | arm={self.arm} | '
            f'paper={self.paper_w_mm:.0f}x{self.paper_h_mm:.0f}mm | '
            f'marker={self.marker_size_mm:.0f}mm | dict={self.aruco_dict_name}'
        )
        self._print_help()

    def _declare_float_param(self, name: str, default: float) -> float:
        """Declare & read a float param, tolerating int CLI overrides."""
        try:
            self.declare_parameter(name, float(default))
            return float(self.get_parameter(name).value)
        except rclpy.exceptions.InvalidParameterTypeException:
            # ROS2 Humble parsed CLI override as int; extract manually
            return self._get_float_from_raw_args(name, default)

    @staticmethod
    def _get_float_from_raw_args(name: str, default: float) -> float:
        import sys
        args = list(sys.argv)
        i = 0
        while i < len(args):
            if args[i] in ('--ros-args', '-p'):
                i += 1
                while i < len(args) and not args[i].startswith('--') and not args[i] == '-p':
                    if ':=' in args[i]:
                        k, v = args[i].split(':=', 1)
                        if k == name:
                            return float(v)
                    i += 1
                continue
            i += 1
        return float(default)

    # ── Parameter loading ──────────────────────────────────────────────────

    def _load_custom_intrinsics_if_needed(self) -> None:
        if not self.custom_intrinsics_path:
            return
        path = Path(self.custom_intrinsics_path).expanduser()
        if not path.is_file():
            self.get_logger().error(f'Intrinsics file not found: {path}')
            return
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        cm = data['camera_matrix']
        if 'k' in cm:
            k = np.asarray(cm['k'], dtype=np.float64).reshape(3, 3)
        else:
            k = np.array([[cm['fx'], 0, cm['cx']],
                          [0, cm['fy'], cm['cy']],
                          [0, 0, 1]], dtype=np.float64)
        dc = data['distortion_coefficients']
        if 'd' in dc:
            d = np.asarray(dc['d'], dtype=np.float64).reshape(-1)
        else:
            d = np.array([dc['k1'], dc['k2'], dc['p1'], dc['p2'], dc['k3']], dtype=np.float64)
        self.camera_matrix = k
        self.dist_coeffs = d
        self._custom_intrinsics_loaded = True
        self.get_logger().info(f'Loaded intrinsics: fx={k[0,0]:.1f} fy={k[1,1]:.1f}')

    def _load_hand_eye_calibration(self) -> None:
        if not self.hand_eye_file:
            return
        path = Path(self.hand_eye_file).expanduser()
        if not path.is_file():
            self.get_logger().error(f'Hand-eye file not found: {path}')
            return
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        matrix = data.get('base_T_camera', {}).get('matrix')
        if matrix is None:
            self.get_logger().error('Missing base_T_camera.matrix in hand-eye file')
            return
        self.T_base_cam = np.asarray(matrix, dtype=np.float64)
        self.get_logger().info(f'Loaded hand-eye calibration from {path}')

    # ── ROS callbacks ──────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._custom_intrinsics_loaded:
            return
        if len(msg.k) != 9:
            return
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64).reshape(-1)

    def _endpose_cb(self, msg: OculusControllers) -> None:
        use_left = self.arm == 'left'
        pose_valid = bool(msg.left_valid) if use_left else bool(msg.right_valid)
        if not pose_valid:
            return
        pose = msg.left_pose if use_left else msg.right_pose
        try:
            T = _pose_to_transform(pose)
        except ValueError:
            return
        stamp = _stamp_to_seconds(msg.header.stamp)
        frame_id = str(msg.header.frame_id)
        self.pose_buffer.append((stamp, frame_id, T))
        if len(self.pose_buffer) > 400:
            self.pose_buffer.pop(0)

    def _image_cb(self, msg: Image) -> None:
        encoding = msg.encoding.lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if encoding == 'bgr8':
            self.latest_bgr = raw.reshape(msg.height, msg.step)[:, :msg.width * 3] \
                .reshape(msg.height, msg.width, 3).copy()
        elif encoding == 'rgb8':
            arr = raw.reshape(msg.height, msg.step)[:, :msg.width * 3] \
                .reshape(msg.height, msg.width, 3)
            self.latest_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif encoding in ('mono8', '8uc1'):
            arr = raw.reshape(msg.height, msg.step)[:, :msg.width]
            self.latest_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    # ── ArUco detection ────────────────────────────────────────────────────

    def _detect_markers(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict,
                parameters=cv2.aruco.DetectorParameters_create())
        return corners, ids

    def _compute_marker_positions_base(self) -> bool:
        """Detect markers and compute their 3D positions in base_link."""
        if self.latest_bgr is None:
            self.get_logger().warn('No image received yet.')
            return False
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn('No camera intrinsics available.')
            return False
        if self.T_base_cam is None:
            self.get_logger().warn('No hand-eye calibration loaded.')
            return False

        corners, ids = self._detect_markers(self.latest_bgr)
        if ids is None or len(ids) < 4:
            self.get_logger().warn(
                f'Need 4 markers, detected {len(ids) if ids is not None else 0}.')
            return False

        # Collect marker corners for PnP (all 4 markers together)
        obj_points = []
        img_points = []
        for i, mid in enumerate(ids.flatten()):
            if mid not in self.marker_positions_mm:
                continue
            pos_mm = self.marker_positions_mm[mid]
            # Each marker's 4 corners (marker_size × marker_size, centred on marker position)
            half = self.marker_size_mm / 2.0
            obj_points.append([pos_mm[0] - half, pos_mm[1] - half, 0.0])
            obj_points.append([pos_mm[0] + half, pos_mm[1] - half, 0.0])
            obj_points.append([pos_mm[0] + half, pos_mm[1] + half, 0.0])
            obj_points.append([pos_mm[0] - half, pos_mm[1] + half, 0.0])
            c = corners[i].reshape(4, 2)
            img_points.extend(c.astype(np.float32))

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # Scale from mm to m
        obj_points_m = obj_points / 1000.0

        success, rvec, tvec = cv2.solvePnP(
            obj_points_m, img_points, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            self.get_logger().warn('PnP failed.')
            return False

        self.latest_rvec = rvec
        self.latest_tvec = tvec

        R_board_cam, _ = cv2.Rodrigues(rvec)          # board → camera
        t_board_cam = tvec.reshape(3)                  # board origin in camera frame

        # Board normal in camera frame, then transform to base_link
        n_board_cam = R_board_cam @ np.array([0, 0, 1], dtype=np.float64)
        R_base_cam = self.T_base_cam[:3, :3]
        self.board_normal_base = R_base_cam @ n_board_cam
        self.board_normal_base /= np.linalg.norm(self.board_normal_base)

        # Compute each marker's 3D position in base_link
        self.marker_positions_base.clear()
        for mid in [1, 2, 3, 4]:
            pos_board_m = self.marker_positions_mm[mid] / 1000.0  # in board frame (m)
            pos_cam = R_board_cam @ pos_board_m + t_board_cam
            pos_base_hom = self.T_base_cam @ np.array(
                [pos_cam[0], pos_cam[1], pos_cam[2], 1.0], dtype=np.float64)
            self.marker_positions_base[mid] = pos_base_hom[:3]

        self.detection_done = True
        self.current_target_idx = 0
        self.get_logger().info(
            f'Detected 4 markers in base_link: '
            + ' | '.join(
                f'#{mid}=({self.marker_positions_base[mid][0]:.4f}, '
                f'{self.marker_positions_base[mid][1]:.4f}, '
                f'{self.marker_positions_base[mid][2]:.4f})'
                for mid in [1, 2, 3, 4]))
        return True

    # ── Pose capture ───────────────────────────────────────────────────────

    def _get_latest_pose(self) -> np.ndarray | None:
        if not self.pose_buffer:
            return None
        return self.pose_buffer[-1][2]

    def _capture_current_pose(self) -> bool:
        if not self.detection_done:
            self.get_logger().warn('Run detection first (press d).')
            return False

        T = self._get_latest_pose()
        if T is None:
            self.get_logger().warn('No end-effector pose received yet.')
            return False

        mid = self.current_target_idx + 1  # marker IDs are 1-4
        if mid not in self.marker_positions_base:
            self.get_logger().warn(f'Marker #{mid} position unknown.')
            return False

        self.captures.append({
            'marker_id': mid,
            'marker_base_m': self.marker_positions_base[mid].tolist(),
            'T_base_flange': T,
        })
        self.get_logger().info(
            f'Captured marker #{mid} ({len(self.captures)}/4): '
            f'flange_t=[{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}]')

        self.current_target_idx = min(self.current_target_idx + 1, 3)
        return True

    # ── TCP computation ────────────────────────────────────────────────────

    def _compute_tcp(self) -> tuple[np.ndarray, np.ndarray, list[float]] | None:
        """Solve for T_flange_tip (translation + rotation).

        Returns (t_tcp, R_tcp, residuals_mm).
        """
        if len(self.captures) < 4:
            self.get_logger().warn(f'Need 4 captures, got {len(self.captures)}.')
            return None

        # ── Translation ──
        # For each capture:  R_flange_i * t_tcp = P_marker_i_base - t_flange_i
        # Stack into A * x = b
        A_blocks, b_blocks = [], []
        for cap in self.captures:
            T_flange = cap['T_base_flange']
            R_flange = T_flange[:3, :3]
            t_flange = T_flange[:3, 3]
            P_marker = np.array(cap['marker_base_m'], dtype=np.float64)
            A_blocks.append(R_flange)
            b_blocks.append(P_marker - t_flange)

        A = np.vstack(A_blocks)
        b = np.concatenate(b_blocks)
        t_tcp, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        residuals_mm = [float(r) * 1000.0 for r in residuals] if len(residuals) > 0 else []

        # ── Rotation ──
        # Constraint: R_flange * R_tcp * e_z = -n_board (gripper Z anti-parallel to board normal)
        if self.board_normal_base is not None:
            n_board = self.board_normal_base
            v_estimates = []
            for cap in self.captures:
                R_flange = cap['T_base_flange'][:3, :3]
                v_i = R_flange.T @ (-n_board)
                v_estimates.append(v_i)
            v = np.mean(v_estimates, axis=0)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                self.get_logger().warn('Degenerate rotation: board normal undefined.')
                R_tcp = np.eye(3, dtype=np.float64)
            else:
                v = v / v_norm
                e_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                axis = np.cross(e_z, v)
                sin_a = np.linalg.norm(axis)
                cos_a = np.dot(e_z, v)
                if sin_a < 1e-12:
                    R_tcp = np.eye(3, dtype=np.float64)
                else:
                    axis /= sin_a
                    K = _skew(axis)
                    R_tcp = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
        else:
            R_tcp = np.eye(3, dtype=np.float64)

        return t_tcp, R_tcp, residuals_mm

    # ── Display ─────────────────────────────────────────────────────────────

    def _overlay_text(self, img: np.ndarray, text: str, y: int, color=(255, 255, 255)):
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    def _draw_status(self, img: np.ndarray) -> np.ndarray:
        disp = img.copy()
        y = 26
        intr_ok = self.camera_matrix is not None
        he_ok = self.T_base_cam is not None
        self._overlay_text(disp, f'intrinsics: {"OK" if intr_ok else "waiting"}  '
                           f'hand_eye: {"OK" if he_ok else "no"}  '
                           f'captures: {len(self.captures)}/4', y); y += 24

        if self.detection_done:
            self._overlay_text(disp, 'MARKERS DETECTED (in base_link):', y, (0, 255, 200)); y += 22
            for mid in [1, 2, 3, 4]:
                if mid in self.marker_positions_base:
                    p = self.marker_positions_base[mid]
                    marker = ' <-- TARGET' if mid == self.current_target_idx + 1 else ''
                    self._overlay_text(disp,
                        f'  #{mid}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}){marker}', y,
                        (0, 255, 0) if mid == self.current_target_idx + 1 else (200, 200, 200))
                    y += 20
        else:
            self._overlay_text(disp, 'Press [d] to detect markers', y, (0, 255, 255)); y += 24

        if self.calibration_done:
            self._overlay_text(disp, 'TCP CALIBRATION COMPLETE', y, (0, 255, 0)); y += 24

        y += 8
        self._overlay_text(disp,
            '[d]detect  [s]capture pose  [c]compute  [r]reset  [q]quit', y, (180, 180, 180))
        return disp

    def _print_help(self) -> None:
        self.get_logger().info(
            'Controls: [d]=detect markers  [s]=capture flange pose  '
            '[c]=compute TCP  [r]=reset  [q]=quit')

    # ── Main loop ──────────────────────────────────────────────────────────

    def spin(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.latest_bgr is None:
                continue

            display = self._draw_status(self.latest_bgr)
            if self.show_preview:
                h, w = display.shape[:2]
                if w > 1280:
                    scale = 1280.0 / w
                    display = cv2.resize(display, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA)
                cv2.imshow('TCP Calibration', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.get_logger().info('Quit.')
                break
            elif key == ord('d'):
                self._compute_marker_positions_base()
            elif key == ord('s'):
                self._capture_current_pose()
            elif key == ord('c'):
                self._run_calibration()
            elif key == ord('r'):
                self._reset()

        cv2.destroyAllWindows()

    def _reset(self) -> None:
        self.marker_positions_base.clear()
        self.board_normal_base = None
        self.captures.clear()
        self.current_target_idx = 0
        self.detection_done = False
        self.calibration_done = False
        self.get_logger().info('Reset.')

    def _run_calibration(self) -> None:
        result = self._compute_tcp()
        if result is None:
            return
        t_tcp, R_tcp, residuals_mm = result
        T_tcp = _make_transform(R_tcp, t_tcp)
        q = _rotation_to_quaternion_xyzw(R_tcp)

        # Per-sample residuals
        per_sample = []
        for cap in self.captures:
            T_flange = cap['T_base_flange']
            R_flange = T_flange[:3, :3]
            t_flange = T_flange[:3, 3]
            P_marker = np.array(cap['marker_base_m'], dtype=np.float64)
            pred = R_flange @ t_tcp + t_flange
            err_mm = float(np.linalg.norm(pred - P_marker) * 1000.0)
            per_sample.append({'marker_id': cap['marker_id'], 'error_mm': err_mm})

        mean_err = float(np.mean([s['error_mm'] for s in per_sample])) if per_sample else 0.0

        self.get_logger().info('=' * 55)
        self.get_logger().info(f'TCP CALIBRATION RESULT  ({len(self.captures)} points)')
        self.get_logger().info(f'  Translation: [{t_tcp[0]:.6f}, {t_tcp[1]:.6f}, {t_tcp[2]:.6f}] m')
        self.get_logger().info(f'  Quaternion:  [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]')
        self.get_logger().info(f'  Mean error:  {mean_err:.3f} mm')
        for s in per_sample:
            self.get_logger().info(
                f'    marker #{s["marker_id"]}: {s["error_mm"]:.3f} mm')
        self.get_logger().info(
            '  static_transform: '
            f'ros2 run tf2_ros static_transform_publisher '
            f'{t_tcp[0]:.6f} {t_tcp[1]:.6f} {t_tcp[2]:.6f} '
            f'{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} '
            f'flange gripper_tip')
        self.get_logger().info('=' * 55)

        # Save JSON
        summary = {
            'created_at': datetime.now().isoformat(),
            'arm': self.arm,
            'paper_width_mm': self.paper_w_mm,
            'paper_height_mm': self.paper_h_mm,
            'marker_size_mm': self.marker_size_mm,
            'num_captures': len(self.captures),
            'mean_translation_error_mm': mean_err,
            'per_sample_errors_mm': per_sample,
            'translation_m': [float(v) for v in t_tcp],
            'quaternion_xyzw': [float(v) for v in q],
            'matrix': [[float(v) for v in row] for row in T_tcp.tolist()],
            'static_transform_publisher':
                f'ros2 run tf2_ros static_transform_publisher '
                f'{t_tcp[0]:.6f} {t_tcp[1]:.6f} {t_tcp[2]:.6f} '
                f'{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} '
                f'flange gripper_tip',
        }
        result_path = self.session_dir / 'tcp_calibration_result.json'
        with result_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f'Saved to {result_path}')
        self.calibration_done = True


def main(args=None):
    rclpy.init(args=args)
    node = TcpCalibrationNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
