#!/usr/bin/env python3
"""ROS 2 ChArUco camera intrinsic calibration node."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def _stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class CharucoCameraCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("charuco_camera_calibration")

        self._declare_parameters()

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        self.show_preview = bool(self.get_parameter("show_preview").value)
        self.preview_max_width = int(self.get_parameter("preview_max_width").value)
        self.process_interval_s = float(self.get_parameter("process_interval_s").value)
        self.min_frames = int(self.get_parameter("min_frames").value)
        self.squares_x = int(self.get_parameter("squares_x").value)
        self.squares_y = int(self.get_parameter("squares_y").value)
        self.square_length_m = float(self.get_parameter("square_length_m").value)
        self.marker_length_m = float(self.get_parameter("marker_length_m").value)
        self.aruco_dictionary_name = str(
            self.get_parameter("aruco_dictionary").value
        ).strip()

        self._validate_parameters()

        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.aruco_dictionary_name)
        )

        if hasattr(cv2.aruco, "CharucoBoard_create"):
            self.charuco_board = cv2.aruco.CharucoBoard_create(
                self.squares_x, self.squares_y,
                self.square_length_m, self.marker_length_m, aruco_dict,
            )
        else:
            try:
                self.charuco_board = cv2.aruco.CharucoBoard(
                    (self.squares_x, self.squares_y),
                    self.square_length_m, self.marker_length_m, aruco_dict,
                )
            except TypeError:
                self.charuco_board = cv2.aruco.CharucoBoard(
                    self.squares_x, self.squares_y,
                    self.square_length_m, self.marker_length_m, aruco_dict,
                )

        if hasattr(cv2.aruco, "DetectorParameters_create"):
            detector_params = cv2.aruco.DetectorParameters_create()
        else:
            detector_params = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            self._use_detector_obj = True
        else:
            self.detector_params = detector_params
            self._use_detector_obj = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"camera_calibration_{timestamp}"
        self.images_dir = self.session_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.all_charuco_corners: list[np.ndarray] = []
        self.all_charuco_ids: list[np.ndarray] = []
        self.frame_size: tuple[int, int] | None = None
        self.last_process_time_s = 0.0

        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self._camera_info_callback, qos_profile_sensor_data,
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic,
            self._image_callback, qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"ChArUco calibrator ready: "
            f"image={self.image_topic}, "
            f"board={self.squares_x}x{self.squares_y}, "
            f"square={self.square_length_m:.4f}m, "
            f"marker={self.marker_length_m:.4f}m, "
            f"dict={self.aruco_dictionary_name}"
        )
        self.get_logger().info(
            "Controls: [s]/[space]=capture, [c]=calibrate, [r]=reset, [q]=quit"
        )

    def destroy_node(self) -> bool:
        if self.show_preview:
            cv2.destroyAllWindows()
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("output_dir",
                               str(Path.home() / "camera_calibration"))
        self.declare_parameter("show_preview", True)
        self.declare_parameter("preview_max_width", 1280)
        self.declare_parameter("process_interval_s", 0.15)
        self.declare_parameter("min_frames", 20)
        self.declare_parameter("squares_x", 7)
        self.declare_parameter("squares_y", 5)
        self.declare_parameter("square_length_m", 0.0495)
        self.declare_parameter("marker_length_m", 0.0295)
        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")

    def _validate_parameters(self) -> None:
        if self.min_frames < 5:
            raise ValueError("min_frames must be >= 5")
        if self.square_length_m <= 0 or self.marker_length_m <= 0:
            raise ValueError("square_length_m / marker_length_m must be positive")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        pass  # ChArUco calibration doesn't need prior intrinsics

    def _image_callback(self, msg: Image) -> None:
        now_s = time.monotonic()
        if now_s - self.last_process_time_s < self.process_interval_s:
            return
        self.last_process_time_s = now_s

        try:
            bgr = self._image_msg_to_bgr(msg)
        except ValueError:
            return

        if self.frame_size is None:
            self.frame_size = (msg.width, msg.height)

        display = bgr.copy()
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            self._detect_charuco(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

        detection_ok = charuco_corners is not None and len(charuco_corners) >= 4

        if detection_ok:
            cv2.aruco.drawDetectedCornersCharuco(
                display, charuco_corners, charuco_ids
            )
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        self._draw_overlay(display, detection_ok, charuco_corners)
        self._show_preview(display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        if key in (ord("q"), 27):
            self.get_logger().info("User exit.")
            if rclpy.ok():
                rclpy.shutdown()
            return
        if key in (ord("r"),):
            self.all_charuco_corners.clear()
            self.all_charuco_ids.clear()
            self.get_logger().info("Cleared all captured frames.")
            return
        if key in (ord("s"), ord(" ")):
            if detection_ok:
                self.all_charuco_corners.append(charuco_corners)
                self.all_charuco_ids.append(charuco_ids)
                self._save_frame_image(bgr, len(self.all_charuco_corners) - 1)
                self.get_logger().info(
                    f"Captured frame #{len(self.all_charuco_corners)}: "
                    f"{len(charuco_corners)} charuco corners"
                )
            else:
                self.get_logger().warn("No valid ChArUco board in current frame.")
            return
        if key in (ord("c"),):
            self._run_calibration()
            return

    def _detect_charuco(self, gray):
        if self._use_detector_obj:
            marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray, getattr(cv2.aruco, self.aruco_dictionary_name),
                parameters=self.detector_params,
            )

        if marker_ids is None or len(marker_ids) == 0:
            return None, None, [], []

        count, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.charuco_board
            )

        return charuco_corners, charuco_ids, marker_corners, marker_ids

    def _draw_overlay(self, image, detection_ok, charuco_corners):
        lines = [
            f"Frames: {len(self.all_charuco_corners)} (min={self.min_frames})",
            f"Detection: {'OK' if detection_ok else 'None'} "
            f"({len(charuco_corners) if charuco_corners is not None else 0} corners)",
            "Keys: s=save  c=calibrate  r=reset  q=quit",
        ]
        for i, line in enumerate(lines):
            y = 28 + i * 24
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (30, 30, 30), 1, cv2.LINE_AA)

    def _show_preview(self, image):
        if not self.show_preview:
            return
        if image.shape[1] > self.preview_max_width:
            scale = self.preview_max_width / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        cv2.imshow("ChArUco Camera Calibration", image)

    def _save_frame_image(self, bgr, index):
        path = self.images_dir / f"frame_{index:03d}.png"
        cv2.imwrite(str(path), bgr)

    def _image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == "bgr8":
            arr = data.reshape(msg.height, msg.step)[:, :msg.width * 3]
            return arr.reshape(msg.height, msg.width, 3).copy()
        if encoding == "rgb8":
            arr = data.reshape(msg.height, msg.step)[:, :msg.width * 3]
            return cv2.cvtColor(arr.reshape(msg.height, msg.width, 3),
                                cv2.COLOR_RGB2BGR)
        if encoding in ("mono8", "8uc1"):
            arr = data.reshape(msg.height, msg.step)[:, :msg.width]
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        raise ValueError(encoding)

    def _run_calibration(self) -> None:
        n = len(self.all_charuco_corners)
        if n < self.min_frames:
            self.get_logger().warn(
                f"Need >= {self.min_frames} frames, have {n}"
            )
            return

        if self.frame_size is None:
            self.get_logger().error("No frame size recorded.")
            return

        self.get_logger().info(f"Running calibration with {n} frames...")

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = \
            cv2.aruco.calibrateCameraCharuco(
                charucoCorners=self.all_charuco_corners,
                charucoIds=self.all_charuco_ids,
                board=self.charuco_board,
                imageSize=self.frame_size,
                cameraMatrix=None,
                distCoeffs=None,
            )

        self._write_result(ret, camera_matrix, dist_coeffs)
        self._print_result(ret, camera_matrix, dist_coeffs)

    def _print_result(self, ret, K, D):
        logger = self.get_logger()
        logger.info("=" * 50)
        logger.info(f"Reprojection error: {ret:.4f} px")
        logger.info(f"fx={K[0,0]:.4f}  fy={K[1,1]:.4f}")
        logger.info(f"cx={K[0,2]:.4f}  cy={K[1,2]:.4f}")
        logger.info(f"k1={D[0,0]:.6f}  k2={D[0,1]:.6f}  k3={D[0,4]:.6f}")
        logger.info(f"p1={D[0,2]:.6f}  p2={D[0,3]:.6f}")
        logger.info(f"Saved to: {self.session_dir / 'camera_intrinsics.json'}")

    def _write_result(self, ret, K, D):
        w, h = self.frame_size
        result = {
            "created_at": datetime.now().isoformat(),
            "image_topic": self.image_topic,
            "frame_size": [w, h],
            "reprojection_error_px": float(ret),
            "camera_matrix": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "k": [float(K[0, 0]), 0.0, float(K[0, 2]),
                      0.0, float(K[1, 1]), float(K[1, 2]),
                      0.0, 0.0, 1.0],
            },
            "distortion_coefficients": {
                "model": "plumb_bob",
                "d": [float(D[0, i]) for i in range(5)],
            },
            "board": {
                "type": "charuco",
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length_m": self.square_length_m,
                "marker_length_m": self.marker_length_m,
                "aruco_dictionary": self.aruco_dictionary_name,
            },
            "num_frames": len(self.all_charuco_corners),
        }

        json_path = self.session_dir / "camera_intrinsics.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)

        txt_path = self.session_dir / "camera_intrinsics.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(f"# ChArUco Camera Calibration\n")
            f.write(f"# Date: {result['created_at']}\n")
            f.write(f"# Resolution: {w}x{h}\n")
            f.write(f"# Reprojection error: {ret:.4f} px\n\n")
            f.write(f"[camera_matrix]\n")
            f.write(f"fx = {K[0,0]:.6f}\n")
            f.write(f"fy = {K[1,1]:.6f}\n")
            f.write(f"cx = {K[0,2]:.6f}\n")
            f.write(f"cy = {K[1,2]:.6f}\n\n")
            f.write(f"[distortion_coefficients]\n")
            f.write(f"k1 = {D[0,0]:.6f}\n")
            f.write(f"k2 = {D[0,1]:.6f}\n")
            f.write(f"p1 = {D[0,2]:.6f}\n")
            f.write(f"p2 = {D[0,3]:.6f}\n")
            f.write(f"k3 = {D[0,4]:.6f}\n")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CharucoCameraCalibrationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
