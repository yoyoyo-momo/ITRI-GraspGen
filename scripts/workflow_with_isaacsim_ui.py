import os
import argparse
import logging
import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont
from threading import Thread, Event
from queue import Queue, Empty

import numpy as np
import cv2
from PIL import Image

from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from PointCloud_Generation.PC_transform import (
    silent_transform_multiple_obj_with_name_dict,
)
from common_utils import config, network_config
from common_utils.graspgen_utils import GraspGeneratorUI
from common_utils.actions_format_checker import is_actions_format_valid_v1028
from common_utils.movesets import act_with_name
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from common_utils.custom_logger import CustomFormatter
from common_utils.common_utils import save_json, create_obstacle_info

try:
    from scripts.teapot_handle_detector import TeapotHandleDetector
except ModuleNotFoundError:
    from teapot_handle_detector import TeapotHandleDetector


handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def _resolve_camera_source(source_text: str):
    source_text = str(source_text).strip()
    if source_text.isdigit():
        return int(source_text)
    return source_text


def _open_2d_camera(source_text: str, width: int, height: int):
    source = _resolve_camera_source(source_text)
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if not cap.isOpened():
        raise ValueError(f"Failed to open startup cup camera source: {source_text}")
    return cap


def parse_args():
    parser = argparse.ArgumentParser(description="Workflow UI for batch actions.")
    parser.add_argument(
        "--ckpt_dir",
        default=str(config.FOUNDATIONSTEREO_CHECKPOINT),
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--scale",
        default=1,
        type=float,
        help="downsize the image by scale, must be <=1",
    )
    parser.add_argument("--hiera", default=0, type=int, help="hierarchical inference")
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--out_dir", default="./output/", type=str, help="the directory to save results"
    )
    parser.add_argument(
        "--output-tag",
        default="",
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--erosion_iterations",
        type=int,
        default=1,
        help="Number of erosion iterations for the SAM mask.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=3.0,
        help="max depth for generating pointcloud",
    )
    parser.add_argument(
        "--transform-config",
        type=str,
        default="sim2.json",
        help="transform-config",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default=str(config.GRIPPER_CFG),
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.70,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=5,
        help="Number of top grasps to return when return_topk is True",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="decide if we need confirm for groundingDINO detect and grasp Generation",
    )
    parser.add_argument(
        "--save-fullact",
        action="store_true",
        help="save the fullact",
    )
    parser.add_argument(
        "--use-png",
        type=str,
        default="",
        help="Use exisiting images at sample_data/zed_images instead of the real zed camera",
    )
    parser.add_argument(
        "--startup-cup-auto",
        action="store_true",
        help="Run startup ROI cup detector and auto-select teapot action",
    )
    parser.add_argument(
        "--startup-cup-roi",
        type=str,
        default="",
        help="ROI for startup cup detector in x1,y1,x2,y2",
    )
    parser.add_argument(
        "--startup-cup-prompt",
        type=str,
        default="white cup .",
        help="GroundingDINO prompt for startup cup detector",
    )
    parser.add_argument(
        "--startup-cup-box-threshold",
        type=float,
        default=0.35,
        help="GroundingDINO box threshold for startup cup detector",
    )
    parser.add_argument(
        "--startup-cup-text-threshold",
        type=float,
        default=0.35,
        help="GroundingDINO text threshold for startup cup detector",
    )
    parser.add_argument(
        "--startup-cup-stable-frames",
        type=int,
        default=5,
        help="Stable frame count required before startup action trigger",
    )
    parser.add_argument(
        "--startup-cup-timeout-sec",
        type=float,
        default=0,
        help="Timeout seconds for startup cup detector (<=0 disables timeout)",
    )
    parser.add_argument(
        "--startup-cup-action-one",
        type=str,
        default="Grasp_teapot",
        help="Action name when 1 cup is detected in ROI",
    )
    parser.add_argument(
        "--startup-cup-action-two",
        type=str,
        default="Grasp_teapot_double",
        help="Action name when 2 cups are detected in ROI",
    )
    parser.add_argument(
        "--startup-cup-camera-source",
        type=str,
        default="2",
        help="2D camera source for startup cup detection (index like 0 or URL)",
    )
    parser.add_argument(
        "--startup-cup-camera-width",
        type=int,
        default=0,
        help="Optional frame width for startup cup camera (0 means default)",
    )
    parser.add_argument(
        "--startup-cup-camera-height",
        type=int,
        default=0,
        help="Optional frame height for startup cup camera (0 means default)",
    )
    parser.add_argument(
        "--startup-cup-pick-roi",
        action="store_true",
        help="Open interactive ROI picker and print x1,y1,x2,y2, then exit",
    )

    return parser.parse_args()


class WorkflowExecutor:
    # These actions are fully specified by joint/gripper commands and do not need
    # GraspGen pointcloud/object detection input.
    NO_POINTCLOUD_ACTIONS = {
        "open_grip",
        "joints_rad_pour_hotwater",
        "joints_rad_pour_tealeaf",
        "joints_rad_grasp_filter",
        "joints_rad_swap_teapot",
    }

    # These actions execute directly without grasp sampling.
    DIRECT_ACTIONS = {
        "open_grip",
        "joints_rad_pour_hotwater",
        "joints_rad_pour_tealeaf",
        "joints_rad_grasp_filter",
        "joints_rad_swap_teapot",
    }

    def __init__(self, args, project_root_dir, stop_event: Event, status_queue: Queue):
        self.args = args
        self.project_root_dir = project_root_dir
        self.stop_event = stop_event
        self.status_queue = status_queue

        self.sender = NonBlockingJSONSender(
            port=network_config.GRASPGEN_TO_ISAACSIM_PORT
        )
        self.receiver = NonBlockingJSONReceiver(
            port=network_config.ISAACSIM_TO_GRASPGEN_PORT
        )
        self.pc_generator = None
        self.grasp_generator = None
        self.startup_cup_cap = None

    def _get_pc_generator(self):
        if self.pc_generator is None:
            self._status("Initializing pointcloud/detector models")
            self.pc_generator = PointCloudGenerator(self.args)
        return self.pc_generator

    def _get_grasp_generator(self):
        if self.grasp_generator is None:
            self._status("Initializing grasp generator models")
            self.grasp_generator = GraspGeneratorUI(
                self.args.gripper_config,
                self.args.grasp_threshold,
                self.args.num_grasps,
                self.args.topk_num_grasps,
                not self.args.no_confirm,
            )
        return self.grasp_generator

    def _status(self, text):
        logger.info(text)
        self.status_queue.put(text)

    def _wait_until_response(self, response):
        while response is None:
            if self.stop_event.is_set():
                raise InterruptedError("stopped by user")
            response = self.receiver.capture_data()
            time.sleep(0.01)
        return response

    def _load_actions_json(self, name: str):
        filepath = os.path.join(self.project_root_dir, "actions", name + ".json")
        with open(filepath, "rb") as f:
            return json.load(f), filepath

    @staticmethod
    def _parse_roi_text(roi_text: str):
        if not isinstance(roi_text, str) or not roi_text.strip():
            return None
        parts = [p.strip() for p in roi_text.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "--startup-cup-roi must be x1,y1,x2,y2 (example: 420,180,980,760)"
            )
        x1, y1, x2, y2 = [int(v) for v in parts]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid --startup-cup-roi: x2>x1 and y2>y1 are required")
        return (x1, y1, x2, y2)

    @staticmethod
    def _center_in_roi(box, roi) -> bool:
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx1, ry1, rx2, ry2 = roi
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def _count_cups_in_roi(self, color_bgr, roi) -> int:
        pc_generator = self._get_pc_generator()
        boxes = pc_generator.groundingdino_predictor.predict_boxes(
            color_bgr,
            self.args.startup_cup_prompt,
            box_threshold=self.args.startup_cup_box_threshold,
            text_threshold=self.args.startup_cup_text_threshold,
        )
        cup_count = 0
        for box in boxes:
            phrase = str(box.phrase).lower()
            if "cup" not in phrase:
                continue
            if self._center_in_roi(box.box, roi):
                cup_count += 1
        return cup_count

    def _get_startup_cup_frame(self):
        if self.startup_cup_cap is None:
            self.startup_cup_cap = _open_2d_camera(
                self.args.startup_cup_camera_source,
                self.args.startup_cup_camera_width,
                self.args.startup_cup_camera_height,
            )
        ok, frame = self.startup_cup_cap.read()
        if not ok or frame is None:
            return None
        return frame

    def detect_startup_action(self) -> str | None:
        if not getattr(self.args, "startup_cup_auto", False):
            return None

        roi = self._parse_roi_text(self.args.startup_cup_roi)
        if roi is None:
            raise ValueError("--startup-cup-auto requires --startup-cup-roi")

        stable_required = max(1, int(self.args.startup_cup_stable_frames))
        timeout_arg = float(self.args.startup_cup_timeout_sec)
        timeout_sec = timeout_arg if timeout_arg > 0 else None
        start_ts = time.time()
        last_count = None
        stable_frames = 0
        timeout_text = f"{timeout_sec:.1f}s" if timeout_sec is not None else "disabled"

        self._status(
            "Startup cup detector enabled, ROI="
            f"{roi}, stable_frames={stable_required}, timeout={timeout_text}"
        )

        while True:
            if timeout_sec is not None and (time.time() - start_ts) >= timeout_sec:
                break
            if self.stop_event.is_set():
                raise InterruptedError("stopped by user")

            color_bgr = self._get_startup_cup_frame()
            if color_bgr is None:
                time.sleep(0.05)
                continue

            cup_count = self._count_cups_in_roi(color_bgr, roi)

            if cup_count == last_count and cup_count in (1, 2):
                stable_frames += 1
            else:
                last_count = cup_count
                stable_frames = 1

            self._status(
                "Startup cup detector: "
                f"cups_in_roi={cup_count}, stable={stable_frames}/{stable_required}"
            )

            if stable_frames >= stable_required and cup_count in (1, 2):
                action_name = (
                    self.args.startup_cup_action_one
                    if cup_count == 1
                    else self.args.startup_cup_action_two
                )
                self._load_actions_json(action_name)
                self._status(
                    f"Startup cup detector selected action: {action_name}.json"
                )
                return action_name

            time.sleep(0.05)

        raise ValueError(
            "Startup cup detector timeout before stable 1 or 2 cups in ROI was found"
        )

    def _resolve_detector_image_path(self, step: dict) -> Path | None:
        custom = step.get("detector_image")
        if isinstance(custom, str) and custom.strip():
            candidate = Path(custom)
            if not candidate.is_absolute():
                candidate = Path(self.project_root_dir) / custom
            if candidate.exists():
                return candidate

        if isinstance(self.args.use_png, str) and self.args.use_png.strip():
            left_png = (
                Path(self.project_root_dir)
                / "sample_data"
                / "zed_images"
                / self.args.use_png
                / "left.png"
            )
            if left_png.exists():
                return left_png

        return None

    def _detect_teapot_handle(self, step: dict, default_cfg: dict):
        image_path = self._resolve_detector_image_path(step)
        if image_path is None:
            self._status(
                "Teapot handle detector requested, but no image found; using unknown result"
            )
            return {
                "handle_side": "unknown",
                "confidence": 0.0,
                "image": "",
                "note": "No detector image available.",
            }

        cfg = default_cfg.copy()
        step_cfg = step.get("detector")
        if isinstance(step_cfg, dict):
            cfg.update(step_cfg)

        detector = TeapotHandleDetector(
            target_max_dim=int(cfg.get("target_max_dim", 1400)),
            auto_refine=bool(cfg.get("auto_refine", False)),
            refine_confidence=float(cfg.get("refine_confidence", 0.6)),
        )

        rgb = np.array(Image.open(image_path).convert("RGB"))
        crop_rgb = rgb
        offset_x, offset_y = 0, 0
        dino_bbox = None
        dino_phrase = ""
        dino_logit = 0.0

        dino_prompt = str(cfg.get("dino_prompt", "teapot ."))
        dino_keyword = str(cfg.get("dino_teapot_keyword", "teapot")).lower()
        dino_box_threshold = float(cfg.get("dino_box_threshold", 0.35))
        dino_text_threshold = float(cfg.get("dino_text_threshold", 0.25))
        dino_padding_px = int(cfg.get("dino_padding_px", 24))

        try:
            bgr = rgb[:, :, ::-1]
            pc_generator = self._get_pc_generator()
            boxes = pc_generator.groundingdino_predictor.predict_boxes(
                bgr,
                dino_prompt,
                box_threshold=dino_box_threshold,
                text_threshold=dino_text_threshold,
            )
            if len(boxes) > 0:

                def _box_score(box):
                    phrase = str(box.phrase).lower()
                    matched = 1 if dino_keyword in phrase else 0
                    return matched, float(box.logits)

                best_box = max(boxes, key=_box_score)
                dino_phrase = str(best_box.phrase)
                dino_logit = float(best_box.logits)
                h, w = rgb.shape[:2]
                x1, y1, x2, y2 = [int(v) for v in best_box.box]
                x1 = max(0, x1 - dino_padding_px)
                y1 = max(0, y1 - dino_padding_px)
                x2 = min(w, x2 + dino_padding_px)
                y2 = min(h, y2 + dino_padding_px)
                if x2 > x1 and y2 > y1:
                    dino_bbox = [x1, y1, x2, y2]
                    offset_x, offset_y = x1, y1
                    crop_rgb = rgb[y1:y2, x1:x2]
                    self._status(
                        "GroundingDINO teapot bbox "
                        f"{dino_bbox} phrase={dino_phrase} score={dino_logit:.2f}"
                    )
            else:
                self._status(
                    "GroundingDINO found no teapot box; fallback to full image for handle detection"
                )
        except Exception as e:
            logger.exception(e)
            self._status(
                "GroundingDINO failed for teapot localization; fallback to full image"
            )

        result, _, _ = detector.detect(crop_rgb, image_name=image_path.name)
        if result.center_x is not None:
            result.center_x += offset_x
        if result.center_y is not None:
            result.center_y += offset_y

        self._status(
            f"Teapot handle: {result.handle_side} (conf={result.confidence:.2f}, note={result.note})"
        )
        return {
            "handle_side": result.handle_side,
            "confidence": float(result.confidence),
            "image": str(image_path),
            "dino_bbox": dino_bbox,
            "dino_phrase": dino_phrase,
            "dino_logit": dino_logit,
            "note": result.note,
        }

    @staticmethod
    def _should_run_extra_action(extra_rule: dict, handle_result: dict) -> bool:
        if not isinstance(extra_rule, dict):
            return False
        if not isinstance(handle_result, dict):
            return False

        handle_side = handle_result.get("handle_side", "unknown")
        confidence = float(handle_result.get("confidence", 0.0))

        side_rule = extra_rule.get("when_handle_side")
        if side_rule is not None:
            if not isinstance(side_rule, list):
                return False
            if handle_side not in side_rule:
                return False

        low_conf = extra_rule.get("when_unknown_or_low_confidence")
        if low_conf is not None:
            if not (handle_side == "unknown" or confidence <= float(low_conf)):
                return False

        return True

    @staticmethod
    def _inject_qualifier_context(action: dict, qualifier_context: dict | None):
        if not isinstance(action, dict):
            return action
        if not isinstance(qualifier_context, dict):
            return action

        teapot_handle = qualifier_context.get("teapot_handle")
        if not isinstance(teapot_handle, dict):
            return action

        qualifier_kwargs = action.get("qualifier_kwargs", {})
        if not isinstance(qualifier_kwargs, dict):
            qualifier_kwargs = {}
        qualifier_kwargs = qualifier_kwargs.copy()

        if "handle_side" not in qualifier_kwargs:
            qualifier_kwargs["handle_side"] = teapot_handle.get(
                "handle_side", "unknown"
            )
        if "handle_confidence" not in qualifier_kwargs:
            qualifier_kwargs["handle_confidence"] = float(
                teapot_handle.get("confidence", 0.0)
            )

        merged_action = action.copy()
        merged_action["qualifier_kwargs"] = qualifier_kwargs
        return merged_action

    def run_plan_file(self, plan_name: str):
        if self.stop_event.is_set():
            raise InterruptedError("stopped by user")

        plan, plan_path = self._load_actions_json(plan_name)
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
            raise ValueError(f"bad plan file format: {plan_path}")

        self._status(f"Running plan file: {plan_name}.json")
        detector_defaults = plan.get("detector", {})
        if not isinstance(detector_defaults, dict):
            detector_defaults = {}

        context = {"teapot_handle": None}
        steps = plan["steps"]
        for idx, step in enumerate(steps, start=1):
            if self.stop_event.is_set():
                raise InterruptedError("stopped by user")
            if not isinstance(step, dict):
                raise ValueError(f"Invalid step at index {idx - 1} in {plan_path}")

            action_name = step.get("action_name")
            if not isinstance(action_name, str) or not action_name.strip():
                raise ValueError(f"Invalid action_name at step {idx} in {plan_path}")

            self._status(f"[Plan {idx}/{len(steps)}] {action_name}.json")
            requires_handle_detector = bool(
                step.get("requires_teapot_handle_detector", False)
            )
            if requires_handle_detector:
                context["teapot_handle"] = self._detect_teapot_handle(
                    step, detector_defaults
                )

            extra_rule = step.get("extra_action_if")
            if self._should_run_extra_action(extra_rule, context["teapot_handle"]):
                extra_action_name = extra_rule.get("action_name")
                if isinstance(extra_action_name, str) and extra_action_name.strip():
                    self._status(
                        "Condition matched; running extra action before main action: "
                        f"{extra_action_name}.json"
                    )
                    self.run_action_file(extra_action_name, qualifier_context=context)
                    if requires_handle_detector:
                        self._status("Re-checking teapot handle after extra action")
                        context["teapot_handle"] = self._detect_teapot_handle(
                            step, detector_defaults
                        )

            self.run_action_file(action_name, qualifier_context=context)

        self._status(f"Completed plan file: {plan_name}.json")

    def run_entry(self, name: str):
        data, _ = self._load_actions_json(name)
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            self.run_plan_file(name)
        else:
            self.run_action_file(name)

    def run_action_file(self, action_name: str, qualifier_context: dict | None = None):
        if self.stop_event.is_set():
            raise InterruptedError("stopped by user")

        self._status(f"Running action file: {action_name}.json")
        for _ in range(5):
            self.receiver.capture_data()

        actions, actions_filepath = self._load_actions_json(action_name)

        if not is_actions_format_valid_v1028(actions):
            raise ValueError(f"bad actions file format: {actions_filepath}")

        extra_obstacles = actions.get("extra_obstacles", {})
        requires_pointcloud = any(
            action.get("action") not in self.NO_POINTCLOUD_ACTIONS
            for action in actions["actions"]
        )

        if requires_pointcloud:
            track_names = list(actions["track"])
            detection_success = False
            scene_data = None
            while True:
                for _ in range(20):
                    if self.stop_event.is_set():
                        raise InterruptedError("stopped by user")
                    try:
                        blockages = actions.get("blockages")
                        valid_region = actions.get("valid_region")
                        pc_generator = self._get_pc_generator()
                        scene_data = pc_generator.generate_pointcloud(
                            track_names,
                            need_confirm=False,
                            blockages=blockages,
                            valid_region=valid_region,
                        )
                        detection_success = True
                        break
                    except ValueError as e:
                        logger.exception(f"{e}, try again")
                        time.sleep(0.1)
                        continue
                else:
                    self._status("Failed to detect using groundingDINO")
                if detection_success:
                    break
                time.sleep(0.3)

            scene_data = silent_transform_multiple_obj_with_name_dict(
                scene_data, self.args.transform_config
            )
            scene_data = create_obstacle_info(scene_data, extra_obstacles)
        else:
            self._status(
                "Skipping pointcloud generation (all actions are joint/gripper-only)"
            )
            scene_data = {"object_infos": {}, "obstacles": dict(extra_obstacles)}

        for action in actions["actions"]:
            if self.stop_event.is_set():
                raise InterruptedError("stopped by user")

            action = self._inject_qualifier_context(action, qualifier_context)

            if action["action"] in self.DIRECT_ACTIONS:
                while True:
                    full_acts = act_with_name(
                        action["action"],
                        None,
                        [None],
                        action["args"],
                        scene_data,
                    )
                    if self.args.save_fullact:
                        save_json("fullact", "fullact", full_acts)

                    response = self.receiver.capture_data()
                    if response is not None and response["message"] == "Abort":
                        raise InterruptedError(
                            "aborted by isaacsim, stop current action"
                        )

                    self.sender.send_data(full_acts)
                    response = self._wait_until_response(response)

                    if response["message"] == "Success":
                        break
                    if response["message"] == "Fail":
                        continue
                    if response["message"] == "Abort":
                        raise InterruptedError(
                            "aborted by isaacsim, stop current action"
                        )
            else:
                while True:
                    grasp_generator = self._get_grasp_generator()
                    grasps = grasp_generator.generate_grasp(scene_data, action)
                    full_acts = act_with_name(
                        action["action"],
                        action["target_name"],
                        grasps,
                        action["args"],
                        scene_data,
                    )
                    if self.args.save_fullact:
                        save_json("fullact", "fullact_", full_acts)

                    response = self.receiver.capture_data()
                    if response is not None and response["message"] == "Abort":
                        raise InterruptedError(
                            "aborted by isaacsim, stop current action"
                        )

                    self.sender.send_data(full_acts)
                    response = self._wait_until_response(response)

                    if response["message"] == "Success":
                        break
                    if response["message"] == "Fail":
                        continue
                    if response["message"] == "Abort":
                        raise InterruptedError(
                            "aborted by isaacsim, stop current action"
                        )

        self.sender.send_data(["EOF"])
        response = self._wait_until_response(self.receiver.capture_data())
        if response["message"] != "EOF and ROS2 Complete":
            if response["message"] == "Abort":
                raise InterruptedError("aborted by isaacsim, stop current action")
            raise ValueError(f"Unknown message {response['message']}")

        self._status(f"Completed action file: {action_name}.json")

    def close(self):
        self._status("turning off zed camera")
        try:
            if self.startup_cup_cap is not None:
                self.startup_cup_cap.release()
                self.startup_cup_cap = None
        except Exception as e:
            logger.exception(f"failed to release startup cup camera: {e}")

        try:
            if self.pc_generator is not None:
                self.pc_generator.close()
                time.sleep(0.5)
        except Exception as e:
            logger.exception(f"failed to close point cloud generator: {e}")

        try:
            self.sender.disconnect()
        except Exception as e:
            logger.exception(f"failed to disconnect sender socket: {e}")

        try:
            self.receiver.disconnect()
        except Exception as e:
            logger.exception(f"failed to disconnect receiver socket: {e}")

        self._status("workflow executor terminated")


class WorkflowUI:
    def __init__(self, root, args, project_root_dir):
        self.root = root
        self.args = args
        self.project_root_dir = project_root_dir
        self.actions_dir = os.path.join(project_root_dir, "actions")

        self.stop_event = Event()
        self.status_queue = Queue()
        self.worker = None

        self.root.title("IsaacSim Workflow UI")
        self.root.geometry("980x620")

        self._build_ui()
        self._refresh_available_actions()
        self._poll_status()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if getattr(self.args, "startup_cup_auto", False):
            self.root.after(200, self._start_run)

    def _build_ui(self):
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        top = tk.Frame(container)
        top.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(top)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        center = tk.Frame(top)
        center.pack(side=tk.LEFT, fill=tk.Y, padx=12)

        right = tk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="Available action files").pack(anchor="w")
        self.available_listbox = tk.Listbox(left, selectmode=tk.EXTENDED)
        self.available_listbox.pack(fill=tk.BOTH, expand=True)
        self.available_listbox.bind("<Double-Button-1>", lambda _: self._add_selected())

        tk.Label(right, text="Selected flow order").pack(anchor="w")
        self.selected_listbox = tk.Listbox(right, selectmode=tk.EXTENDED)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True)

        tk.Button(center, text=">>", width=14, command=self._add_selected).pack(pady=4)
        tk.Button(center, text="<<", width=14, command=self._remove_selected).pack(
            pady=4
        )
        tk.Button(center, text="Move Up", width=14, command=self._move_up).pack(pady=4)
        tk.Button(center, text="Move Down", width=14, command=self._move_down).pack(
            pady=4
        )
        tk.Button(
            center, text="Refresh", width=14, command=self._refresh_available_actions
        ).pack(pady=4)

        control = tk.Frame(container)
        control.pack(fill=tk.X, pady=(10, 4))

        self.run_button = tk.Button(
            control,
            text="Run Workflow",
            height=2,
            command=self._start_run,
            bg="#2e7d32",
            fg="white",
        )
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.stop_button = tk.Button(
            control,
            text="Stop",
            height=2,
            command=self._stop_run,
            state=tk.DISABLED,
            bg="#8d1b1b",
            fg="white",
        )
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(container, text="Status").pack(anchor="w", pady=(8, 2))
        self.status_text = tk.Text(container, height=12)
        self.status_text.pack(fill=tk.BOTH, expand=False)

    def _append_status(self, msg):
        self.status_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} | {msg}\n")
        self.status_text.see(tk.END)

    def _poll_status(self):
        try:
            while True:
                msg = self.status_queue.get_nowait()
                self._append_status(msg)
        except Empty:
            pass
        self.root.after(100, self._poll_status)

    def _refresh_available_actions(self):
        self.available_listbox.delete(0, tk.END)
        if not os.path.isdir(self.actions_dir):
            return
        action_files = sorted(
            [
                f[:-5]
                for f in os.listdir(self.actions_dir)
                if f.endswith(".json")
                and os.path.isfile(os.path.join(self.actions_dir, f))
            ]
        )
        for name in action_files:
            self.available_listbox.insert(tk.END, name)

    def _add_selected(self):
        for idx in self.available_listbox.curselection():
            name = self.available_listbox.get(idx)
            self.selected_listbox.insert(tk.END, name)

    def _remove_selected(self):
        indices = list(self.selected_listbox.curselection())
        indices.reverse()
        for idx in indices:
            self.selected_listbox.delete(idx)

    def _move_up(self):
        indices = list(self.selected_listbox.curselection())
        if not indices:
            return
        for idx in indices:
            if idx == 0:
                continue
            value = self.selected_listbox.get(idx)
            self.selected_listbox.delete(idx)
            self.selected_listbox.insert(idx - 1, value)
            self.selected_listbox.select_set(idx - 1)

    def _move_down(self):
        indices = list(self.selected_listbox.curselection())
        if not indices:
            return
        for idx in reversed(indices):
            if idx >= self.selected_listbox.size() - 1:
                continue
            value = self.selected_listbox.get(idx)
            self.selected_listbox.delete(idx)
            self.selected_listbox.insert(idx + 1, value)
            self.selected_listbox.select_set(idx + 1)

    def _set_running_ui(self, running: bool):
        self.run_button.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)

    def _start_run(self):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning("Busy", "Workflow is already running.")
            return

        selected = list(self.selected_listbox.get(0, tk.END))
        if len(selected) == 0 and not self.args.startup_cup_auto:
            messagebox.showwarning("No Actions", "Please add at least one action file.")
            return

        self.stop_event.clear()
        self._set_running_ui(True)
        self.worker = Thread(target=self._run_worker, args=(selected,), daemon=True)
        self.worker.start()

    def _stop_run(self):
        self.stop_event.set()
        self.status_queue.put("Stop requested")

    def _run_worker(self, selected_action_names):
        executor = None
        try:
            self.status_queue.put("Starting workflow")
            executor = WorkflowExecutor(
                self.args,
                self.project_root_dir,
                self.stop_event,
                self.status_queue,
            )
            startup_action = executor.detect_startup_action()
            if startup_action is not None:
                selected_action_names = [startup_action] + list(selected_action_names)

            for idx, action_name in enumerate(selected_action_names, start=1):
                if self.stop_event.is_set():
                    raise InterruptedError("stopped by user")
                self.status_queue.put(
                    f"[{idx}/{len(selected_action_names)}] {action_name}.json"
                )
                executor.run_entry(action_name)
            self.status_queue.put("All selected actions completed")
        except InterruptedError as e:
            self.status_queue.put(f"Workflow interrupted: {e}")
        except Exception as e:
            logger.exception(e)
            self.status_queue.put(f"Workflow failed: {e}")
            error_message = str(e)
            self.root.after(
                0,
                lambda msg=error_message: messagebox.showerror("Workflow Error", msg),
            )
        finally:
            if executor is not None:
                try:
                    executor.close()
                except Exception:
                    pass
            self.root.after(0, lambda: self._set_running_ui(False))

    def _on_close(self):
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self._append_status("Closing requested, waiting for worker to stop...")
            self.root.after(300, self.root.destroy)
            return
        self.root.destroy()


def main():
    args = parse_args()
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_file_dir)

    if args.startup_cup_pick_roi:
        cap = _open_2d_camera(
            args.startup_cup_camera_source,
            args.startup_cup_camera_width,
            args.startup_cup_camera_height,
        )
        try:
            frame = None
            for _ in range(30):
                ok, img = cap.read()
                if ok and img is not None:
                    frame = img
                    break
                time.sleep(0.03)

            if frame is None:
                raise ValueError("Failed to capture image for ROI selection")

            roi = cv2.selectROI(
                "Select startup cup ROI (ENTER=confirm, ESC=cancel)",
                frame,
                showCrosshair=True,
                fromCenter=False,
            )
            cv2.destroyAllWindows()

            x, y, w, h = [int(v) for v in roi]
            if w <= 0 or h <= 0:
                raise ValueError("ROI selection cancelled or invalid size")
            roi_text = f"{x},{y},{x + w},{y + h}"
            logger.info(f"Selected startup ROI: {roi_text}")
            print(f"Selected startup ROI: {roi_text}")
            return
        finally:
            try:
                cap.release()
            except Exception:
                pass

    root = tk.Tk()
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family="Noto Sans", size=12)
    root.option_add("*Font", default_font)

    text_font = tkfont.nametofont("TkTextFont")
    text_font.configure(family="Noto Sans Mono", size=11)

    WorkflowUI(root, args, project_root_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
