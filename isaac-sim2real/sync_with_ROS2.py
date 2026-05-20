#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
# Don't understand how these import helps, like at all
try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")
"""

# Standard Library
import argparse
import time
import queue
import os
import numpy as np
import trimesh
from isaacsim_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from isaacsim_utils import network_config

import cv2
import logging

############################################################

# Third Party
from omni.isaac.kit import SimulationApp  # noqa: E402


def get_headless_mode():
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument(
        "--headless_mode",
        type=str,
        default=None,
        help="To run headless, use one of [native, websocket], webrtc might not work.",
    )
    peek_args, _ = peek_parser.parse_known_args()
    return peek_args.headless_mode


simulation_app = SimulationApp(  # noqa: E402
    {
        "headless": get_headless_mode() is not None,
        "width": "1920",
        "height": "1080",
    }
)


from isaacsim_utils.helper import add_extensions, add_robot_to_scene  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid, sphere  # noqa: E402

# import omni.isaac.core.utils.prims as prims_utils  # noqa: E402
from omni.isaac.core.utils.types import ArticulationAction  # noqa: E402


######### CuRobo ########
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType  # noqa: E402
from curobo.geom.types import Cuboid, WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.types.robot import JointState  # noqa: E402
from curobo.util.logger import log_error, setup_curobo_logger  # noqa: E402
from curobo.util.usd_helper import UsdHelper  # noqa: E402
from curobo.util_file import (  # noqa: E402
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)

logger = logging.getLogger(__name__)
from curobo.wrap.reacher.motion_gen import (  # noqa: E402
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)


def cmd_to_move(cmd_plan):
    return cmd_plan.position.cpu().numpy().tolist()


def _resolve_camera_source_for_cv(source):
    """Resolve camera source string/int for cv2.VideoCapture."""
    try:
        s = str(source).strip()
        if s.isdigit():
            return int(s)
        if s.startswith("/dev/"):
            return s
        if "video-index" in s:
            return f"/dev/v4l/by-id/{s}"
        return s
    except Exception:
        return source


def capture_palm_displacement(
    camera_source,
    width=0,
    height=0,
    pixel_to_m=0.001,
    image_rotation_cw_deg=90.0,
    threshold=None,
    frames_warmup=5,
    preview=False,
    preview_wait_ms=1,
    preview_window_name="Palm Camera Preview",
):
    """Capture one frame from palm camera and compute a correction in robot y/z.

    Returns (dy_m, dz_m, annotated_frame) where:
    - dy_m, dz_m: computed correction (meters)
    - annotated_frame: BGR image with semantic mask/centroid drawn (or None if capture failed)
    """
    logger.info(f"Palm adjust (bridge) start pid={os.getpid()} ts={time.time():.6f}")
    src = _resolve_camera_source_for_cv(camera_source)
    cap = cv2.VideoCapture(src)
    try:
        if not cap.isOpened():
            logger.warning(f"Palm camera source could not be opened: {src}")
            return 0.0, 0.0, None
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        for _ in range(frames_warmup):
            cap.grab()
        ok, frame = cap.read()
        if not ok or frame is None:
            return 0.0, 0.0, None
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th = None
        if threshold is None:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

        # If mostly black (dark object), try inverted threshold
        black_ratio = np.sum(th == 0) / (h * w)
        if black_ratio > 0.3:
            logger.info(
                f"Dark frame detected ({black_ratio * 100:.1f}% black). Using THRESH_BINARY_INV."
            )
            if threshold is None:
                _, th = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            else:
                _, th = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)

        # Morphological filtering to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th = cv2.morphologyEx(
            th, cv2.MORPH_OPEN, kernel, iterations=1
        )  # Remove small noise
        th = cv2.morphologyEx(
            th, cv2.MORPH_CLOSE, kernel, iterations=1
        )  # Fill small holes

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by minimum area to reject noise
        min_contour_area = 500  # pixels; adjust as needed
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        if valid_contours:
            logger.info(
                f"Found {len(contours)} contours total, {len(valid_contours)} above min_area={min_contour_area}. "
                f"Areas: {[f'{cv2.contourArea(c):.0f}' for c in valid_contours[:5]]}"
            )
            contours = valid_contours
        else:
            logger.warning(
                f"Found {len(contours)} contours but none >= {min_contour_area} pixels. "
                f"Try lowering min_contour_area or adjusting threshold."
            )

        # Calculate centroid early so we can show it in preview
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M.get("m00", 0) != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                logger.info(
                    f"Palm centroid detected at pixel (cx={cx:.1f}, cy={cy:.1f}) in frame {w}x{h}"
                )

        if preview:
            try:
                preview_frame = frame.copy()
                cv2.drawMarker(
                    preview_frame,
                    (w // 2, h // 2),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(largest)
                    cv2.rectangle(
                        preview_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2
                    )
                    # Draw all valid contours
                    cv2.drawContours(preview_frame, contours, -1, (0, 128, 255), 2)
                    # Highlight the largest contour
                    cv2.drawContours(preview_frame, [largest], 0, (0, 255, 0), 3)
                    # Mark the calculated centroid
                    if cx is not None and cy is not None:
                        cv2.drawMarker(
                            preview_frame,
                            (int(cx), int(cy)),
                            (255, 0, 0),
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=15,
                            thickness=2,
                        )
                        cv2.putText(
                            preview_frame,
                            "centroid",
                            (int(cx) + 5, int(cy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                label = "Palm preview"
                if threshold is None:
                    label += " | Otsu"
                else:
                    label += f" | threshold={int(threshold)}"
                if cx is not None and cy is not None:
                    label += f" | centroid ({cx:.0f},{cy:.0f})"
                cv2.putText(
                    preview_frame,
                    label,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                mask_vis = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
                preview_row = np.hstack((preview_frame, mask_vis))
                cv2.imshow(preview_window_name, preview_row)
                cv2.waitKey(int(preview_wait_ms))
            except cv2.error as e:
                error_text = str(e)
                if (
                    "cvShowImage" in error_text
                    or "The function is not implemented" in error_text
                ):
                    logger.info(
                        "Palm preview unavailable; continuing without preview window."
                    )
                else:
                    logger.warning(f"Palm preview unavailable: {e}")
            except Exception as e:
                logger.warning(f"Palm preview unavailable: {e}")

        # Create annotated frame for saving (with contours/centroid drawn)
        annotated_frame = frame.copy()
        cv2.drawMarker(
            annotated_frame,
            (w // 2, h // 2),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            cv2.rectangle(annotated_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            # Draw all valid contours
            cv2.drawContours(annotated_frame, contours, -1, (0, 128, 255), 2)
            # Highlight the largest contour
            cv2.drawContours(annotated_frame, [largest], 0, (0, 255, 0), 3)
            # Mark the calculated centroid
            if cx is not None and cy is not None:
                cv2.drawMarker(
                    annotated_frame,
                    (int(cx), int(cy)),
                    (255, 0, 0),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=15,
                    thickness=2,
                )

        if not contours:
            return 0.0, 0.0, annotated_frame
        if cx is None or cy is None:
            return 0.0, 0.0, annotated_frame
        dx_px_display = cx - (w / 2.0)
        dy_px_display = cy - (h / 2.0)

        # Rotate the measured offset back into the camera/tool frame before scaling.
        theta = np.deg2rad(float(image_rotation_cw_deg))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        dx_px = cos_t * dx_px_display + sin_t * dy_px_display
        dy_px = -sin_t * dx_px_display + cos_t * dy_px_display

        dy_m = float(-dx_px) * float(pixel_to_m)
        dz_m = float(-dy_px) * float(pixel_to_m)
        return dy_m, dz_m, annotated_frame
    finally:
        try:
            cap.release()
        except Exception:
            pass


def quaternion_wxyz_to_euler_deg(quaternion: list[float]) -> list[float]:
    w, x, y, z = [float(v) for v in quaternion]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg([roll, pitch, yaw]).tolist()


def pose_goal_to_mmdeg(goal: list[float]) -> list[float]:
    xyz_mm = [float(v) * 1000.0 for v in goal[:3]]
    rpy_deg = quaternion_wxyz_to_euler_deg(goal[3:])
    return xyz_mm + rpy_deg


def tool_frame_offset_to_base(
    pose_mmdeg: list[float], dy_m: float, dz_m: float, pose_unit_scale: float
) -> tuple[float, float, float]:
    """Rotate a local tool-frame y/z correction into the robot base frame.

    The palm camera correction is measured in the tool's local image plane, so the
    displacement must be rotated by the current end-effector orientation before it
    is added to the cartesian pose.
    """
    if len(pose_mmdeg) < 6:
        return 0.0, dy_m * pose_unit_scale, dz_m * pose_unit_scale

    roll_deg = float(pose_mmdeg[3])
    pitch_deg = float(pose_mmdeg[4])
    yaw_deg = float(pose_mmdeg[5])
    rotation = trimesh.transformations.euler_matrix(
        np.deg2rad(roll_deg),
        np.deg2rad(pitch_deg),
        np.deg2rad(yaw_deg),
        axes="sxyz",
    )[:3, :3]
    local_offset = np.array([dy_m, dz_m, 0.0], dtype=float) * float(pose_unit_scale)
    base_offset = rotation @ local_offset
    return float(base_offset[0]), float(base_offset[1]), float(base_offset[2])


def init_pose_matric(args, motion_gen):
    pose_metric = None
    if args.constrain_grasp_approach:
        pose_metric = PoseCostMetric.create_grasp_approach_metric()
    if args.reach_partial_pose is not None:
        reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
        pose_metric = PoseCostMetric(
            reach_partial_pose=True, reach_vec_weight=reach_vec
        )
    if args.hold_partial_pose is not None:
        hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
        pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
    return pose_metric


# dataclass
class Move:
    def __init__(self, ROS2_move: dict, cmd_plan: list):
        self.ROS2_move = ROS2_move
        self.cmd_plan = cmd_plan


def sanitize_xyz(values, fallback=0.0) -> np.ndarray:
    xyz = np.asarray(values, dtype=np.float64).reshape(-1)
    if xyz.size < 3:
        xyz = np.pad(xyz, (0, 3 - xyz.size), constant_values=fallback)
    xyz = xyz[:3]
    if not np.all(np.isfinite(xyz)):
        xyz = np.where(np.isfinite(xyz), xyz, fallback)
    return xyz


def sanitize_dims(values, min_dim=1e-4) -> np.ndarray:
    dims = np.asarray(values, dtype=np.float64).reshape(-1)
    if dims.size < 3:
        dims = np.pad(dims, (0, 3 - dims.size), constant_values=min_dim)
    dims = np.abs(dims[:3])
    if not np.all(np.isfinite(dims)):
        dims = np.where(np.isfinite(dims), dims, min_dim)
    return np.maximum(dims, min_dim)


def get_cuboid_list(move: dict, obstacles: dict) -> list:
    cuboids = []
    cuboids.append(
        Cuboid(
            name="table",
            pose=[0, 0, -1.97] + [1, 0, 0, 0],
            dims=[4, 4, 4],
        )
    )
    for i, obstacle_name in enumerate(obstacles):
        if not (
            "ignore_obstacles" in move and obstacle_name in move["ignore_obstacles"]
        ):
            middle_point = np.mean(
                [
                    obstacles[obstacle_name]["max"],
                    obstacles[obstacle_name]["min"],
                ],
                axis=0,
            )
            scale = np.array(obstacles[obstacle_name]["max"]) - np.array(
                obstacles[obstacle_name]["min"]
            )
            safe_middle_point = sanitize_xyz(middle_point)
            safe_scale = sanitize_dims(scale)
            cuboids.append(
                Cuboid(
                    name=f"obs_{i}",
                    pose=safe_middle_point.tolist() + [1, 0, 0, 0],
                    dims=safe_scale.tolist(),
                )
            )
    return cuboids


def basic_world_config():
    # just a big table.
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5
    return WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)


def basic_motion_gen(tensor_args, robot_cfg, world_cfg):
    trajopt_tsteps = 32
    trajopt_dt = None
    optimize_dt = True
    trim_steps = None
    interpolation_dt = 0.05
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    return MotionGen(motion_gen_config)


def basic_plan_config():
    max_attempts = 4
    enable_finetune_trajopt = True

    return MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5,
    )


def zero_obstacle_world_config(usd_help, robot_prim_path):
    return usd_help.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[
            robot_prim_path,
            "/World/defaultGroundPlane",
            "/curobo",
            "/World/table",
        ],
    ).get_collision_check_world()


def still_joint_states(joint_states: list, tensor_args: TensorDeviceType, sim_js_names):
    return JointState(
        position=tensor_args.to_device(joint_states),
        velocity=tensor_args.to_device([0.0] * len(joint_states)),
        acceleration=tensor_args.to_device([0.0] * len(joint_states)),
        jerk=tensor_args.to_device([0.0] * len(joint_states)),
        joint_names=sim_js_names,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize_spheres",
        action="store_true",
        help="When True, visualizes robot spheres",
        default=False,
    )
    parser.add_argument(
        "--headless_mode",
        type=str,
        default=None,
        help="To run headless, use one of [native, websocket], webrtc might not work.",
    )
    parser.add_argument(
        "--robot", type=str, default="tm5s.yml", help="robot configuration to load"
    )
    parser.add_argument(
        "--external_asset_path",
        type=str,
        default=None,
        help="Path to external assets when loading an externally located robot",
    )
    parser.add_argument(
        "--external_robot_configs_path",
        type=str,
        default=None,
        help="Path to external robot config when loading an external robot",
    )
    parser.add_argument(
        "--constrain_grasp_approach",
        action="store_true",
        help="When True, approaches grasp with fixed orientation and motion only along z axis.",
        default=False,
    )
    parser.add_argument(
        "--reach_partial_pose",
        nargs=6,
        metavar=("qx", "qy", "qz", "x", "y", "z"),
        help="Reach partial pose",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hold_partial_pose",
        nargs=6,
        metavar=("qx", "qy", "qz", "x", "y", "z"),
        help="Hold partial pose while moving to goal",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--base_platform_height",
        type=float,
        default=0.08,
        help="Robot base lift in meters (e.g., 0.08 for an 8cm platform).",
    )
    return parser.parse_args()


def main():
    ###### Basic setup ######
    args = parse_args()
    setup_curobo_logger("warn")
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage

    ###### Setup Robot ######
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]
    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = (
            args.external_robot_configs_path
        )
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    robot, robot_prim_path = add_robot_to_scene(
        robot_cfg,
        my_world,
        position=np.array([0.0, 0.0, args.base_platform_height], dtype=np.float64),
    )

    world_cfg = basic_world_config()
    tensor_args = TensorDeviceType()
    motion_gen = basic_motion_gen(tensor_args, robot_cfg, world_cfg)
    plan_config = basic_plan_config()

    print("warming up...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, get_headless_mode())

    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")
    zero_obstacles = zero_obstacle_world_config(usd_help, robot_prim_path)
    pose_metric = init_pose_matric(args, motion_gen)

    ###### states ######
    planned_action_queue = queue.Queue()
    ROS2_fail_queue = queue.Queue()  # put message in if ROS2 fail
    cmd_plan = None
    cmd_idx = 0
    articulation_controller = robot.get_articulation_controller()
    tick = 0
    spheres = None
    wait_ros2 = False
    graspgen_receiver = NonBlockingJSONReceiver(
        port=network_config.GRASPGEN_TO_ISAACSIM_PORT
    )
    graspgen_sender = NonBlockingJSONSender(
        port=network_config.ISAACSIM_TO_GRASPGEN_PORT
    )
    # When a new planned action is accepted from graspgen, set this flag so
    # we can ACK only after the planned action has been executed. This prevents
    # the UI from assuming the motion completed immediately after planning.
    pending_graspgen_ack = False
    ros2_receiver = NonBlockingJSONReceiver(port=network_config.ROS2_TO_ISAACSIM_PORT)
    ros2_sender = NonBlockingJSONSender(port=network_config.ISAACSIM_TO_ROS2_PORT)
    sim_js = robot.get_joints_state()
    sim_js_names = robot.dof_names
    planned_action_moves: list = []
    idx_list = [0, 1, 2, 3, 4, 5]
    temp_cuboid_paths = []
    last_cartesian_pose: list[float] | None = None
    pending_cartesian_translation: list[float] | None = None

    default_config = [0.0991, -0.4892, 2.1493, 1.5596, -1.4944, 3.0737]

    last_joint_states = default_config
    temp_cuboid_paths = []
    common_js_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    graspgen_eof = False

    while simulation_app.is_running():
        # try handle first
        for _ in range(1):
            ###### Got message from gripper_server.py ######
            if not ROS2_fail_queue.empty():
                # clear plan
                while not planned_action_queue.empty():
                    planned_action_queue.get()
                notice = (
                    ROS2_fail_queue.get()
                )  # catch the fail, let isaacsim continue to move
                print("get Notice", notice)
                if notice["message"] == "Abort":
                    last_joint_states = default_config
                    graspgen_sender.send_data({"message": "Abort"})
                    # eat datas
                    for _ in range(5):
                        time.sleep(0.1)
                        graspgen_receiver.capture_data()
                elif notice["message"] == "ROS2 Complete":
                    if graspgen_eof:
                        graspgen_sender.send_data({"message": "EOF and ROS2 Complete"})
                    else:
                        print("ROS2 complete but not EOF yet.")
                else:
                    raise ValueError("Unknown message")

                continue
            ###### Got message from graspgen ######
            graspgen_datas = graspgen_receiver.capture_data()
            if graspgen_datas is None:
                continue
            if graspgen_datas[0] == "EOF":
                logger.info("Received EOF from graspgen")
                graspgen_eof = True
                # If ROS2 is already complete, send the final response immediately
                if not ROS2_fail_queue.empty():
                    notice = ROS2_fail_queue.get()
                    if notice.get("message") == "ROS2 Complete":
                        logger.info(
                            "ROS2 already complete; sending EOF and ROS2 Complete response"
                        )
                        graspgen_sender.send_data({"message": "EOF and ROS2 Complete"})
                        ROS2_fail_queue.put(notice)  # put it back for normal flow
                continue
            elif graspgen_datas[0] == "Reset_to_default":
                last_joint_states = default_config
                continue
            print("-------------Received new action--------------")
            # Only reset graspgen_eof if this is a real action (not EOF or Reset),
            # to avoid race condition where EOF flag gets cleared before ROS2 complete processes it
            if len(graspgen_datas) > 0 and isinstance(graspgen_datas[0], dict):
                graspgen_eof = False
            for graspgen_data in graspgen_datas:
                before_move_joints = last_joint_states
                curobo_planned_action_moves: list[Move] = []
                skip_curobo_for_action = bool(graspgen_data.get("skip_curobo", False))
                for move in graspgen_data["moves"]:
                    cuboids = get_cuboid_list(move, graspgen_data["obstacles"])
                    obstacles = WorldConfig(cuboid=cuboids)
                    if "no_obstacles" in move:
                        motion_gen.update_world(zero_obstacles)
                    else:
                        motion_gen.update_world(obstacles)
                    # start handle move
                    if move["type"] == "gripper":
                        curobo_planned_action_moves.append(Move(move, None))
                        continue

                    if move.get("type") == "workflow_palm_adjust":
                        logger.info(
                            "Skipping workflow-only palm adjust marker in sync_with_ROS2"
                        )
                        continue

                    if move.get("type") == "palm_adjust":
                        camera_source = move.get("camera_source", "0")
                        camera_width = move.get("camera_width", 0)
                        camera_height = move.get("camera_height", 0)
                        pixel_to_m = move.get("pixel_to_m", 0.001)
                        image_rotation_cw_deg = move.get("image_rotation_cw_deg", 90.0)
                        threshold = move.get("threshold", None)
                        adjust_z = move.get("adjust_z", False)
                        target_name = move.get("target_name", None)
                        preview = bool(move.get("preview", False))
                        preview_wait_ms = int(move.get("preview_wait_ms", 1))
                        ROS2_move = {
                            "type": "palm_adjust",
                            "wait_time": move.get("wait_time", 0.0),
                            "camera_source": camera_source,
                            "camera_width": camera_width,
                            "camera_height": camera_height,
                            "pixel_to_m": pixel_to_m,
                            "image_rotation_cw_deg": image_rotation_cw_deg,
                            "threshold": threshold,
                            "adjust_z": adjust_z,
                            "target_name": target_name,
                            "preview": preview,
                            "preview_wait_ms": preview_wait_ms,
                        }
                        curobo_planned_action_moves.append(Move(ROS2_move, None))
                        continue

                    skip_curobo = skip_curobo_for_action or bool(
                        move.get("skip_curobo", False)
                    )
                    if skip_curobo:
                        if "joints_values" in move:
                            positions = move["joints_values"]
                            ROS2_move = {
                                "type": "arm",
                                "wait_time": move["wait_time"],
                                "joints_values": positions,
                            }
                            if "custom_vel" in move:
                                ROS2_move["custom_vel"] = move["custom_vel"]
                            if "custom_acc" in move:
                                ROS2_move["custom_acc"] = move["custom_acc"]
                            if "custom_blend" in move:
                                ROS2_move["custom_blend"] = move["custom_blend"]
                            last_joint_states = positions[-1]
                        elif "joints_goal" in move:
                            positions = [move["joints_goal"]]
                            ROS2_move = {
                                "type": "arm",
                                "wait_time": move["wait_time"],
                                "joints_values": positions,
                            }
                            if "custom_vel" in move:
                                ROS2_move["custom_vel"] = move["custom_vel"]
                            if "custom_acc" in move:
                                ROS2_move["custom_acc"] = move["custom_acc"]
                            if "custom_blend" in move:
                                ROS2_move["custom_blend"] = move["custom_blend"]
                            last_joint_states = positions[-1]
                        elif "goal" in move:
                            goal = move["goal"]
                            cartesian_pose = pose_goal_to_mmdeg(goal)
                            ROS2_move = {
                                "type": "PTP",
                                "wait_time": move["wait_time"],
                                "cartesian_poses": [cartesian_pose],
                            }
                            if "custom_vel" in move:
                                ROS2_move["custom_vel"] = move["custom_vel"]
                            if "custom_acc" in move:
                                ROS2_move["custom_acc"] = move["custom_acc"]
                        else:
                            print(
                                "skip_curobo requires joints_values, joints_goal, or goal in move"
                            )
                            last_joint_states = before_move_joints
                            break
                        curobo_planned_action_moves.append(Move(ROS2_move, None))
                        continue

                    print("curoboing")
                    curobo_cu_js = still_joint_states(
                        last_joint_states, tensor_args, sim_js_names
                    )
                    if "goal" in move:
                        ik_goal = Pose(
                            position=tensor_args.to_device(move["goal"][:3]),
                            quaternion=tensor_args.to_device(move["goal"][3:]),
                        )

                        plan_config.pose_cost_metric = pose_metric
                        result = motion_gen.plan_single(
                            curobo_cu_js.unsqueeze(0), ik_goal, plan_config
                        )
                    elif "joints_goal" in move:
                        print("ALRIGHT?0")
                        joints_goal = JointState(
                            position=tensor_args.to_device(move["joints_goal"]),
                            velocity=tensor_args.to_device(sim_js.velocities)
                            * 0.0,  # * 0.0,
                            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                            joint_names=sim_js_names,
                        )

                        plan_config.pose_cost_metric = pose_metric
                        result = motion_gen.plan_single_js(
                            curobo_cu_js.unsqueeze(0),
                            joints_goal.unsqueeze(0),
                            plan_config,
                        )
                        print("ALRIGHT?3")

                    succ = result.success.item()  # ik_result.success.item()
                    if succ:
                        print("YES YES YES?")
                        new_cmd_plan = result.get_interpolated_plan()
                        new_cmd_plan = motion_gen.get_full_js(new_cmd_plan)
                        new_cmd_plan = new_cmd_plan.get_ordered_joint_state(
                            common_js_names
                        )
                        # The following code block shows how to prune the plan to keep only the first and last waypoints
                        # Emulates IK method's behavior.
                        if "no_curobo" in move:
                            new_cmd_plan = JointState(
                                position=new_cmd_plan.position[[0, -1]],
                                velocity=new_cmd_plan.velocity[[0, -1]],
                                acceleration=new_cmd_plan.acceleration[[0, -1]],
                                jerk=new_cmd_plan.jerk[[0, -1]],
                                joint_names=new_cmd_plan.joint_names,
                            )
                        positions = cmd_to_move(new_cmd_plan)
                        ROS2_move = {
                            "type": move["type"],
                            "wait_time": move["wait_time"],
                            # "cmd_plan": cmd_plan, # only for later reuse by isaacsim, not for ROS2
                            "joints_values": positions,
                        }
                        if "goal" in move:
                            cartesian_pose = pose_goal_to_mmdeg(move["goal"])
                            ROS2_move["cartesian_poses"] = [cartesian_pose]
                        if "custom_vel" in move:
                            ROS2_move["custom_vel"] = move["custom_vel"]
                        if "custom_acc" in move:
                            ROS2_move["custom_acc"] = move["custom_acc"]
                        if "custom_blend" in move:
                            ROS2_move["custom_blend"] = move["custom_blend"]

                        curobo_planned_action_moves.append(
                            Move(ROS2_move, new_cmd_plan)
                        )
                        last_joint_states = positions[-1]
                    else:
                        print("This plan failed.")
                        last_joint_states = before_move_joints
                        break

                else:  # success!
                    print("-------------Successfully handled new action--------------")
                    # Enqueue the planned action for execution and defer the Success
                    # acknowledgement until the action has actually been applied.
                    planned_action_queue.put(
                        {"moves": curobo_planned_action_moves, "obstacles": cuboids}
                    )
                    pending_graspgen_ack = True
                    break  # stop trying other acts
            else:  # all graspgen_datas failed
                graspgen_sender.send_data({"message": "Fail"})
        # end of handle section

        if wait_ros2:
            ros2_response = ros2_receiver.capture_data()
            if ros2_response is not None:
                wait_ros2 = False
                if ros2_response["message"] == "Success":
                    print("receiver successfulness.")
                    if len(planned_action_moves) == 0 and planned_action_queue.empty():
                        ROS2_fail_queue.put({"message": "ROS2 Complete"})
                        print("ROS2 Complete, go check that!")
                    # Can continue to do the following steps, no need to stuck
                elif ros2_response["message"] == "Fail":
                    print("receiver failedness.")
                    # reset simulation robot position
                    robot.set_joint_positions(default_config, idx_list)
                    robot._articulation_view.set_max_efforts(
                        values=np.array([5000 for i in range(len(idx_list))]),
                        joint_indices=idx_list,
                    )
                    cmd_plan = None
                    planned_action_moves = []
                    # Finished handled
                    ROS2_fail_queue.put(
                        {"message": "Abort"}
                    )  # tell that thread to handle
                    continue  # Go and please stuck

        # Step
        my_world.step(render=True)
        step_index = my_world.current_time_step_index
        ###### Print the press play hint ######
        if not my_world.is_playing():
            if tick % 100 == 0:
                print("**** Click Play to start simulation *****")
            tick += 1
            continue

        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]),
                joint_indices=idx_list,
            )
        if step_index < 20:
            continue

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = still_joint_states(sim_js.positions, tensor_args, sim_js_names)

        cu_js.velocity *= 0.0
        cu_js.acceleration *= 0.0

        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
        ###### update past_pose
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
        if not wait_ros2 and len(planned_action_moves) > 0:
            # planned action
            wait_ros2 = True
            move: Move = planned_action_moves.pop(0)
            if (
                pending_cartesian_translation is not None
                and isinstance(move.ROS2_move, dict)
                and "cartesian_poses" in move.ROS2_move
            ):
                adjusted_cartesian_poses = []
                for cartesian_pose in move.ROS2_move["cartesian_poses"]:
                    adjusted_pose = list(cartesian_pose)
                    for axis in range(3):
                        adjusted_pose[axis] = float(adjusted_pose[axis]) + float(
                            pending_cartesian_translation[axis]
                        )
                    adjusted_cartesian_poses.append(adjusted_pose)
                move.ROS2_move["cartesian_poses"] = adjusted_cartesian_poses
                pending_cartesian_translation = None
            if isinstance(move.ROS2_move, dict) and "cartesian_poses" in move.ROS2_move:
                last_cartesian_pose = list(move.ROS2_move["cartesian_poses"][-1])

            if (
                isinstance(move.ROS2_move, dict)
                and move.ROS2_move.get("type") == "palm_adjust"
            ):
                camera_source = move.ROS2_move.get("camera_source", "0")
                camera_width = move.ROS2_move.get("camera_width", 0)
                camera_height = move.ROS2_move.get("camera_height", 0)
                pixel_to_m = move.ROS2_move.get("pixel_to_m", 0.001)
                image_rotation_cw_deg = move.ROS2_move.get(
                    "image_rotation_cw_deg", 90.0
                )
                threshold = move.ROS2_move.get("threshold", None)
                target_name = move.ROS2_move.get("target_name", None)
                preview = bool(move.ROS2_move.get("preview", False))
                preview_wait_ms = int(move.ROS2_move.get("preview_wait_ms", 1))
                try:
                    print(
                        f"Palm adjust start after robot reached target: source={camera_source}, size={camera_width}x{camera_height}"
                    )
                    dy_m, dz_m, annotated_frame = capture_palm_displacement(
                        camera_source,
                        width=camera_width,
                        height=camera_height,
                        pixel_to_m=pixel_to_m,
                        image_rotation_cw_deg=image_rotation_cw_deg,
                        threshold=threshold,
                        target_name=target_name,
                        preview=preview,
                        preview_wait_ms=preview_wait_ms,
                    )

                    palm_msg = f"Palm adjust computed dy={dy_m:.4f}m dz={dz_m:.4f}m"
                    print(palm_msg)
                    logger.info(palm_msg)
                    if abs(dy_m) < 1e-6 and abs(dz_m) < 1e-6:
                        logger.info(
                            "Palm adjust: no significant displacement detected."
                        )
                        wait_ros2 = True
                        continue

                    if last_cartesian_pose is None:
                        logger.warning(
                            "Palm adjust: no prior cartesian pose available, skipping adjustment."
                        )
                        wait_ros2 = True
                        continue

                    new_pose = last_cartesian_pose.copy()
                    # Determine unit of last_cartesian_pose: if values are large (>10), assume mm; otherwise meters
                    try:
                        sample_val = abs(float(new_pose[0]))
                    except Exception:
                        sample_val = 0.0
                    if sample_val > 10.0:
                        # last pose is in mm, convert meters->mm
                        scale = 1000.0
                        unit = "mm"
                    else:
                        scale = 1.0
                        unit = "m"

                    add_dx, add_dy, add_dz = tool_frame_offset_to_base(
                        new_pose, dy_m, dz_m, scale
                    )
                    print(
                        f"Palm adjust offsets in base frame: add_dx={add_dx:.4f} {unit}, add_dy={add_dy:.4f} {unit}, add_dz={add_dz:.4f} {unit}"
                    )
                    new_pose[0] = float(new_pose[0]) + add_dx
                    new_pose[1] = float(new_pose[1]) + add_dy
                    new_pose[2] = float(new_pose[2]) + add_dz * (
                        1.0 if adjust_z else 0.0
                    )
                    logger.info(
                        f"Palm adjust unit detected: {unit}, raw dy={dy_m:.4f}m dz={dz_m:.4f}m, "
                        f"base-frame add dx={add_dx:.4f} {unit}, dy={add_dy:.4f} {unit}, dz={add_dz:.4f} {unit}"
                    )
                    pending_cartesian_translation = [
                        float(new_pose[i]) - float(last_cartesian_pose[i])
                        for i in range(3)
                    ]
                    # Log and save preview for debugging (use annotated frame from capture)
                    print(f"Sending palm-adjust PTP correction: {new_pose}")
                    logger.info(f"Sending palm-adjust PTP correction: {new_pose}")
                    if annotated_frame is not None:
                        # Add adjustment text to annotated frame
                        text_label = f"dy={dy_m:.4f}m dz={dz_m:.4f}m"
                        cv2.putText(
                            annotated_frame,
                            text_label,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        out_dir = os.path.join(os.getcwd(), "output", "palm_previews")
                        try:
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(
                                out_dir, f"palm_preview_{int(time.time())}.png"
                            )
                            cv2.imwrite(out_path, annotated_frame)
                            logger.info(f"Saved palm preview frame: {out_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save palm preview frame: {e}")

                    success = ros2_sender.send_data(
                        {
                            "type": "PTP",
                            "wait_time": float(move.ROS2_move.get("wait_time", 0.0)),
                            "cartesian_poses": [new_pose],
                        }
                    )
                    if not success:
                        logger.warning(
                            "Failed to send palm-adjust PTP correction to ROS2 sender."
                        )
                    else:
                        logger.info("Palm-adjust PTP sent to ROS2 sender successfully.")
                    cmd_idx = 0
                    cmd_plan = None
                    last_cartesian_pose = new_pose.copy()
                    # Block here until ROS2 completes the correction
                    while True:
                        ros2_response = ros2_receiver.capture_data()
                        if ros2_response is not None:
                            if ros2_response["message"] == "Success":
                                logger.info(
                                    "Palm adjust correction completed successfully."
                                )
                                break
                            elif ros2_response["message"] == "Fail":
                                logger.warning("Palm adjust correction failed.")
                                break
                            elif ros2_response["message"] == "Abort":
                                logger.warning("Palm adjust correction aborted.")
                                break
                        time.sleep(0.01)
                    continue
                except Exception as e:
                    print(f"Palm adjust failed: {e}")
                    logger.exception(f"Palm adjust failed: {e}")
                    continue

            # For ROS2
            ros2_sender.send_data(move.ROS2_move)
            # For isaac sim animation
            cmd_idx = 0
            cmd_plan = move.cmd_plan

        # If we previously accepted a planned action from graspgen, only ACK it
        # once the planned moves have been executed (no pending cmd_plan and no
        # queued planned actions). This prevents the UI from assuming motion is
        # complete immediately after planning.
        if (
            pending_graspgen_ack
            and len(planned_action_moves) == 0
            and planned_action_queue.empty()
            and not wait_ros2
            and cmd_plan is None
        ):
            try:
                graspgen_sender.send_data({"message": "Success"})
                logger.info(
                    "Acknowledged graspgen Success after planned action executed."
                )
            except Exception:
                logger.exception("Failed to send delayed graspgen Success ack.")
            pending_graspgen_ack = False

        if len(planned_action_moves) == 0 and not planned_action_queue.empty():
            # currently no plan but we have more in queue, can start grab a new planned_action and apply here.
            planned_action = planned_action_queue.get()
            planned_action_moves: list = planned_action["moves"]
            # visualize cuboids
            if temp_cuboid_paths:
                for path in temp_cuboid_paths:
                    stage.RemovePrim(path)  # this may race condition
                temp_cuboid_paths = []
            cube: Cuboid
            for i, cube in enumerate(planned_action["obstacles"]):
                prim_path = f"/World/temp_obstacle_{i}"
                safe_position = sanitize_xyz(cube.pose[:3])
                safe_scale = sanitize_dims(cube.dims)
                cuboid.VisualCuboid(
                    prim_path=prim_path,
                    position=safe_position,
                    scale=safe_scale,
                    color=np.array([0.0, 0.0, 1.0]),  # Blue
                )
                temp_cuboid_paths.append(prim_path)
    simulation_app.close()


if __name__ == "__main__":
    main()
