import trimesh
import logging
import numpy as np
from common_utils.qualification import get_left_up_and_front

HOME_SIGNAL = [326.8, -140.2, 212.6, 90.0, 0, 90.0]

logger = logging.getLogger(__name__)

FAST_ARM_CUSTOM_VEL = 40
FAST_ARM_CUSTOM_ACC = 10


def _apply_startup_cup_offset(
    base_position: list[float], scene_data: dict, cup_index: int
) -> list[float]:
    """Shift a base xyz position by startup cup ROI offset projected into robot x/y.

    Uses pixel-to-meter conversion (meter_per_pixel) if available, otherwise falls back to gain method.
    """
    flip_x = scene_data.get("startup_cup_offset_flip_x", True)

    # Try to use shared pixel-to-meter conversion for this ROI.
    meter_per_pixel = scene_data.get("startup_cup_meter_per_pixel")
    offsets_px = scene_data.get("startup_cup_offsets_px")

    if isinstance(offsets_px, list) and cup_index < len(offsets_px):
        try:
            offset_px = offsets_px[cup_index]

            if not (isinstance(meter_per_pixel, list) and len(meter_per_pixel) >= 2):
                # Backward compatibility with per-cup conversion data shape.
                meters_per_pixel_list = scene_data.get("startup_cup_meters_per_pixel")
                if isinstance(meters_per_pixel_list, list) and cup_index < len(
                    meters_per_pixel_list
                ):
                    meter_per_pixel = meters_per_pixel_list[cup_index]

            if (
                isinstance(meter_per_pixel, list)
                and len(meter_per_pixel) >= 2
                and isinstance(offset_px, list)
                and len(offset_px) >= 2
            ):
                dx_px = float(offset_px[0])
                dy_px = float(offset_px[1])
                meter_per_pixel_x = float(meter_per_pixel[0])
                meter_per_pixel_y = float(meter_per_pixel[1])

                # Convert pixel offset to meters
                dx_m = dx_px * meter_per_pixel_x
                dy_m = dy_px * meter_per_pixel_y

                shifted = list(base_position)
                if flip_x:
                    shifted[0] = float(shifted[0]) - dx_m
                else:
                    shifted[0] = float(shifted[0]) + dx_m
                shifted[1] = float(shifted[1]) + dy_m
                # if dy_m >= -0.2:
                #     shifted[1] += 0.07
                # else:
                #     shifted[1] += 0.02
                return shifted
                # return base_position
        except (TypeError, ValueError, IndexError):
            pass  # Fall back to gain-based method

    # Fall back to normalized offset with gain (legacy method)
    offsets = scene_data.get("startup_cup_offsets_norm")
    if not isinstance(offsets, list) or cup_index >= len(offsets):
        return list(base_position)

    offset = offsets[cup_index]
    if not isinstance(offset, list) or len(offset) < 2:
        return list(base_position)

    try:
        dx_norm = float(offset[0])
        dy_norm = float(offset[1])
        gain_x = float(scene_data.get("startup_cup_offset_gain_x", 0.10))
        gain_y = float(scene_data.get("startup_cup_offset_gain_y", 0.10))
    except (TypeError, ValueError):
        return list(base_position)

    shifted = list(base_position)
    if flip_x:
        shifted[0] = float(shifted[0]) - dx_norm * gain_x
    else:
        shifted[0] = float(shifted[0]) + dx_norm * gain_x
    shifted[1] = float(shifted[1]) + dy_norm * gain_y
    if dy_norm >= -0.2:
        shifted[1] += 0.07
    else:
        shifted[1] += 0.02
    return shifted


def pick_and_pour_and_put_back(grasp: np.array) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p * 1000 for p in position]
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)

    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})


def grab_and_pour_and_place_back(
    grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p * 1000 for p in position]
    logger.info(position)
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific fixed poses
    if isinstance(args[0], list):
        ready_pour_position = args[0]
    elif isinstance(args[0], str):
        obj_points = scene_data["object_infos"][args[0]]["points"]
        mass_center = np.mean(obj_points, axis=0)
        mass_center = [p * 1000 for p in mass_center]
        # std = np.std(obj_points, axis=0)
        ready_pour_position = [
            mass_center[0] - 175,
            mass_center[1] + 150,
            mass_center[2] + 250,
        ]
    ready_pour_pose = ready_pour_position + [90, 0, 90]
    pour_pose = ready_pour_position + [-90, -55, -90]
    before_grasp_position = [p - f * 60 for p, f in zip(position, front, strict=False)]
    grasp_position = [p + f * 60 for p, f in zip(position, front, strict=False)]
    after_grasp_position = grasp_position[:2] + [grasp_position[2] + 250]

    release_position = grasp_position[:2] + [grasp_position[2] + 5]
    after_release_position = before_grasp_position
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append(
        {
            "type": "move_arm",
            "goal": before_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "move_arm",
            "goal": grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "goal": "grab"})
    moves.append(
        {
            "type": "move_arm",
            "goal": after_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "move_arm", "goal": ready_pour_pose, "wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": pour_pose, "wait_time": 1.0})
    moves.append({"type": "move_arm", "goal": ready_pour_pose, "wait_time": 0.0})
    moves.append(
        {
            "type": "move_arm",
            "goal": after_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "move_arm",
            "goal": release_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "goal": "release"})
    moves.append(
        {
            "type": "move_arm",
            "goal": after_release_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    return moves


def grab_and_pour_and_place_back_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> dict:
    obstacles = []
    for obj_name in scene_data["object_infos"]:
        if target_name != obj_name:
            obstacles.append(
                {
                    "mass_center": list(
                        np.mean(scene_data["object_infos"][obj_name]["points"], axis=0)
                    ),
                    "std": list(
                        np.std(scene_data["object_infos"][obj_name]["points"], axis=0)
                    ),
                }
            )
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific fixed poses
    if isinstance(args[0], list):
        ready_pour_position = args[0]
    elif isinstance(args[0], str):
        obj_points = scene_data["object_infos"][args[0]]["points"]
        mass_center = np.mean(obj_points, axis=0)
        # std = np.std(obj_points, axis=0)
        ready_pour_position = [
            mass_center[0] - 0.175,
            mass_center[1] + 0.150,
            # mass_center[2] + 0.250,
            mass_center[2] + 0.150,
        ]
    ready_pour_pose = ready_pour_position + [0.5, 0.5, 0.5, 0.5]
    pour_pose = ready_pour_position + [-0.271, 0.653, -0.271, 0.653]
    before_grasp_position = [
        p - f * 0.100 for p, f in zip(position, front, strict=False)
    ]
    grasp_position = [p + f * 0.060 for p, f in zip(position, front, strict=False)]
    # after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.250]

    release_position = grasp_position[:2] + [grasp_position[2] + 0.005]
    after_release_position = before_grasp_position
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "grip_type": "close", "wait_time": 1.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append({"type": "arm", "goal": ready_pour_pose, "wait_time": 0.0})
    moves.append({"type": "arm", "goal": pour_pose, "wait_time": 1.0})
    moves.append({"type": "arm", "goal": ready_pour_pose, "wait_time": 0.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append(
        {
            "type": "arm",
            "goal": release_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {
            "type": "arm",
            "goal": after_release_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )

    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def grab_and_pour_and_place_back_curobo_by_rotation(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> dict:
    obstacles = scene_data["obstacles"]
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()

    # Grasp Position
    before_grasp_position = [
        p - f * 0.050 for p, f in zip(position, front, strict=False)
    ]
    grasp_position = [p + f * 0.048 for p, f in zip(position, front, strict=False)]

    # specific fixed poses
    if isinstance(args[0], list):
        middle_point = np.array(args[0])
    elif isinstance(args[0], str):
        obj_bounding_box = scene_data["obstacles"][args[0]]
        middle_point = np.mean(
            [
                obj_bounding_box["max"],
                obj_bounding_box["min"],
            ],
            axis=0,
        )
    ## Ready pour position
    grasp_angle = np.arctan2(grasp_position[1], grasp_position[0])
    target_angle = np.arctan2(middle_point[1], middle_point[0])

    # compute radius using mass_center[0] and mass_center[1]
    radius = np.linalg.norm(middle_point[:2]) - 0.20

    angle_diff = target_angle - grasp_angle
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi

    if angle_diff < 0:  # Clockwise
        goal_angle = target_angle + np.deg2rad(5)
    else:  # Counter-clockwise
        goal_angle = target_angle - np.deg2rad(5)
    ready_pour_position = [
        radius * np.cos(goal_angle),
        radius * np.sin(goal_angle),
        middle_point[2] + 0.200,
    ]
    q_z_rotation = trimesh.transformations.quaternion_about_axis(goal_angle, [0, 0, 1])
    q_y_rotation = trimesh.transformations.quaternion_about_axis(
        -np.arcsin(front[2]), [0, 1, 0]
    )
    q_base = np.array([0.5, 0.5, 0.5, 0.5])
    q_base_tilt = trimesh.transformations.quaternion_multiply(
        q_y_rotation, q_base
    ).tolist()
    ready_pour_rotation = trimesh.transformations.quaternion_multiply(
        q_z_rotation, q_base_tilt
    ).tolist()
    if angle_diff < 0:  # Clockwise
        pour_angle = np.deg2rad(45)
    else:  # Counter-clockwise
        pour_angle = -np.deg2rad(45)
    # apply pour_angle on ready_pour_rotation using vector[mass_center[0], mass_center[1], 0] as axis:
    pour_axis = np.array([ready_pour_position[0], ready_pour_position[1], 0])
    axis_norm = np.linalg.norm(pour_axis)
    if axis_norm > 1e-6:  # Avoid division by zero
        pour_axis /= axis_norm
        q_pour = trimesh.transformations.quaternion_about_axis(pour_angle, pour_axis)
        pour_rotation1 = trimesh.transformations.quaternion_multiply(
            q_pour, np.array(ready_pour_rotation)
        ).tolist()
        pour_rotation2 = trimesh.transformations.quaternion_multiply(
            q_pour, np.array(pour_rotation1)
        ).tolist()
        pour_rotation3 = trimesh.transformations.quaternion_multiply(
            q_pour, np.array(pour_rotation2)
        ).tolist()
    else:
        # Axis is zero, cannot determine pour direction. Fallback to a default pour.
        raise ValueError(f"axis_norm={axis_norm}")
        # pour_rotation = [-0.271, 0.653, -0.271, 0.653]

    ready_pour_pose = ready_pour_position + ready_pour_rotation
    # pour_pose1 = ready_pour_position + pour_rotation1
    # pour_pose2 = ready_pour_position + pour_rotation2
    pour_pose3 = ready_pour_position + pour_rotation3

    # after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.250]

    release_position = grasp_position[:2] + [grasp_position[2] + 0.005]
    after_release_position = before_grasp_position
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": grasp_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "no_curobo": True,
            "ignore_obstacles": [target_name],
        }
    )
    moves.append({"type": "gripper", "grip_type": "close", "wait_time": 1.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append(
        {
            "type": "arm",
            "goal": ready_pour_pose,
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )
    # moves.append(
    #     {"type": "arm", "goal": pour_pose1, "no_curobo": True, "wait_time": 0.0}
    # )
    # moves.append(
    #     {"type": "arm", "goal": pour_pose2, "no_curobo": True, "wait_time": 0.0}
    # )
    moves.append(
        {"type": "arm", "goal": pour_pose3, "no_curobo": True, "wait_time": 1.0}
    )
    # moves.append(
    #     {"type": "arm", "goal": pour_pose2, "no_curobo": True, "wait_time": 0.0}
    # )
    # moves.append(
    #     {"type": "arm", "goal": pour_pose1, "no_curobo": True, "wait_time": 0.0}
    # )
    moves.append(
        {"type": "arm", "goal": ready_pour_pose, "no_curobo": True, "wait_time": 0.0}
    )
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append(
        {
            "type": "arm",
            "goal": release_position + quaternion_orientation,
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {
            "type": "arm",
            "goal": after_release_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
            "no_curobo": True,
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": after_release_position[:2]
            + [after_release_position[2] + 0.1]
            + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
        }
    )

    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def grab_and_drop(target_name: str, grasp: np.array, args: list, scene_data: dict) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    obstacles = scene_data["obstacles"]
    position = [p * 1000 for p in position]
    # euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    # euler_orientation = np.rad2deg(euler_orientation).tolist()
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific drop point
    # drop_pose = args[0]
    tea_amount = scene_data.get("teapot_tea_amount", 0)

    grasp_angle = 0
    gz_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [0, 0, 1])
    gy_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [0, 1, 0])
    gx_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [1, 0, 0])

    q_base = np.array([0.5, 0.5, 0.5, 0.5])
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gz_rotation, q_base
    ).tolist()
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gy_rotation, grasp_rotation
    ).tolist()
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gx_rotation, grasp_rotation
    ).tolist()

    grasp_position = [p + f * 0 for p, f in zip(position, front, strict=False)]
    before_grasp_position = [grasp_position[0] - 0.05, grasp_position[1], grasp_position[2]]
    after_grasp_position = grasp_position[:2] + [grasp_position[2] + 20]
    drop_position = grasp_position[:2] + [grasp_position[2] + 0.005]
    # forward_signal = HOME_SIGNAL
    # forward_signal[0] += 200
    # moves.append({"type": "move_arm", "goal": forward_signal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 0.5})
    moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": grasp_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
        }
    )
    moves.append({"type": "gripper", "grip_type": "grasp", "wait_time": 1.0})
    moves.append(
        {
            "type": "arm",
            "goal": after_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": drop_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
        }
    )
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    full_act = {
        "moves": moves,
        "obstacles": obstacles,
        "skip_curobo": True,
        "teapot_tea_amount_after": int(max(0, tea_amount)),
    }

    return full_act


def move_to(grasp: np.array, args: list, scene_data: dict) -> list[dict]:
    pose = args[0] + [90, 0, 90]
    moves = []
    moves.append({"type": "move_arm", "goal": pose, "wait_time": 0.0})
    return moves


def move_to_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    pose = args[0] + [0.5, 0.5, 0.5, 0.5]
    obstacles = []
    for obj_name in scene_data["object_infos"]:
        obstacles.append(
            {
                "mass_center": list(
                    np.mean(scene_data["object_infos"][obj_name]["points"], axis=0)
                ),
                "std": list(
                    np.std(scene_data["object_infos"][obj_name]["points"], axis=0)
                ),
            }
        )
    moves = []
    moves.append({"type": "arm", "goal": pose, "wait_time": 0.0})
    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def joints_rad_move_to_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    joints_goal = args[0]
    obstacles = scene_data["obstacles"]
    moves = []
    moves.append({"type": "arm", "joints_goal": joints_goal, "wait_time": 0.0})
    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def open_grip(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = []
    moves = []
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "grasp", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def grab_and_place_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> dict:
    obstacles = scene_data["obstacles"]
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()

    start_position = [
        0.09911725,
        -0.4893,
        2.1495,
        1.5597,
        -1.49436836,
        3.0734,
    ]

    grasp_angle = 0
    gz_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [0, 0, 1])
    gy_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [0, 1, 0])
    gx_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [1, 0, 0])

    q_base = np.array([0.5, 0.5, 0.5, 0.5])
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gz_rotation, q_base
    ).tolist()
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gy_rotation, grasp_rotation
    ).tolist()
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gx_rotation, grasp_rotation
    ).tolist()

    # Grasp Position
    before_grasp_position = [
        p - f * 0.060 for p, f in zip(position, front, strict=False)
    ]
    before_grasp_position = before_grasp_position[:2] + [
        before_grasp_position[2] + 0.08
    ]
    grasp_position = [p - f * 0.030 for p, f in zip(position, front, strict=False)]
    # grasp_position = grasp_position[:2] + [grasp_position[2] + 0.020]
    grasp_position = grasp_position[:2] + [0.002]

    after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.08]

    # ready_pour_position = args[0]
    ready_pour_position = _apply_startup_cup_offset(args[0], scene_data, cup_index=0)
    pour_position = [
        ready_pour_position[0],
        ready_pour_position[1] - 0.05,
        ready_pour_position[2] + 0.070,
    ]

    target_angle = -np.pi / 2

    qz_rotation = trimesh.transformations.quaternion_about_axis(target_angle, [0, 0, 1])
    qy_rotation = trimesh.transformations.quaternion_about_axis(
        -np.arcsin(front[2]), [0, 1, 0]
    )
    q_base = np.array([0.5, 0.5, 0.5, 0.5])
    q_base_tilt = trimesh.transformations.quaternion_multiply(qy_rotation, q_base)
    ready_pour_rotation = trimesh.transformations.quaternion_multiply(
        qz_rotation, q_base_tilt
    ).tolist()

    pourX_rotation = trimesh.transformations.quaternion_about_axis(
        np.deg2rad(50), [1, 0, 0]
    )
    pouring_rotation = trimesh.transformations.quaternion_multiply(
        pourX_rotation, ready_pour_rotation
    ).tolist()

    release_position = [
        grasp_position[0],
        grasp_position[1],
        grasp_position[2] + 0.002,
    ]

    after_release_position = [
        release_position[0] - 0.15,
        release_position[1],
        release_position[2] + 0.28,
    ]

    pre_grasp_moves = []
    pre_grasp_moves.append(
        {"type": "gripper", "grip_type": "tri open", "wait_time": 1.0}
    )
    pre_grasp_moves.append(
        {"type": "arm", "joints_goal": start_position, "wait_time": 0.0}
    )
    pre_grasp_moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    # pre_grasp_moves.append(
    #     {
    #         "type": "arm",
    #         "goal": before_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #         "no_obstacles": "yesyesyes",
    #         "ignore_obstacles": [target_name],
    #     }
    # )
    pre_grasp_moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + grasp_rotation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
        }
    )
    # pre_grasp_moves.append(
    #     {
    #         "type": "arm",
    #         "goal": grasp_position + quaternion_orientation,
    #         "wait_time": 0.5,
    #         "no_obstacles": "yesyesyes",
    #         "no_curobo": True,
    #         "ignore_obstacles": [target_name],
    #     }
    # )

    pre_grasp_moves.append(
        {
            "type": "arm",
            "goal": grasp_position + grasp_rotation,
            "wait_time": 0.5,
            "no_obstacles": "yesyesyes",
            "no_curobo": True,
            "ignore_obstacles": [target_name],
        }
    )

    pre_grasp_moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    pre_grasp_moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})

    # pre_grasp_moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )

    after_grasp_joint_goal = [1.0328, -0.0647, 2.1169, 1.1269, -0.5437, 3.1617]
    pre_grasp_moves.append(
        {"type": "arm", "joints_goal": after_grasp_joint_goal, "wait_time": 0.0}
    )

    pour_moves = []
    pour_moves.append(
        {
            "type": "arm",
            "goal": ready_pour_position + ready_pour_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )

    pour_moves.append(
        {
            "type": "arm",
            "goal": pour_position + pouring_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 4.0,
            "ignore_obstacles": [target_name],
        }
    )

    pour_moves.append(
        {
            "type": "arm",
            "goal": ready_pour_position + ready_pour_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )

    post_pour_moves = []

    tea_amount = int(scene_data.get("teapot_tea_amount", 0))
    tea_capacity = max(1, int(scene_data.get("teapot_capacity", 1)))
    swapped_for_continuation = False
    # If no tea before desired pour, swap first.
    if tea_amount <= 0:
        pre_grasp_moves = append_swap_part1_actions(
            pre_grasp_moves,
            target_name,
            grasp,
            None,
            scene_data,
            insert_before_last=False,
        )
        tea_amount = tea_capacity
        swapped_for_continuation = True

    # This action performs one pour.
    tea_amount -= 1
    if swapped_for_continuation:
        # Complete the swap only after this pour is finished.
        post_pour_moves = append_swap_part2_actions(
            post_pour_moves,
            target_name,
            grasp,
            None,
            scene_data,
        )
    elif tea_amount <= 0:
        # Tea runs out exactly after this pour: swap now for next action.
        post_pour_moves = append_swap_part1_actions(
            post_pour_moves,
            target_name,
            grasp,
            None,
            scene_data,
        )
        post_pour_moves = append_swap_part2_actions(
            post_pour_moves,
            target_name,
            grasp,
            None,
            scene_data,
        )
        tea_amount = tea_capacity
    else:
        post_pour_moves.append(
            {
                "type": "arm",
                "goal": after_grasp_position + grasp_rotation,
                "no_obstacles": "yesyesyes",
                "wait_time": 0.0,
                "ignore_obstacles": [target_name],
            }
        )

        post_pour_moves.append(
            {
                "type": "arm",
                "goal": release_position + grasp_rotation,
                "wait_time": 0.5,
                "no_obstacles": "yesyesyes",
                "ignore_obstacles": [target_name],
            }
        )

        post_pour_moves.append(
            {"type": "gripper", "grip_type": "open", "wait_time": 1.0}
        )
        post_pour_moves.append(
            {
                "type": "arm",
                "goal": after_release_position + grasp_rotation,
                "wait_time": 0.0,
                "no_obstacles": "yesyesyes",
                "ignore_obstacles": [target_name],
                "no_curobo": True,
            }
        )

    post_pour_moves.append(
        {"type": "arm", "joints_goal": start_position, "wait_time": 0.0}
    )
    post_pour_moves.append(
        {"type": "gripper", "grip_type": "tri open", "wait_time": 1.0}
    )

    moves = pre_grasp_moves + pour_moves + post_pour_moves
    for move in moves:
        if move.get("type") == "arm":
            move.setdefault("custom_vel", FAST_ARM_CUSTOM_VEL)
            move.setdefault("custom_acc", FAST_ARM_CUSTOM_ACC)
    full_act = {
        "moves": moves,
        "obstacles": obstacles,
        "skip_curobo": True,
        "teapot_tea_amount_after": int(max(0, tea_amount)),
    }
    return full_act


def grab_and_place_double(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> dict:
    obstacles = scene_data["obstacles"]
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()

    start_position = [
        0.09911725,
        -0.4893,
        2.1495,
        1.5597,
        -1.49436836,
        3.0734,
    ]

    grasp_angle = 0
    gz_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [0, 0, 1])
    gy_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [0, 1, 0])
    gx_rotation = trimesh.transformations.quaternion_about_axis(grasp_angle, [1, 0, 0])

    q_base = np.array([0.5, 0.5, 0.5, 0.5])
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gz_rotation, q_base
    ).tolist()
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gy_rotation, grasp_rotation
    ).tolist()
    grasp_rotation = trimesh.transformations.quaternion_multiply(
        gx_rotation, grasp_rotation
    ).tolist()

    # Grasp Position
    before_grasp_position = [
        p - f * 0.060 for p, f in zip(position, front, strict=False)
    ]
    before_grasp_position = before_grasp_position[:2] + [
        before_grasp_position[2] + 0.08
    ]
    grasp_position = [p - f * 0.020 for p, f in zip(position, front, strict=False)]
    # grasp_position = grasp_position[:2] + [grasp_position[2] + 0.010]
    grasp_position = grasp_position[:2] + [0.040]

    after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.08]

    ready_pour_position = _apply_startup_cup_offset(args[0], scene_data, cup_index=0)
    pour_position = [
        ready_pour_position[0],
        ready_pour_position[1] - 0.05,
        ready_pour_position[2] + 0.070,
    ]

    ready_pour_position_second = _apply_startup_cup_offset(
        args[0], scene_data, cup_index=1
    )
    pour_position_second = [
        ready_pour_position_second[0],
        ready_pour_position_second[1] - 0.05,
        ready_pour_position_second[2] + 0.070,
    ]

    target_angle = -np.pi / 2

    qz_rotation = trimesh.transformations.quaternion_about_axis(target_angle, [0, 0, 1])
    qy_rotation = trimesh.transformations.quaternion_about_axis(
        -np.arcsin(front[2]), [0, 1, 0]
    )
    q_base = np.array([0.5, 0.5, 0.5, 0.5])
    q_base_tilt = trimesh.transformations.quaternion_multiply(qy_rotation, q_base)
    ready_pour_rotation = trimesh.transformations.quaternion_multiply(
        qz_rotation, q_base_tilt
    ).tolist()

    pourX_rotation = trimesh.transformations.quaternion_about_axis(
        np.deg2rad(50), [1, 0, 0]
    )
    pouring_rotation = trimesh.transformations.quaternion_multiply(
        pourX_rotation, ready_pour_rotation
    ).tolist()

    after_pour_position = [ready_pour_position[0] - 0.05] + ready_pour_position[1:]

    after_pour_position = [
        ready_pour_position_second[0] - 0.05
    ] + ready_pour_position_second[1:]

    release_position = [
        grasp_position[0] - 0.010,
        grasp_position[1],
        grasp_position[2] + 0.004,
    ]
    after_release_position = [
        release_position[0] - 0.15,
        release_position[1],
        release_position[2] + 0.28,
    ]

    pre_grasp_moves = []
    pre_grasp_moves.append(
        {"type": "gripper", "grip_type": "tri open", "wait_time": 1.0}
    )
    pre_grasp_moves.append(
        {"type": "arm", "joints_goal": start_position, "wait_time": 0.0}
    )
    pre_grasp_moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    # pre_grasp_moves.append(
    #     {
    #         "type": "arm",
    #         "goal": before_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #         "no_obstacles": "yesyesyes",
    #         "ignore_obstacles": [target_name],
    #     }
    # )
    pre_grasp_moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + grasp_rotation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
        }
    )
    # pre_grasp_moves.append(
    #     {
    #         "type": "arm",
    #         "goal": grasp_position + quaternion_orientation,
    #         "wait_time": 0.5,
    #         "no_obstacles": "yesyesyes",
    #         "no_curobo": True,
    #         "ignore_obstacles": [target_name],
    #     }
    # )
    pre_grasp_moves.append(
        {
            "type": "arm",
            "goal": grasp_position + grasp_rotation,
            "wait_time": 0.5,
            "no_obstacles": "yesyesyes",
            "no_curobo": True,
            "ignore_obstacles": [target_name],
        }
    )

    pre_grasp_moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    pre_grasp_moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})

    # pre_grasp_moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "no_obstacles": "yesyesyes",
    #         "wait_time": 0.0,
    #         "ignore_obstacles": [target_name],
    #     }
    # )

    after_grasp_joint_goal = [1.0328, -0.0647, 2.1169, 1.1269, -0.5437, 3.1617]
    pre_grasp_moves.append(
        {"type": "arm", "joints_goal": after_grasp_joint_goal, "wait_time": 0.0}
    )

    first_pour_moves = []
    first_pour_moves.append(
        {
            "type": "arm",
            "goal": ready_pour_position + ready_pour_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )

    first_pour_moves.append(
        {
            "type": "arm",
            "goal": pour_position + pouring_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 2.5,
            "ignore_obstacles": [target_name],
        }
    )

    first_pour_moves.append(
        {
            "type": "arm",
            "goal": ready_pour_position + ready_pour_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )

    second_pour_moves = []
    second_pour_moves.append(
        {
            "type": "arm",
            "goal": ready_pour_position_second + ready_pour_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )

    second_pour_moves.append(
        {
            "type": "arm",
            "goal": pour_position_second + pouring_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 5.0,
            "ignore_obstacles": [target_name],
        }
    )

    second_pour_moves.append(
        {
            "type": "arm",
            "goal": after_pour_position + ready_pour_rotation,
            "no_obstacles": "yesyesyes",
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )

    post_pour_moves = []

    tea_amount = int(scene_data.get("teapot_tea_amount", 0))
    tea_capacity = max(1, int(scene_data.get("teapot_capacity", 1)))
    swapped_for_continuation = False

    # If no tea before first pour, swap first.
    if tea_amount <= 0:
        pre_grasp_moves = append_swap_part1_actions(
            pre_grasp_moves,
            target_name,
            grasp,
            None,
            scene_data,
            insert_before_last=False,
        )
        tea_amount = tea_capacity
        swapped_for_continuation = True

    # First pour.
    tea_amount -= 1

    # Scenario 1: double pour but tea runs out after first pour.
    if tea_amount <= 0:
        first_pour_moves = append_swap_part1_actions(
            first_pour_moves,
            target_name,
            grasp,
            None,
            scene_data,
        )
        tea_amount = tea_capacity
        swapped_for_continuation = True

    # Second pour.
    tea_amount -= 1

    if swapped_for_continuation:
        # Complete the swap only after this pour sequence is finished.
        post_pour_moves = append_swap_part2_actions(
            post_pour_moves,
            target_name,
            grasp,
            None,
            scene_data,
        )
    elif tea_amount <= 0:
        # Tea runs out exactly after second pour: swap now for next action.
        post_pour_moves = append_swap_part1_actions(
            post_pour_moves,
            target_name,
            grasp,
            None,
            scene_data,
        )
        post_pour_moves = append_swap_part2_actions(
            post_pour_moves,
            target_name,
            grasp,
            None,
            scene_data,
        )
        tea_amount = tea_capacity
    else:
        post_pour_moves.append(
            {
                "type": "arm",
                "goal": after_grasp_position + quaternion_orientation,
                "no_obstacles": "yesyesyes",
                "wait_time": 0.0,
                "ignore_obstacles": [target_name],
            }
        )

        post_pour_moves.append(
            {
                "type": "arm",
                "goal": release_position + quaternion_orientation,
                "wait_time": 0.5,
                "no_obstacles": "yesyesyes",
                "ignore_obstacles": [target_name],
            }
        )

        post_pour_moves.append(
            {"type": "gripper", "grip_type": "open", "wait_time": 1.0}
        )
        post_pour_moves.append(
            {
                "type": "arm",
                "goal": after_release_position + quaternion_orientation,
                "wait_time": 0.0,
                "no_obstacles": "yesyesyes",
                "ignore_obstacles": [target_name],
                "no_curobo": True,
            }
        )

    post_pour_moves.append(
        {"type": "arm", "joints_goal": start_position, "wait_time": 0.0}
    )
    post_pour_moves.append(
        {"type": "gripper", "grip_type": "tri open", "wait_time": 1.0}
    )

    moves = pre_grasp_moves + first_pour_moves + second_pour_moves + post_pour_moves
    for move in moves:
        if move.get("type") == "arm":
            move.setdefault("custom_vel", FAST_ARM_CUSTOM_VEL)
            move.setdefault("custom_acc", FAST_ARM_CUSTOM_ACC)
    full_act = {
        "moves": moves,
        "obstacles": obstacles,
        "skip_curobo": True,
        "teapot_tea_amount_after": int(max(0, tea_amount)),
    }
    return full_act


def grab_bottle_and_place_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> dict:
    obstacles = scene_data["obstacles"]
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()

    # Grasp Position
    before_grasp_position = [
        p - f * 0.050 for p, f in zip(position, front, strict=False)
    ]
    before_grasp_position = before_grasp_position[:2] + [before_grasp_position[2] + 0.05]
    # grasp_position = [p + f * 0.048 for p, f in zip(position, front, strict=False)]
    grasp_position = [p - f * 0 for p, f in zip(position, front, strict=False)]

    release_position = grasp_position[:2] + [grasp_position[2] + 0.005]
    after_release_position = (
        [release_position[0] - 0.05]
        + [release_position[1] - 0.05]
        + [release_position[2]]
    )
    moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": grasp_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "no_curobo": True,
            "ignore_obstacles": [target_name],
        }
    )
    moves.append({"type": "gripper", "grip_type": "grasp", "wait_time": 1.0})

    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": before_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )

    moves.append(
        {
            "type": "arm",
            "goal": release_position + quaternion_orientation,
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {
            "type": "arm",
            "goal": after_release_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "ignore_obstacles": [target_name],
            "no_curobo": True,
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": after_release_position[:2]
            + [after_release_position[2] + 0.1]
            + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
        }
    )

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def grab_and_rotate(target_name: str, grasp: np.array, args: list, scene_data: dict):
    obstacles = scene_data["obstacles"]
    moves = []
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()

    before_grasp_position = [
        p - f * 0.050 for p, f in zip(position, front, strict=False)
    ]
    grasp_position = [p + f * 0.048 for p, f in zip(position, front, strict=False)]

    # release position is the same as grasp position, but with 90 degree rotation around z axis
    q_z_rotation = trimesh.transformations.quaternion_about_axis(
        np.deg2rad(args[0]), [0, 0, 1]
    )
    release_rotation = trimesh.transformations.quaternion_multiply(
        q_z_rotation, quaternion_orientation
    ).tolist()
    release_position = grasp_position
    # after_release_position = grasp_position[:2] + [grasp_position[2] + 0.005]

    moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": grasp_position + quaternion_orientation,
            "wait_time": 0.0,
            "no_obstacles": "yesyesyes",
            "no_curobo": True,
            "ignore_obstacles": [target_name],
        }
    )
    # moves.append({"type": "gripper", "grip_type": "close", "wait_time": 1.0})
    moves.append(
        {
            "type": "arm",
            "goal": release_position + release_rotation,
            "wait_time": 0.0,
            "ignore_obstacles": [target_name],
        }
    )
    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_release_position + release_rotation,
    #         "wait_time": 0.0,
    #         "no_obstacles": "yesyesyes",
    #         "ignore_obstacles": [target_name],
    #         "no_curobo": True,
    #     }
    # )
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_release_position[:2]
    #         + [after_release_position[2] + 0.1]
    #         + release_rotation,
    #         "wait_time": 0.0,
    #         "no_obstacles": "yesyesyes",
    #     }
    # )

    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def gripper_test(target_name: str, grasp: np.array, args: list, scene_data: dict):
    obstacles = scene_data["obstacles"]
    moves = []

    # quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))

    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": args[0] + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "grasp", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "tri close", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 1.0})
    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_pour_tealeaf(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    ready_joint_goal = args[0]
    before_grasp_joint_goal = args[1]
    grasp_joint_goal = args[2]
    after_grasp_joint_goal = args[3]
    ready_pour_joint_goal = args[4]
    pour_joint_goal = args[5]

    # moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "grasp", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": after_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": ready_pour_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": pour_joint_goal, "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": ready_pour_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": after_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_pour_hotwater(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    ready_joint_goal = args[0]
    before_grasp_joint_goal = args[1]
    grasp_joint_goal = args[2]
    ready_pour_joint_goal = args[3]
    pour_joint_goal = args[4]

    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "grasp", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": ready_pour_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": pour_joint_goal, "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": ready_pour_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    # moves.append({"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_grasp_filter(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    ready_joint_goal = args[0]
    before_grasp_joint_goal = args[1]
    grasp_joint_goal = args[2]
    ready_place_joint_goal = args[3]
    place_joint_goal = args[4]

    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 0.0})
    # moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 10.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri close", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": place_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 0.0})
    moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_putback_filter(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    # ready_joint_goal = args[0]
    before_grasp_joint_goal = args[1]
    grasp_joint_goal = args[2]
    ready_place_joint_goal = args[3]
    place_joint_goal = args[4]

    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 0.0})
    # moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 10.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri close", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": place_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 0.0})
    # moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_grasp_lid(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    ready_joint_goal = args[0]
    before_grasp_joint_goal = args[1]
    grasp_joint_goal = args[2]
    ready_place_joint_goal = args[3]
    place_joint_goal = args[4]

    moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})
    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 10.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri close", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": place_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    # moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_putback_lid(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    ready_joint_goal = args[0]
    before_grasp_joint_goal = args[1]
    grasp_joint_goal = args[2]
    ready_place_joint_goal = args[3]
    place_joint_goal = args[4]

    # moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})
    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 10.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri close", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": place_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "tri open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": ready_place_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": ready_joint_goal, "wait_time": 0.0})

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_swap_teapot(
    target_name: str,
    grasp: np.array,
    args: list,
    scene_data: dict,
    putdown: bool = False,
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    before_temp_joint_goal = args[0]
    temp_joint_goal = args[1]
    before_grasp_joint_goal = args[2]
    grasp_joint_goal = args[3]
    before_place_joint_goal = args[4]
    place_joint_goal = args[5]

    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": temp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": before_place_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": place_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_place_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": temp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_swap_teapot_part1(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    before_temp_joint_goal = args[0]
    temp_joint_goal = args[1]
    before_grasp_joint_goal = args[2]
    grasp_joint_goal = args[3]

    # Part 1: move to and grab the spare teapot.
    # moves.append({"type": "gripper", "grip_type": "open", "wait_time": 0.0})
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": temp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


def joints_rad_swap_teapot_part2(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    obstacles = scene_data["obstacles"]
    moves = []

    before_temp_joint_goal = args[0]
    temp_joint_goal = args[1]
    before_grasp_joint_goal = args[2]
    grasp_joint_goal = args[3]
    before_place_joint_goal = args[4]
    place_joint_goal = args[5]

    # Part 2: place replacement teapot, then fetch original back.
    moves.append(
        {"type": "arm", "joints_goal": before_place_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": place_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_place_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": temp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "hook", "wait_time": 1.0})
    moves.append({"type": "gripper", "grip_type": "aid", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_temp_joint_goal, "wait_time": 0.0}
    )
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )
    moves.append({"type": "arm", "joints_goal": grasp_joint_goal, "wait_time": 0.0})
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {"type": "arm", "joints_goal": before_grasp_joint_goal, "wait_time": 0.0}
    )

    full_act = {"moves": moves, "obstacles": obstacles, "skip_curobo": True}
    return full_act


action_dict = {
    "grab_and_pour_and_place_back": grab_and_pour_and_place_back,
    "grab_and_pour_and_place_back_curobo": grab_and_pour_and_place_back_curobo_by_rotation,
    "grab_and_drop": grab_and_drop,
    "move_to": move_to,
    "move_to_curobo": move_to_curobo,
    "joints_rad_move_to_curobo": joints_rad_move_to_curobo,
    "open_grip": open_grip,
    "grab_and_place_curobo": grab_and_place_curobo,
    "grab_and_place_double": grab_and_place_double,
    "grab_bottle_and_place_curobo": grab_bottle_and_place_curobo,
    "grab_and_rotate": grab_and_rotate,
    "gripper_test": gripper_test,
    "joints_rad_pour_tealeaf": joints_rad_pour_tealeaf,
    "joints_rad_pour_hotwater": joints_rad_pour_hotwater,
    "joints_rad_grasp_filter": joints_rad_grasp_filter,
    "joints_rad_putback_filter": joints_rad_putback_filter,
    "joints_rad_grasp_lid": joints_rad_grasp_lid,
    "joints_rad_putback_lid": joints_rad_putback_lid,
    "joints_rad_swap_teapot": joints_rad_swap_teapot,
    "joints_rad_swap_teapot_part1": joints_rad_swap_teapot_part1,
    "joints_rad_swap_teapot_part2": joints_rad_swap_teapot_part2,
}


def act(action: str, grasp: np.array, args: list, scene_data: dict) -> list[dict]:
    if action not in action_dict:
        logger.error(f"There is no such action: {action}")
    action_method = action_dict[action]
    return action_method(grasp, args, scene_data)


def act_with_name(
    action: str,
    target_name: str,
    grasps: list[np.ndarray],
    args: list,
    scene_data: dict,
) -> list[dict]:
    if action not in action_dict:
        logger.error(f"There is no such action: {action}")
    action_method = action_dict[action]
    return [action_method(target_name, grasp, args, scene_data) for grasp in grasps]


def append_swap_actions(
    act_list: list[dict],
    target_name: str,
    grasp: np.array,
    args: list | None,
    scene_data: dict,
    insert_before_last: bool = True,
) -> list[dict]:
    swap_args = args
    if not isinstance(swap_args, list) or len(swap_args) == 0:
        candidate = scene_data.get("teapot_swap_action_args")
        swap_args = candidate if isinstance(candidate, list) else []
    if len(swap_args) == 0:
        logger.warning("No teapot swap args found in scene_data; skip swap insertion")
        return act_list

    swap_actions = joints_rad_swap_teapot(target_name, grasp, swap_args, scene_data)[
        "moves"
    ]
    if len(act_list) == 0:
        return swap_actions
    if not insert_before_last:
        return swap_actions + act_list

    # Insert swap actions before the last action (which is usually moving back to ready position)
    return act_list + swap_actions


def append_swap_part1_actions(
    act_list: list[dict],
    target_name: str,
    grasp: np.array,
    args: list | None,
    scene_data: dict,
    insert_before_last: bool = True,
) -> list[dict]:
    swap_args = args
    if not isinstance(swap_args, list) or len(swap_args) == 0:
        candidate = scene_data.get("teapot_swap_action_args")
        swap_args = candidate if isinstance(candidate, list) else []
    if len(swap_args) == 0:
        logger.warning(
            "No teapot swap args found in scene_data; skip part1 swap insertion"
        )
        return act_list

    swap_actions = joints_rad_swap_teapot_part1(
        target_name, grasp, swap_args, scene_data
    )["moves"]
    if len(act_list) == 0:
        return swap_actions
    if not insert_before_last:
        return swap_actions + act_list
    return act_list + swap_actions


def append_swap_part2_actions(
    act_list: list[dict],
    target_name: str,
    grasp: np.array,
    args: list | None,
    scene_data: dict,
    insert_before_last: bool = True,
) -> list[dict]:
    swap_args = args
    if not isinstance(swap_args, list) or len(swap_args) == 0:
        candidate = scene_data.get("teapot_swap_action_args")
        swap_args = candidate if isinstance(candidate, list) else []
    if len(swap_args) == 0:
        logger.warning(
            "No teapot swap args found in scene_data; skip part2 swap insertion"
        )
        return act_list

    swap_actions = joints_rad_swap_teapot_part2(
        target_name, grasp, swap_args, scene_data
    )["moves"]
    if len(act_list) == 0:
        return swap_actions
    if not insert_before_last:
        return swap_actions + act_list
    return act_list + swap_actions
