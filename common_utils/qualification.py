import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_left_up_and_front(grasp: np.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = grasp[:3, 0]
    up = grasp[:3, 1]
    front = grasp[:3, 2]
    return left, up, front


def cup_qualifier(grasp: np.array, min_point: np.ndarray, max_point: np.ndarray):
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    position += front * 0.20  # offset
    if abs(left[2]) > 0.2:
        return False
    if up[2] < 0.85:
        return False
    # Rule: planar 2D angle between grasp approach (front) vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(90):
        return False

    if position[2] < 0.05:  # for safety
        return False
    if position[2] > min_point[2] + (max_point[2] - min_point[2]) * 0.75:  # too high
        return False
    if position[2] < min_point[2] + (max_point[2] - min_point[2]) * 0.3:  # too low
        return False
    return True


def small_cup_qualifier(grasp: np.array, mass_center, obj_std):
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    if up[2] < 0.7:
        return False
    # Rule: planar 2D angle between grasp approach (front) vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(90):
        return False

    # if position[2] < 0.05:  # for safety
    #     return False
    # if position[2] > mass_center[2] + obj_std[2] * 2:  # too high
    #     return False
    # if position[2] < mass_center[2] - obj_std[2] * 1.5:  # too low
    #     return False
    return True  #


def small_cube_qualifier(grasp: np.array, mass_center, obj_std):
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    if front[0] < 0:
        return False
    if front[2] > -0.2:  # not facing down
        return False
    # Rule: planar 2D angle between grasp approach (front) vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(30):
        return False

    if position[2] < 0.05:  # for safety
        return False
    if position[2] < mass_center[2] - obj_std[2]:  # too low
        return False
    return True


def teapot_body_qualifier(
    grasp: np.array, min_point: np.ndarray, max_point: np.ndarray
):
    """Qualifier for grasping teapot body (top-down on the barrel)"""
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)

    center = (min_point + max_point) / 2.0
    size = max_point - min_point

    # Prefer downward approach but be lenient
    if front[2] > -0.2:  # relaxed from -0.5
        return False

    # Gripper orientation check - more lenient
    if up[2] < 0.3:  # relaxed from 0.6
        return False

    # Keep grasp somewhat near center in XY but be generous
    dist_xy = np.linalg.norm(np.array(position[:2]) - center[:2])
    max_radius = np.linalg.norm(size[:2]) / 1.5  # relaxed from /3.0 - allow outer half
    if dist_xy > max_radius:
        return False

    # Wide Z band
    min_z = min_point[2] + size[2] * 0.1  # relaxed from 0.25
    max_z = max_point[2]  # relaxed - allow top
    if position[2] < min_z or position[2] > max_z:
        return False

    return True


def teapot_handle_qualifier(
    grasp: np.array, min_point: np.ndarray, max_point: np.ndarray
):
    """Qualifier for grasping teapot handle (side approach)"""
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)

    center = (min_point + max_point) / 2.0
    size = max_point - min_point

    # Prefer side/horizontal approach but be lenient
    front_xy_norm = np.linalg.norm(front[:2])
    if front_xy_norm < 0.15:  # relaxed from 0.3 - mostly vertical, reject
        return False

    # Allow wider range of approach angles
    if front[2] < -0.7 or front[2] > 0.5:  # relaxed from -0.3 to 0.3
        return False

    # Gripper orientation - very lenient
    if up[2] < 0.0:  # relaxed from 0.3 - just not completely upside down
        return False

    # Keep grasp at outer edges but be generous
    dist_xy = np.linalg.norm(np.array(position[:2]) - center[:2])
    min_radius = (
        np.linalg.norm(size[:2]) / 6.0
    )  # relaxed from /4.0 - outer sixth is fine
    if dist_xy < min_radius:
        return False

    # Wide Z band
    min_z = min_point[2] + size[2] * 0.1  # relaxed from 0.2
    max_z = max_point[2] - size[2] * 0.1  # relaxed from 0.2
    if position[2] < min_z or position[2] > max_z:
        return False

    return True


def teapot_qualifier(grasp: np.array, min_point: np.ndarray, max_point: np.ndarray):
    """Qualifier for grasping teapot (body or handle)"""
    # position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)

    # Prevent top-down approach
    if up[2] < 0.4:
        return False
    
    # Prevent coming from the front (spout) side
    if front[0] < 0:
        return False

    # Prefer handle-like approach (side approach)
    # return teapot_handle_qualifier(grasp, min_point, max_point)
    return True


qualifier_dict = {
    "small_cup_qualifier": small_cup_qualifier,
    "cup_qualifier": cup_qualifier,
    "small_cube_qualifier": small_cube_qualifier,
    "teapot_body_qualifier": teapot_body_qualifier,
    "teapot_handle_qualifier": teapot_handle_qualifier,
    "teapot_qualifier": teapot_qualifier,
}


def is_qualified(
    grasp: np.array, qualifier: str, min_point: np.ndarray, max_point: np.ndarray
) -> bool:
    if qualifier not in qualifier_dict:
        logger.error(f"There is no such qualifier: {qualifier}")
        raise KeyError
    qualification_method = qualifier_dict[qualifier]
    return qualification_method(grasp, min_point, max_point)
