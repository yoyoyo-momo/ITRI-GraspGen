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

def teapot_body_qualifier(grasp: np.array, min_point: np.ndarray, max_point: np.ndarray):
    """Qualifier for grasping teapot body (top-down on the barrel)"""
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    
    center = (min_point + max_point) / 2.0
    size = max_point - min_point
    
    # Prefer top-down approach
    if front[2] > -0.5:
        return False
    
    # Gripper should be mostly upright
    if up[2] < 0.6:
        return False
    
    # Keep grasp near center in XY (avoid handle and spout)
    dist_xy = np.linalg.norm(np.array(position[:2]) - center[:2])
    max_radius = np.linalg.norm(size[:2]) / 3.0  # Central third of XY footprint
    if dist_xy > max_radius:
        return False
    
    # Z within mid-height band (avoid lid and base)
    min_z = min_point[2] + size[2] * 0.25
    max_z = max_point[2] - size[2] * 0.15
    if position[2] < min_z or position[2] > max_z:
        return False
    
    return True


def teapot_handle_qualifier(grasp: np.array, min_point: np.ndarray, max_point: np.ndarray):
    """Qualifier for grasping teapot handle (side approach)"""
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    
    center = (min_point + max_point) / 2.0
    size = max_point - min_point
    
    # Prefer side/horizontal approach (not top-down, not straight-on)
    front_xy_norm = np.linalg.norm(front[:2])
    if front_xy_norm < 0.3:  # mostly vertical, reject
        return False
    
    # Allow some upward tilt but not too steep
    if front[2] < -0.3 or front[2] > 0.3:
        return False
    
    # Gripper moderately upright
    if up[2] < 0.3:
        return False
    
    # Keep grasp far from center in XY (prefer periphery where handle is)
    dist_xy = np.linalg.norm(np.array(position[:2]) - center[:2])
    min_radius = np.linalg.norm(size[:2]) / 4.0  # At least outer quarter
    if dist_xy < min_radius:
        return False
    
    # Z within mid-height band (handles typically mid-body)
    min_z = min_point[2] + size[2] * 0.2
    max_z = max_point[2] - size[2] * 0.2
    if position[2] < min_z or position[2] > max_z:
        return False
    
    return True


qualifier_dict = {
    "small_cup_qualifier": small_cup_qualifier,
    "cup_qualifier": cup_qualifier,
    "small_cube_qualifier": small_cube_qualifier,
    "teapot_body_qualifier": teapot_body_qualifier,
    "teapot_handle_qualifier": teapot_handle_qualifier,
}


def is_qualified(
    grasp: np.array, qualifier: str, min_point: np.ndarray, max_point: np.ndarray
) -> bool:
    if qualifier not in qualifier_dict:
        logger.error(f"There is no such qualifier: {qualifier}")
        raise KeyError
    qualification_method = qualifier_dict[qualifier]
    return qualification_method(grasp, min_point, max_point)
