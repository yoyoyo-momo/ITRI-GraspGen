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


def teapot_lid_qualifier(grasp: np.array, min_point: np.ndarray, max_point: np.ndarray):
    """Qualifier for grasping teapot lid (top-down on the barrel) - MORE VERTICAL"""
    # position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)

    # Enforce VERTICAL gripper (inverted to grasp from above)
    if up[2] > 0.2:  # Green must point mostly downward (inverted)
        return False

    # Enforce DOWNWARD approach (blue pointing DOWN)
    if front[0] < -0.1:  # Blue must point downward
        return False

    # Must not be tilted too much
    if abs(left[2]) > 0.1:
        return False
    return True


def teapot_qualifier(grasp: np.array, min_point: np.ndarray, max_point: np.ndarray):
    """Qualifier for grasping teapot (body or handle)"""
    # position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)

    # Prevent top-down approach
    if up[2] < 0.8:
        return False

    # Prevent coming from the front (spout) side
    if front[0] < 0:
        return False

    return True


def dummy_qualifier(grasp: np.array, min_point: np.ndarray, max_point: np.ndarray):
    return True


qualifier_dict = {
    "small_cup_qualifier": small_cup_qualifier,
    "cup_qualifier": cup_qualifier,
    "small_cube_qualifier": small_cube_qualifier,
    "teapot_lid_qualifier": teapot_lid_qualifier,
    "teapot_qualifier": teapot_qualifier,
    "dummy_qualifier": dummy_qualifier,
}


def is_qualified(
    grasp: np.array, qualifier: str, min_point: np.ndarray, max_point: np.ndarray
) -> bool:
    if qualifier not in qualifier_dict:
        logger.error(f"There is no such qualifier: {qualifier}")
        raise KeyError
    qualification_method = qualifier_dict[qualifier]
    return qualification_method(grasp, min_point, max_point)
