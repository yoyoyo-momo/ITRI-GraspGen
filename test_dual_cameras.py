#!/usr/bin/env python3
"""
Test script to verify dual camera setup and run GraspGen
Usage:
    # Test with real cameras
    python test_dual_cameras.py --camera-serial-main 12345678 --camera-serial-batter 87654321

    # Test with PNG images
    python test_dual_cameras.py --use-png-main "scene1" --use-png-batter "scene2"

    # Test GraspGen on both cameras
    python test_dual_cameras.py --use-png-main "scene1" --use-png-batter "scene2" --test-graspgen
"""

import argparse
import cv2
import logging
import numpy as np
import torch
from PointCloud_Generation.zed_utils import ZedCamera
from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from common_utils.graspgen_utils import GraspGenerator
from common_utils import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Test dual camera setup and GraspGen")
    parser.add_argument(
        "--camera-serial-main",
        type=int,
        default=None,
        help="Serial number of main ZED camera",
    )
    parser.add_argument(
        "--camera-serial-batter",
        type=int,
        default=None,
        help="Serial number of batter ZED camera",
    )
    parser.add_argument(
        "--use-png-main",
        type=str,
        default="",
        help="Use PNG images for main camera (path in sample_data/zed_images/)",
    )
    parser.add_argument(
        "--use-png-batter",
        type=str,
        default="",
        help="Use PNG images for batter camera (path in sample_data/zed_images/)",
    )
    parser.add_argument(
        "--test-graspgen",
        action="store_true",
        help="Test GraspGen on both cameras",
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
        help="Threshold for valid grasps",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=5,
        help="Number of top grasps to return",
    )
    return parser.parse_args()


def test_graspgen_on_dual_cameras(pc_generator, args):
    """Test GraspGen on point clouds from both cameras using PointCloudGenerator"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING GRASPGEN ON DUAL CAMERAS")
    logger.info("=" * 60)

    # Initialize GraspGen
    logger.info("Initializing GraspGen...")
    grasp_generator = GraspGenerator(
        args.gripper_config, args.grasp_threshold, args.num_grasps, args.topk_num_grasps
    )
    logger.info("✓ GraspGen initialized\n")

    results = {}

    for camera_id in ["main", "batter"]:
        logger.info(f"--- Testing {camera_id.upper()} camera ---")

        try:
            # Use PointCloudGenerator to get point cloud from camera
            logger.info(f"Generating point cloud from {camera_id} camera...")
            
            # Generate point cloud for "cup" target from specified camera
            scene_data = pc_generator.generate_pointcloud(
                target_names=["cup"],
                need_confirm=False,
                camera_id=camera_id
            )

            if scene_data is None or "object_infos" not in scene_data:
                logger.warning(f"⚠ Failed to generate point cloud from {camera_id} camera\n")
                continue

            # Get the cup point cloud (first object)
            if not scene_data["object_infos"] or len(scene_data["object_infos"]) == 0:
                logger.warning(f"⚠ No objects detected in {camera_id} camera\n")
                continue

            # Extract point cloud
            if isinstance(scene_data["object_infos"], list):
                # List format
                object_info = scene_data["object_infos"][0]
                pointcloud = object_info["points"]
            else:
                # Dict format
                pointcloud = list(scene_data["object_infos"].values())[0]["points"]

            if len(pointcloud) < 100:
                logger.warning(
                    f"⚠ Too few valid points ({len(pointcloud)}) from {camera_id} camera\n"
                )
                continue

            logger.info(f"✓ Generated point cloud with {len(pointcloud)} points")

            # Run GraspGen
            logger.info(f"Running GraspGen inference on {camera_id} camera...")
            grasp = grasp_generator.auto_select_valid_cup_grasp(pointcloud)

            if grasp is not None:
                grasp_pos = grasp[:3, 3]
                logger.info(f"✓ Valid grasp found from {camera_id} camera!")
                logger.info(
                    f"  Grasp position: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]"
                )

                results[camera_id] = {
                    'grasp': grasp,
                    'grasp_pos': grasp_pos,
                    'num_points': len(pointcloud),
                    'success': True
                }
            else:
                logger.warning(f"✗ No valid grasp found from {camera_id} camera\n")
                results[camera_id] = {'success': False}

        except Exception as e:
            logger.error(f"✗ Error processing {camera_id} camera: {e}\n")
            results[camera_id] = {'success': False}

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if len([r for r in results.values() if r.get("success")]) == 2:
        logger.info("✓ Both cameras successfully generated valid grasps!")

        main_pos = results["main"]["grasp_pos"]
        batter_pos = results["batter"]["grasp_pos"]
        pos_diff = np.linalg.norm(main_pos - batter_pos)

        logger.info(
            f"\nGrasp position difference: {pos_diff:.4f}m ({pos_diff * 100:.2f}cm)"
        )
        logger.info(
            f"Main camera:    [{main_pos[0]:.3f}, {main_pos[1]:.3f}, {main_pos[2]:.3f}]"
        )
        logger.info(
            f"Batter camera:  [{batter_pos[0]:.3f}, {batter_pos[1]:.3f}, {batter_pos[2]:.3f}]"
        )

    elif len([r for r in results.values() if r.get("success")]) == 1:
        logger.info("⚠ Only one camera generated a valid grasp")
        if results.get("main", {}).get("success"):
            logger.info("✓ Main camera succeeded")
        else:
            logger.info("✓ Batter camera succeeded")
    else:
        logger.info("✗ Neither camera generated a valid grasp")

    return results


def main():
    args = parse_args()

    # Initialize PointCloudGenerator
    logger.info("=" * 60)
    logger.info("Initializing PointCloudGenerator...")
    logger.info("=" * 60)
    try:
        pc_generator = PointCloudGenerator(args)
        logger.info("✓ PointCloudGenerator initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize PointCloudGenerator: {e}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("Both cameras are OPEN and READY!")
    logger.info("=" * 60)

    # If testing GraspGen, run the test and exit
    if args.test_graspgen:
        test_graspgen_on_dual_cameras(pc_generator, args)
        logger.info("\nGraspGen dual camera test completed!")
        return

    # Test capturing from both cameras (original functionality)
    cameras = {}
    
    # Initialize main camera
    logger.info("=" * 60)
    logger.info("Initializing MAIN camera...")
    logger.info("=" * 60)
    try:
        cameras["main"] = ZedCamera(
            use_png=args.use_png_main,
            camera_serial=args.camera_serial_main,
            camera_id="main",
        )
        logger.info("✓ Main camera initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize main camera: {e}")
        return

    # Initialize batter camera
    logger.info("\n" + "=" * 60)
    logger.info("Initializing BATTER camera...")
    logger.info("=" * 60)
    try:
        cameras["batter"] = ZedCamera(
            use_png=args.use_png_batter,
            camera_serial=args.camera_serial_batter,
            camera_id="batter",
        )
        logger.info("✓ Batter camera initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize batter camera: {e}")
        cameras["main"].close()
        return

    cv2.namedWindow("Camera Test")

    current_camera = "main"
    logger.info(f"\nStarting with '{current_camera}' camera")
    logger.info("Press 'M' for main camera, 'B' for batter camera, 'Q' to quit")

    while True:
        camera = cameras[current_camera]

        # Capture images
        status, left_image, right_image = camera.capture_images()

        if status != 0:  # sl.ERROR_CODE.SUCCESS
            # Get image data
            left_img = left_image.get_data()[:, :, :3]  # Drop alpha channel

            # Add text overlay
            display_img = left_img.copy()
            cv2.putText(
                display_img,
                f"Camera: {current_camera.upper()}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_img,
                "M=Main | B=Batter | Q=Quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

            cv2.imshow("Camera Test", display_img)
        else:
            logger.error(f"Failed to capture from {current_camera} camera")

        # Handle key presses
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q") or key == 27:  # 'q', 'Q', or ESC
            logger.info("\nExiting...")
            break
        elif key == ord("m") or key == ord("M"):
            if current_camera != "main":
                current_camera = "main"
                logger.info(f"\n→ Switched to '{current_camera}' camera")
        elif key == ord("b") or key == ord("B"):
            if current_camera != "batter":
                current_camera = "batter"
                logger.info(f"\n→ Switched to '{current_camera}' camera")

    # Cleanup
    logger.info("\n" + "=" * 60)
    logger.info("Closing cameras...")
    logger.info("=" * 60)
    for camera_id, camera in cameras.items():
        camera.close()
        logger.info(f"✓ Closed {camera_id} camera")

    cv2.destroyAllWindows()
    logger.info("\nTest completed successfully!")


if __name__ == "__main__":
    main()
