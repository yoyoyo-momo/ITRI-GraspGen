#!/usr/bin/env python3
"""
Test script to verify dual camera setup
Usage:
    # Test with real cameras
    python test_dual_cameras.py --camera-serial-main 12345678 --camera-serial-batter 87654321
    
    # Test with PNG images
    python test_dual_cameras.py --use-png-main "scene1" --use-png-batter "scene2"
"""

import argparse
import cv2
import logging
from PointCloud_Generation.zed_utils import ZedCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Test dual camera setup")
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
    return parser.parse_args()


def main():
    args = parse_args()
    
    cameras = {}
    
    # Initialize main camera
    logger.info("=" * 60)
    logger.info("Initializing MAIN camera...")
    logger.info("=" * 60)
    try:
        cameras["main"] = ZedCamera(
            use_png=args.use_png_main,
            camera_serial=args.camera_serial_main,
            camera_id="main"
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
            camera_id="batter"
        )
        logger.info("✓ Batter camera initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize batter camera: {e}")
        cameras["main"].close()
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("Both cameras are OPEN and READY!")
    logger.info("=" * 60)
    
    # Test capturing from both cameras
    cv2.namedWindow("Camera Test")
    
    current_camera = "main"
    logger.info(f"\nStarting with '{current_camera}' camera")
    logger.info("Press 'M' for main camera, 'B' for batter camera, 'Q' to quit")
    
    while True:
        camera = cameras[current_camera]
        
        # Capture images
        status, left_image, right_image = camera.capture_images()
        
        if status == 0:  # sl.ERROR_CODE.SUCCESS
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
                2
            )
            cv2.putText(
                display_img,
                "M=Main | B=Batter | Q=Quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1
            )
            
            cv2.imshow("Camera Test", display_img)
        else:
            logger.error(f"Failed to capture from {current_camera} camera")
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q', or ESC
            logger.info("\nExiting...")
            break
        elif key == ord('m') or key == ord('M'):
            if current_camera != "main":
                current_camera = "main"
                logger.info(f"\n→ Switched to '{current_camera}' camera")
        elif key == ord('b') or key == ord('B'):
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
