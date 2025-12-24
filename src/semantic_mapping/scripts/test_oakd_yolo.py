#!/usr/bin/env python3
"""
OAK-D YOLO Test - Standalone test script for OAK-D with YOLOv8

Run this BEFORE deploying to TurtleBot4 to verify:
1. OAK-D hardware is working
2. YOLO blob file is correctly converted
3. Detection FPS meets requirements
4. Depth estimation is accurate

Usage:
    python3 test_oakd_yolo.py --blob yolov8n_6shaves.blob

Requirements:
    pip install depthai opencv-python numpy
"""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Colors for visualization (BGR)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]


def create_pipeline(blob_path: str, fps: int = 15, 
                    conf_threshold: float = 0.5,
                    preview_size: tuple = (640, 480),
                    nn_size: tuple = (640, 640)):
    """Create OAK-D pipeline with YOLO detection"""
    import depthai as dai
    
    pipeline = dai.Pipeline()
    
    # ============ RGB Camera ============
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(preview_size[0], preview_size[1])
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(fps)
    
    # ============ Stereo Depth ============
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(preview_size[0], preview_size[1])
    
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    # ============ Image Manip (resize for NN) ============
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(nn_size[0], nn_size[1])
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    cam_rgb.preview.link(manip.inputImage)
    
    # ============ YOLO Neural Network ============
    nn = pipeline.create(dai.node.YoloDetectionNetwork)
    nn.setBlobPath(blob_path)
    nn.setConfidenceThreshold(conf_threshold)
    nn.setNumClasses(80)
    nn.setCoordinateSize(4)
    nn.setAnchors([])  # YOLOv8 is anchor-free
    nn.setAnchorMasks({})
    nn.setIouThreshold(0.5)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    
    manip.out.link(nn.input)
    
    # ============ Outputs ============
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    nn.out.link(xout_nn.input)
    
    return pipeline


def get_depth_at_bbox(depth_frame: np.ndarray, bbox: tuple, 
                      margin_ratio: float = 0.2) -> float:
    """Get median depth value within bounding box center region"""
    x1, y1, x2, y2 = bbox
    
    # Use center region only
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    margin_x = int((x2 - x1) * margin_ratio)
    margin_y = int((y2 - y1) * margin_ratio)
    
    x1_roi = max(0, cx - margin_x)
    x2_roi = min(depth_frame.shape[1], cx + margin_x)
    y1_roi = max(0, cy - margin_y)
    y2_roi = min(depth_frame.shape[0], cy + margin_y)
    
    roi = depth_frame[y1_roi:y2_roi, x1_roi:x2_roi]
    valid = roi[(roi > 100) & (roi < 10000)]  # 10cm to 10m in mm
    
    if len(valid) == 0:
        return 0.0
    
    return float(np.median(valid)) / 1000.0  # Convert to meters


def main():
    parser = argparse.ArgumentParser(description='Test OAK-D YOLO detection')
    parser.add_argument('--blob', type=str, default='yolov8n_6shaves.blob',
                        help='Path to YOLO blob file')
    parser.add_argument('--fps', type=int, default=15,
                        help='Camera FPS')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--preview-size', type=int, nargs=2, default=[640, 480],
                        help='Preview size (width height)')
    parser.add_argument('--nn-size', type=int, nargs=2, default=[640, 640],
                        help='NN input size (width height)')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth estimation')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save output to video file')
    args = parser.parse_args()
    
    # Check blob file
    if not Path(args.blob).exists():
        print(f"❌ Blob file not found: {args.blob}")
        print("\nTo convert YOLOv8 to blob:")
        print("  1. yolo export model=yolov8n.pt format=onnx")
        print("  2. Go to https://blobconverter.luxonis.com/")
        print("  3. Upload yolov8n.onnx, select 6 shaves")
        print("  4. Download and rename to yolov8n_6shaves.blob")
        return
    
    try:
        import depthai as dai
    except ImportError:
        print("❌ depthai not installed. Run: pip install depthai")
        return
    
    print("=" * 60)
    print("OAK-D YOLO Detection Test")
    print("=" * 60)
    print(f"  Blob file: {args.blob}")
    print(f"  Camera FPS: {args.fps}")
    print(f"  Confidence: {args.conf}")
    print(f"  Preview size: {args.preview_size}")
    print(f"  NN input size: {args.nn_size}")
    print(f"  Depth enabled: {not args.no_depth}")
    print("=" * 60)
    
    # Create pipeline
    print("\n🔧 Creating pipeline...")
    pipeline = create_pipeline(
        blob_path=args.blob,
        fps=args.fps,
        conf_threshold=args.conf,
        preview_size=tuple(args.preview_size),
        nn_size=tuple(args.nn_size)
    )
    
    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, args.fps,
            (args.preview_size[0], args.preview_size[1])
        )
    
    # FPS tracking
    fps_history = []
    detection_counts = []
    
    print("🚀 Starting OAK-D device...")
    
    with dai.Device(pipeline) as device:
        print(f"✅ OAK-D connected: {device.getMxId()}")
        print("\nPress 'q' to quit, 's' to save screenshot\n")
        
        # Get queues
        rgb_q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        depth_q = device.getOutputQueue("depth", maxSize=4, blocking=False)
        det_q = device.getOutputQueue("detections", maxSize=4, blocking=False)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame_start = time.time()
            
            # Get frames
            in_rgb = rgb_q.tryGet()
            in_depth = depth_q.tryGet() if not args.no_depth else None
            in_det = det_q.tryGet()
            
            if in_rgb is None:
                continue
            
            rgb_frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame() if in_depth else None
            
            # Process detections
            detections = in_det.detections if in_det else []
            detection_counts.append(len(detections))
            
            # Draw detections
            for i, det in enumerate(detections):
                x1 = int(det.xmin * rgb_frame.shape[1])
                y1 = int(det.ymin * rgb_frame.shape[0])
                x2 = int(det.xmax * rgb_frame.shape[1])
                y2 = int(det.ymax * rgb_frame.shape[0])
                
                label = COCO_CLASSES[det.label] if det.label < len(COCO_CLASSES) else f"cls_{det.label}"
                color = COLORS[det.label % len(COLORS)]
                
                # Get depth
                depth_str = ""
                if depth_frame is not None:
                    depth_m = get_depth_at_bbox(depth_frame, (x1, y1, x2, y2))
                    if depth_m > 0:
                        depth_str = f" {depth_m:.2f}m"
                
                # Draw box
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                text = f"{label}: {det.confidence:.2f}{depth_str}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(rgb_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(rgb_frame, text, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 100:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Draw info
            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"Detections: {len(detections)}",
                f"Frame: {frame_count}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(rgb_frame, line, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("OAK-D YOLO Test", rgb_frame)
            
            # Show depth if available
            if depth_frame is not None:
                depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_vis)
            
            # Save video
            if video_writer:
                video_writer.write(rgb_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.png"
                cv2.imwrite(filename, rgb_frame)
                print(f"📸 Saved: {filename}")
            
            frame_count += 1
        
        # Print summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"  Total frames: {frame_count}")
        print(f"  Elapsed time: {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count / elapsed:.1f}")
        print(f"  Average detections/frame: {np.mean(detection_counts):.1f}")
        print(f"  Max detections/frame: {max(detection_counts) if detection_counts else 0}")
        print("=" * 60)
        
        # Performance assessment
        print("\n📊 Performance Assessment:")
        avg_fps = frame_count / elapsed
        if avg_fps >= 15:
            print(f"  ✅ FPS ({avg_fps:.1f}) meets target (≥15)")
        elif avg_fps >= 10:
            print(f"  ⚠️ FPS ({avg_fps:.1f}) is acceptable but below target")
        else:
            print(f"  ❌ FPS ({avg_fps:.1f}) is too low for real-time")
            print("     Try: --preview-size 416 416 --nn-size 416 416")
    
    # Cleanup
    cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()
        print(f"\n📹 Video saved: {args.save_video}")


if __name__ == '__main__':
    main()
