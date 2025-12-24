#!/usr/bin/env python3
"""
Detection Interface - Base classes for object detection

Provides unified interface for different detection backends:
- SimDetector: YOLO on CPU/GPU for Gazebo simulation
- OakDDetector: YOLO on OAK-D VPU for real TurtleBot4

All detectors publish to /object_detections with the same format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class BoundingBox:
    """2D bounding box in image coordinates"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, 
                (self.y_min + self.y_max) // 2)
    
    @property
    def width(self) -> int:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class Detection:
    """Single object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    position_3d: Optional[Tuple[float, float, float]] = None  # x, y, z in camera frame
    
    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': {
                'x_min': self.bbox.x_min,
                'y_min': self.bbox.y_min,
                'x_max': self.bbox.x_max,
                'y_max': self.bbox.y_max,
            },
            'position_3d': self.position_3d,
        }


@dataclass
class DetectionResult:
    """Complete detection result for a single frame"""
    timestamp: float
    frame_id: str
    image_width: int
    image_height: int
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    
    def filter_by_confidence(self, min_conf: float) -> 'DetectionResult':
        filtered = [d for d in self.detections if d.confidence >= min_conf]
        return DetectionResult(
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            image_width=self.image_width,
            image_height=self.image_height,
            detections=filtered,
            inference_time_ms=self.inference_time_ms
        )
    
    def filter_by_class(self, class_names: List[str]) -> 'DetectionResult':
        filtered = [d for d in self.detections if d.class_name in class_names]
        return DetectionResult(
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            image_width=self.image_width,
            image_height=self.image_height,
            detections=filtered,
            inference_time_ms=self.inference_time_ms
        )


class BaseDetector(ABC):
    """Abstract base class for object detection"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.class_names: List[str] = []
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector"""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray, depth: Optional[np.ndarray] = None) -> DetectionResult:
        """Run detection on an image"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup resources"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def get_3d_position(self, bbox: BoundingBox, depth_image: np.ndarray,
                        camera_intrinsics: dict) -> Optional[Tuple[float, float, float]]:
        """Estimate 3D position from bounding box and depth"""
        cx_box, cy_box = bbox.center
        
        # Sample depth in center region
        margin = min(bbox.width, bbox.height) // 4
        x1 = max(0, cx_box - margin)
        x2 = min(depth_image.shape[1], cx_box + margin)
        y1 = max(0, cy_box - margin)
        y2 = min(depth_image.shape[0], cy_box + margin)
        
        depth_region = depth_image[y1:y2, x1:x2]
        valid_depths = depth_region[(depth_region > 0.1) & (depth_region < 10.0)]
        
        if len(valid_depths) == 0:
            return None
        
        z = float(np.median(valid_depths))
        
        fx = camera_intrinsics.get('fx', 525.0)
        fy = camera_intrinsics.get('fy', 525.0)
        cx = camera_intrinsics.get('cx', depth_image.shape[1] / 2)
        cy = camera_intrinsics.get('cy', depth_image.shape[0] / 2)
        
        x = (cx_box - cx) * z / fx
        y = (cy_box - cy) * z / fy
        
        return (x, y, z)


# COCO class names (YOLOv8 default - 80 classes)
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
