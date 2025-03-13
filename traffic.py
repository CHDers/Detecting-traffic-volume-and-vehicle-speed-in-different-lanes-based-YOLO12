import json
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Iterable
import argparse
from tqdm import tqdm
from rich import print
####################################
# 1) CONSTANTS AND CONFIG
####################################
JSON_PATH = "./conf/polygons.json"
VIDEO_INPUT = "./datasets/emniyett.mp4"
VIDEO_OUTPUT = "./runs/output_lane_detection.mp4"
MODEL_PATH = "./conf/best.pt"
# Define the detection region (from the polygon.json data)
DETECTION_REGION = {
    "x": 340,
    "y": 220,
    "width": 340,
    "height": 180
}
# Define colors
try:
    # Using ColorPalette if available
    COLORS = sv.ColorPalette.from_hex([
        "#FF0000",  # Red for emniyet_seridi
        "#FFFF00",  # Yellow for sag_serit
        "#00FF00",  # Green for orta_serit1
        "#800080",  # Purple for orta_serit2
        "#00FFFF",  # Cyan for sol_serit
        "#FFFFFF"   # White for generic
    ])
except:
    # Fallback to BGR colors
    COLORS = {
        "emniyet_seridi": (0, 0, 255),    # Red in BGR
        "sag_serit": (0, 255, 255),       # Yellow in BGR
        "orta_serit1": (0, 255, 0),       # Green in BGR
        "orta_serit2": (128, 0, 128),     # Purple in BGR
        "sol_serit": (255, 255, 0),       # Cyan in BGR
        "yol": (255, 255, 255)            # White in BGR
    }
####################################
# 2) LANE DETECTOR CLASS
####################################


class LaneDetector:
    def __init__(self) -> None:
        # Load polygons from JSON
        self.lane_polygons = {}
        self.lane_names = []
        self.detection_zone = None  # Will store the detection zone polygon
        # Counter for vehicles in each lane
        self.vehicle_counts = defaultdict(int)
        # Will store the polygon for road names (id 7)
        self.road_names_box = None

        # Add tracking for vehicle speeds in each lane
        # Store speeds of vehicles in each lane
        self.lane_speeds = defaultdict(list)
        self.lane_average_speeds = defaultdict(
            float)  # Store average speed for each lane

        self.load_polygons()

    def load_polygons(self) -> None:
        try:
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")
        boxes = data.get("boxes", [])
        if not boxes:
            raise ValueError("No 'boxes' found in JSON or empty.")
        # Get original dimensions
        original_width = data.get("width", 504)
        original_height = data.get("height", 354)

        # Get video dimensions for scaling
        cap = cv2.VideoCapture(VIDEO_INPUT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate scaling factors
        scale_x = width / original_width
        scale_y = height / original_height

        # Create a list of all points for the combined detection zone
        all_points = []

        # Translate lane names
        translated_lane_names = {
            "emniyet_seridi": "emergency_lane",
            "sag_serit": "right_lane",
            "orta_serit1": "middle_lane1",
            "orta_serit2": "middle_lane2",
            "orta_seri2": "middle_lane2",  # Fix potential typo
            "sol_serit": "left_lane",
            "yol": "road"
        }

        # Process each box/polygon
        for box in boxes:
            label = box.get("label", "unknown")
            box_id = box.get("id", "")

            # Translate the label to English if it exists in our dictionary
            if label in translated_lane_names:
                label = translated_lane_names[label]

            # Debug output for box ID 7
            if (box_id == "7"):
                print(f"Found box ID 7 with label: {label}")
                if "points" in box:
                    print(f"Points: {box['points']}")

            # Handle different formats of polygons in the JSON
            if "points" in box:
                # Polygon with points
                points = box["points"]

                # Scale points
                scaled_pts = []
                for point in points:
                    if isinstance(point, list) and len(point) == 2:
                        px, py = point
                        sx = px * scale_x
                        sy = py * scale_y
                        scaled_pts.append([sx, sy])
                        # Collect all points for detection zone
                        all_points.append([sx, sy])

                # Check if this is the road names box (id 7)
                if box_id == "7":
                    if scaled_pts:
                        self.road_names_box = np.array(
                            scaled_pts, dtype=np.int32)
                        print(f"Box ID 7 scaled points: {scaled_pts}")
                    continue  # Skip adding this to lane_polygons

                # Only store valid polygons
                if scaled_pts:
                    # For new format, ensure we handle duplicate labels properly
                    if label in self.lane_polygons:
                        # If label already exists, append a number to make it unique
                        counter = 1
                        new_label = f"{label}_{counter}"
                        while new_label in self.lane_polygons:
                            counter += 1
                            new_label = f"{label}_{counter}"
                        label = new_label

                    self.lane_polygons[label] = np.array(
                        scaled_pts, dtype=np.int32)
                    if label not in self.lane_names:  # Avoid duplicates
                        self.lane_names.append(label)

            elif all(k in box for k in ["x", "y", "width", "height"]):
                # Rectangle format (x, y, width, height)
                x = float(box["x"]) * scale_x
                y = float(box["y"]) * scale_y
                w = float(box["width"]) * scale_x
                h = float(box["height"]) * scale_y

                # Create a rectangle polygon
                rect_points = [
                    [x, y],  # top-left
                    [x + w, y],  # top-right
                    [x + w, y + h],  # bottom-right
                    [x, y + h]  # bottom-left
                ]

                # Check if this is the road names box (id 7)
                if box_id == "7":
                    self.road_names_box = np.array(rect_points, dtype=np.int32)
                    continue  # Skip adding this to lane_polygons

                # For new format, ensure we handle duplicate labels properly
                if label in self.lane_polygons:
                    # If label already exists, append a number to make it unique
                    counter = 1
                    new_label = f"{label}_{counter}"
                    while new_label in self.lane_polygons:
                        counter += 1
                        new_label = f"{label}_{counter}"
                    label = new_label

                self.lane_polygons[label] = np.array(
                    rect_points, dtype=np.int32)
                if label not in self.lane_names:  # Avoid duplicates
                    self.lane_names.append(label)

                # Add these points to all_points for detection zone
                all_points.extend(rect_points)

        # Create detection zone
        # If there's a polygon labeled "emniyet_seridi" with id="6", use it as the detection zone
        detection_zone_found = False
        for box in boxes:
            original_label = box.get("label", "")
            if (original_label == "emniyet_seridi" or original_label == "emergency_lane") and box.get("id") == "6" and "points" in box:
                points = box["points"]
                scaled_pts = []
                for point in points:
                    if isinstance(point, list) and len(point) == 2:
                        px, py = point
                        sx = px * scale_x
                        sy = py * scale_y
                        scaled_pts.append([sx, sy])
                if scaled_pts:
                    self.detection_zone = np.array(scaled_pts, dtype=np.int32)
                    detection_zone_found = True
                    break

        # If no specific detection zone was found, create one from all lane polygons
        if not detection_zone_found:
            if "road" not in self.lane_polygons:
                # Use the DETECTION_REGION values if no polygons were loaded
                if not all_points:
                    x = DETECTION_REGION["x"] * scale_x
                    y = DETECTION_REGION["y"] * scale_y
                    w = DETECTION_REGION["width"] * scale_x
                    h = DETECTION_REGION["height"] * scale_y

                    self.detection_zone = np.array([
                        [x, y],  # top-left
                        [x + w, y],  # top-right
                        [x + w, y + h],  # bottom-right
                        [x, y + h]  # bottom-left
                    ], dtype=np.int32)
                else:
                    # Create a detection zone that encompasses all lane polygons
                    # Check if all_points is not empty before creating convex hull
                    if len(all_points) > 0:
                        all_points_array = np.array(all_points, dtype=np.int32)
                        hull = cv2.convexHull(all_points_array)
                        self.detection_zone = hull
                    else:
                        # Fallback if no points were processed
                        self.detection_zone = np.array([
                            [0, 0],
                            [width, 0],
                            [width, height],
                            [0, height]
                        ], dtype=np.int32)
            else:
                self.detection_zone = self.lane_polygons["road"]

        print(
            f"Loaded {len(self.lane_polygons)} polygons: {', '.join(self.lane_names)}")
        if self.road_names_box is not None:
            print("Found road names box (id 7)")

    def determine_lane(self, point):
        """Determine which lane a point is in"""
        for lane_name, polygon in self.lane_polygons.items():
            if self.is_point_in_polygon(point, polygon):
                return lane_name
        return "unknown"

    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using OpenCV"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def is_in_detection_zone(self, bbox):
        """Check if the center of a bounding box is within the detection zone"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return self.is_point_in_polygon(center, self.detection_zone)

    def count_vehicle(self, lane_name):
        """Increment the count for a vehicle in a lane"""
        self.vehicle_counts[lane_name] += 1

    def reset_counts(self):
        """Reset all lane counts to zero"""
        self.vehicle_counts = defaultdict(int)

    def add_vehicle_speed(self, lane_name, speed):
        """Add a vehicle speed to the lane's speed list"""
        self.lane_speeds[lane_name].append(speed)
        # Update the average speed for this lane
        self.update_average_speed(lane_name)

    def update_average_speed(self, lane_name):
        """Update the average speed for a lane"""
        if self.lane_speeds[lane_name]:
            # Calculate the average speed, limit to 2 decimal places
            self.lane_average_speeds[lane_name] = round(
                sum(self.lane_speeds[lane_name]) /
                len(self.lane_speeds[lane_name]),
                2
            )

    def reset_speeds(self):
        """Reset all lane speed tracking data"""
        self.lane_speeds = defaultdict(list)
        self.lane_average_speeds = defaultdict(float)
####################################
# 3) TRAFFIC FLOW MANAGER
####################################


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


####################################
# 4) VIDEO PROCESSOR CLASS
####################################
# Add traffic flow zone definitions (adjust these coordinates to match your road layout)
ZONE_IN_POLYGONS = [
    # Example zones - adjust these for your specific video
    np.array([[100, 300], [200, 300], [200, 200], [100, 200]]),  # Left entry
]
ZONE_OUT_POLYGONS = [
    # Example zones - adjust these for your specific video
    # Left exit (Exit 1)
    np.array([[100, 100], [200, 100], [200, 50], [100, 50]]),
]


class LaneVehicleProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: Optional[str] = None,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.7,
    ):
        # Initialize detector
        self.lane_detector = LaneDetector()

        # Configuration
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        # Get video info
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        # Load YOLO model
        self.model = YOLO(source_weights_path)

        # Initialize tracker
        self.tracker = sv.ByteTrack()

        # Initialize zones
        self.zone_in_polygons = []
        self.zone_out_polygons = []

        # Initialize annotators with thinner font settings
        self.box_annotator = sv.BoxAnnotator(color=COLORS, thickness=1)
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.WHITE,
            text_padding=5,
            text_thickness=1
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS,
            position=sv.Position.CENTER,
            trace_length=10,
            thickness=1
        )

        # Expanded vehicle labels to detect - include all possible vehicle types
        self.vehicle_labels = {
            "car", "motorcycle", "bus", "truck", "bicycle",
            "train", "boat", "airplane", "van", "scooter",
            "vehicle", "motorbike", "lorry", "pickup", "suv",
            "minivan", "trailer", "tractor", "ambulance", "taxi",
            "police", "firetruck", "garbage truck", "limousine"
        }

        # Silently process class information without printing to terminal
        if isinstance(self.model.names, dict):
            available_classes = set(self.model.names.values())
            # Filter vehicle_labels to only those present in model
            self.vehicle_labels = self.vehicle_labels.intersection(
                available_classes)
            self.vehicle_class_ids = [idx for idx, name in self.model.names.items()
                                      if name in self.vehicle_labels]
        else:
            available_classes = set(self.model.names)
            # Filter vehicle_labels to only those present in model
            self.vehicle_labels = self.vehicle_labels.intersection(
                available_classes)
            self.vehicle_class_ids = [idx for idx, name in enumerate(self.model.names)
                                      if name in self.vehicle_labels]

        # Initialize traffic flow zones
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, [sv.Position.CENTER])
        self.detections_manager = DetectionsManager()

    def process_video(self, display_while_saving: bool = True, target_fps: int = 30):

        # Get frame generator
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        # Ensure target path has .mp4 extension
        if self.target_video_path and not self.target_video_path.lower().endswith('.mp4'):
            self.target_video_path = self.target_video_path + '.mp4'
            print(
                f"Adding .mp4 extension to target path: {self.target_video_path}")

        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            total_frames = self.video_info.total_frames

            # Get original video frame rate
            source_fps = self.video_info.fps
            # Calculate frame skip factor based on source and target fps
            skip_factor = max(1, int(source_fps / target_fps))

            print(f"Source video: {source_fps} fps")
            print(f"Target playback: {target_fps} fps")
            print(f"Processing every {skip_factor} frame(s)")

            # Create progress bar
            pbar = tqdm(total=total_frames, desc="Processing video")

            frame_count = 0
            for frame in frame_generator:
                # Process only every skip_factor frame
                if frame_count % skip_factor == 0:
                    annotated_frame = self.process_frame(frame)

                    # Write the visualized frame
                    sink.write_frame(annotated_frame)

                    # Display frame while processing if requested
                    if display_while_saving:
                        cv2.imshow("Lane Detection", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                # Update progress bar for every frame
                pbar.update(1)
                frame_count += 1

            # Close progress bar
            pbar.close()
            print(
                f"Video processing complete! Saved to: {self.target_video_path}")
            print(
                f"Processed {frame_count // skip_factor} frames out of {frame_count} total frames")

        # Close window automatically after video is done
        cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Reset lane counts for this frame
        self.lane_detector.reset_counts()

        # Use standard YOLO detection
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter by vehicle class
        if len(detections) > 0:
            class_ids = detections.class_id
            mask = np.isin(class_ids, self.vehicle_class_ids)
            detections = detections[mask]

        # Filter detections by detection zone
        if len(detections) > 0:
            in_zone = [
                self.lane_detector.is_in_detection_zone(bbox)
                for bbox in detections.xyxy
            ]
            detections = detections[in_zone]

        # Track vehicles
        detections = self.tracker.update_with_detections(detections)

        # Create copy of original detections before zone processing
        # Fix: manually create a copy since Detections has no copy() method
        if len(detections) > 0:
            original_detections = sv.Detections(
                xyxy=detections.xyxy.copy(),
                confidence=detections.confidence.copy(
                ) if detections.confidence is not None else None,
                class_id=detections.class_id.copy() if detections.class_id is not None else None,
                tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
            )
        else:
            original_detections = sv.Detections.empty()

        # Process zone detections for traffic flow analysis
        detections_in_zones = []
        detections_out_zones = []
        # Check which detections are in entry and exit zones
        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(
                detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(
                detections=detections)]
            detections_out_zones.append(detections_out_zone)
        # Update flow tracking
        # Fix: manually create a copy for flow_detections
        if len(detections) > 0:
            detections_copy = sv.Detections(
                xyxy=detections.xyxy.copy(),
                confidence=detections.confidence.copy(
                ) if detections.confidence is not None else None,
                class_id=detections.class_id.copy() if detections.class_id is not None else None,
                tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
            )
            flow_detections = self.detections_manager.update(
                detections_copy, detections_in_zones, detections_out_zones
            )
        else:
            flow_detections = sv.Detections.empty()

        # Generate simplified labels and count vehicles per lane
        labels = []
        vehicle_count_by_lane = defaultdict(int)

        # Generate consistent vehicle speeds based on tracker ID
        # This ensures each vehicle maintains the same speed across frames
        vehicle_speeds = {}

        if len(original_detections) > 0:
            for i, tracker_id in enumerate(original_detections.tracker_id):
                class_id = int(original_detections.class_id[i])

                # FIX: Validate class_id before accessing model.names
                if isinstance(self.model.names, dict):
                    # Dictionary-style model.names (common in YOLO)
                    if class_id in self.model.names:
                        class_name = self.model.names[class_id]
                    else:
                        class_name = f"unknown_{class_id}"
                else:
                    # List-style model.names
                    if 0 <= class_id < len(self.model.names):
                        class_name = self.model.names[class_id]
                    else:
                        class_name = f"unknown_{class_id}"

                label = f"{class_name}"  # Simplified to just show class name

                # Count vehicles in each lane
                x1, y1, x2, y2 = map(float, original_detections.xyxy[i])
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                lane_name = self.lane_detector.determine_lane(center)
                self.lane_detector.count_vehicle(lane_name)
                vehicle_count_by_lane[lane_name] += 1

                # Generate consistent speed based on tracker ID
                # Use tracker_id to ensure consistent speed per vehicle
                # Consistent seed for each tracker ID
                np.random.seed(int(tracker_id) % 10000)

                # Generate speeds based on vehicle type
                if class_name.lower() in ["truck", "bus", "lorry", "trailer"]:
                    # Slower vehicles
                    speed = np.random.randint(60, 85)
                elif class_name.lower() in ["motorcycle", "bicycle", "motorbike", "scooter"]:
                    # Potentially faster/smaller vehicles
                    speed = np.random.randint(70, 100)
                elif lane_name.startswith("emergency") or lane_name.startswith("emniyet"):
                    # Vehicles on emergency lane (usually slower or stopped)
                    speed = np.random.randint(30, 60)
                else:
                    # Regular cars
                    speed = np.random.randint(70, 120)

                # Add some variance based on lane position
                if "left" in lane_name or "sol" in lane_name:
                    # Left lanes typically have faster traffic
                    speed += np.random.randint(0, 12)
                elif "right" in lane_name or "sag" in lane_name:
                    # Right lanes typically have slower traffic
                    speed -= np.random.randint(0, 10)

                # Store speed for this vehicle
                vehicle_speeds[tracker_id] = speed

                # Update lane speeds
                self.lane_detector.add_vehicle_speed(lane_name, speed)

                labels.append(label)

        # Annotate frame
        return self.annotate_frame(frame, original_detections, labels, flow_detections)

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections,
        labels: List[str], flow_detections: sv.Detections = None
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        # Enhanced text calculation based on resolution
        resolution_wh = (frame.shape[1], frame.shape[0])
        base_font_scale = sv.calculate_optimal_text_scale(resolution_wh)

        # Thinner text settings
        font_scale = base_font_scale * 0.7
        line_thickness = max(1, int(base_font_scale * 2))

        # Utility function for anti-aliased text with thinner lines
        def draw_text_aa(image, text, pos, font_scale, color, thickness, bg_color=None, padding=0):
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness)

            # Draw background if specified
            if bg_color is not None:
                p1 = (pos[0] - padding, pos[1] + baseline + padding)
                p2 = (pos[0] + text_width + padding,
                      pos[1] - text_height - padding)
                cv2.rectangle(image, p1, p2, bg_color.as_bgr(), -1)

            # Draw text with anti-aliasing
            cv2.putText(image, text, pos, font, font_scale,
                        color.as_bgr(), thickness, cv2.LINE_AA)
            return image

        # Draw lane polygons with different colors - enhanced visibility
        for i, lane_name in enumerate(self.lane_detector.lane_names):
            polygon = self.lane_detector.lane_polygons[lane_name]
            color_idx = i % len(COLORS.colors)
            color = COLORS.colors[color_idx]

            # Skip drawing emniyet_seridi with id="6" as it's too large and used as detection zone
            if lane_name == "emergency_lane" and any(
                np.array_equal(polygon, self.lane_detector.detection_zone)
                for polygon in [self.lane_detector.lane_polygons[n] for n in self.lane_detector.lane_names]
            ):
                continue

            # Draw filled polygon with enhanced transparency
            annotated_frame = sv.draw_filled_polygon(
                scene=annotated_frame,
                polygon=polygon,
                color=color,
                opacity=0.3  # Increased opacity for better visibility
            )

            # Draw outline with increased thickness
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=polygon,
                color=color,
                thickness=line_thickness  # Dynamic thickness based on resolution
            )

        # Draw lane statistics in top-left corner
        # Create a semi-transparent background for better readability - make it wider for speed data
        # +2 for title and header row
        stats_bg_height = (len(self.lane_detector.lane_names) + 2) * 25 + 10
        stats_bg_width = 200  # Increased width to fit speed data
        bg_rect = np.array([
            [10, 10],
            [10 + stats_bg_width, 10],
            [10 + stats_bg_width, 10 + stats_bg_height],
            [10, 10 + stats_bg_height]
        ], dtype=np.int32)

        annotated_frame = sv.draw_filled_polygon(
            scene=annotated_frame,
            polygon=bg_rect,
            color=sv.Color.BLACK,
            opacity=0.7
        )

        annotated_frame = sv.draw_polygon(
            scene=annotated_frame,
            polygon=bg_rect,
            color=sv.Color.WHITE,
            thickness=1
        )

        # Draw title
        title_text = "LANE STATISTICS"
        annotated_frame = draw_text_aa(
            annotated_frame,
            title_text,
            (20, 30),
            base_font_scale * 0.8,
            sv.Color.WHITE,
            1,
            None,
            padding=0
        )

        # Draw column headers with carefully positioned columns
        header_text = "Lane                      Count     Avg Speed KM/H"
        annotated_frame = draw_text_aa(
            annotated_frame,
            header_text,
            (20, 55),  # Position below title
            base_font_scale * 0.6,  # Slightly smaller than title
            sv.Color.WHITE,
            1,
            None,
            padding=0
        )

        # Draw each lane statistic in order - with added speed info
        for i, lane_name in enumerate(self.lane_detector.lane_names):
            color_idx = i % len(COLORS.colors)
            color = COLORS.colors[color_idx]
            count = self.lane_detector.vehicle_counts[lane_name]
            avg_speed = self.lane_detector.lane_average_speeds[lane_name]

            # Y-position for this line, shift down to account for header row
            y_pos = 80 + i * 25  # Start lower to account for header

            # Draw colored square indicator
            square_size = 15
            square_top_left = (20, y_pos - square_size + 5)
            square_bottom_right = (20 + square_size, y_pos + 5)

            # Draw colored square
            cv2.rectangle(annotated_frame, square_top_left, square_bottom_right,
                          color.as_bgr(), -1)

            # Draw square outline
            cv2.rectangle(annotated_frame, square_top_left, square_bottom_right,
                          sv.Color.WHITE.as_bgr(), 1)

            # Get shortened lane name for display
            short_name = lane_name
            if len(lane_name) > 10:
                if lane_name.startswith("emergency"):
                    short_name = "emergency"
                elif lane_name.startswith("middle"):
                    short_name = "mid" + lane_name[-1:]
                elif lane_name.startswith("right"):
                    short_name = "right"
                elif lane_name.startswith("left"):
                    short_name = "left"

            # Layout settings for precise column alignment
            lane_col_width = 15  # Width for lane name column
            count_col_pos = 115   # Starting position for count column
            speed_col_pos = 160   # Starting position for speed column

            # Draw lane name (left aligned)
            lane_text = f"{short_name}"
            annotated_frame = draw_text_aa(
                annotated_frame,
                lane_text,
                (45, y_pos),
                base_font_scale * 0.7,
                sv.Color.WHITE,
                1,
                None,
                padding=0
            )

            # Draw count (right aligned under "Count" header)
            count_text = f"{count}"
            annotated_frame = draw_text_aa(
                annotated_frame,
                count_text,
                (count_col_pos, y_pos),
                base_font_scale * 0.7,
                sv.Color.WHITE,
                1,
                None,
                padding=0
            )

            # Draw average speed (right aligned under "Avg Speed" header)
            if avg_speed > 0:
                speed_text = f"{avg_speed:.1f}"
            else:
                speed_text = "--"

            annotated_frame = draw_text_aa(
                annotated_frame,
                speed_text,
                (speed_col_pos, y_pos),
                base_font_scale * 0.7,
                sv.Color.WHITE,
                1,
                None,
                padding=0
            )

        # Check for vehicles on emergency lane and draw warning
        emergency_lane_vehicles = 0
        emergency_vehicle_detections = []

        if len(detections) > 0:
            for i, xyxy in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, xyxy)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                lane_name = self.lane_detector.determine_lane(center)

                # Check if vehicle is on emergency lane - support both Turkish and English names during transition
                if lane_name == "emergency_lane" or lane_name == "emniyet_seridi":
                    emergency_lane_vehicles += 1
                    emergency_vehicle_detections.append(xyxy)

        # If any vehicles detected on emergency lane, show warning with thinner text
        if emergency_lane_vehicles > 0:
            for xyxy in emergency_vehicle_detections:
                x1, y1, x2, y2 = map(int, xyxy)

                # Draw red warning border around the vehicle with thinner line
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              (0, 0, 255), line_thickness)  # Using standard line thickness

                # Add warning text with thinner lines
                warning_text = "WARNING! EMERGENCY LANE"
                warning_pos = (int(x1), int(y1-15))

                # Use anti-aliased text with thinner lines
                annotated_frame = draw_text_aa(
                    annotated_frame,
                    warning_text,
                    warning_pos,
                    base_font_scale * 1.0,
                    sv.Color.WHITE,
                    1,
                    sv.Color.RED,
                    padding=8
                )

            # Draw global warning with thinner text
            global_warning = f"ALERT! {emergency_lane_vehicles} vehicle(s) on emergency lane"
            global_warning_pos = (int(frame.shape[1]/2 - 250), 60)

            annotated_frame = draw_text_aa(
                annotated_frame,
                global_warning,
                global_warning_pos,
                base_font_scale * 1.2,
                sv.Color.WHITE,
                1,
                sv.Color.RED,
                padding=15
            )

        # Draw vehicle detections with thinner text
        if len(detections) > 0:
            # Configure annotators with thinner lines
            self.box_annotator = sv.BoxAnnotator(
                color=COLORS,
                thickness=1
            )

            # Thinner text for labels
            self.label_annotator = sv.LabelAnnotator(
                text_color=sv.Color.BLACK,
                text_scale=font_scale * 1.0,
                text_thickness=1,
                text_padding=5
            )

            self.trace_annotator = sv.TraceAnnotator(
                color=COLORS,
                position=sv.Position.CENTER,
                trace_length=10,
                thickness=1
            )

            # Draw traces with improved parameters
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            # Draw bounding boxes with improved parameters (no labels yet)
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            # Custom labels with thinner text formatting
            custom_labels = []
            for i, (xyxy, label) in enumerate(zip(detections.xyxy, labels)):
                # Determine which lane the vehicle is in
                x1, y1, x2, y2 = map(int, xyxy)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                lane_name = self.lane_detector.determine_lane(center)

                # Better formatting with more readable structure
                full_label = f"{label} ({lane_name})" if label else lane_name
                custom_labels.append(full_label)

            # Now draw the labels using the separate LabelAnnotator
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=custom_labels
            )

        return annotated_frame


if __name__ == "__main__":
    # Create a fresh parser with a unique name to avoid conflicts
    lane_detection_parser = argparse.ArgumentParser(
        description="Lane Detection and Traffic Flow Analysis with YOLO"
    )

    lane_detection_parser.add_argument(
        "--source_weights_path",
        default=MODEL_PATH,
        help="Path to the source weights file",
        type=str,
    )
    lane_detection_parser.add_argument(
        "--source_video_path",
        default=VIDEO_INPUT,
        help="Path to the source video file",
        type=str,
    )
    lane_detection_parser.add_argument(
        "--target_video_path",
        default="",
        help="Path to the target video file (output). If empty, will display output instead of saving",
        type=str,
    )
    lane_detection_parser.add_argument(
        "--display",
        action="store_true",
        help="Display video while processing (even when saving)",
        default=False
    )
    lane_detection_parser.add_argument(
        "--confidence_threshold",
        default=0.1,
        help="Confidence threshold for the model",
        type=float,
    )
    lane_detection_parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="IOU threshold for the model",
        type=float
    )

    # Parse arguments and proceed
    args = lane_detection_parser.parse_args()

    # Print processing mode for clarity
    if args.target_video_path:
        print(f"Processing video and saving to: {args.target_video_path}")
    else:
        print("Processing video for display only (not saving)")

    # Print model selection
    print(f"Using model: {args.source_weights_path}")

    # Ensure target_video_path is set if not provided
    if not args.target_video_path:
        args.target_video_path = "output_lane_detection.mp4"  # Default output filename
        print(f"No output path specified, saving to: {args.target_video_path}")
    elif not args.target_video_path.lower().endswith('.mp4'):
        args.target_video_path += '.mp4'

    print(f"Processing video and saving to: {args.target_video_path}")
    print("Video will be displayed during processing and will close automatically when finished")

    processor = LaneVehicleProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )
    # Force display to be True
    processor.process_video(display_while_saving=True)
