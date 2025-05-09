from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Tuple
import cv2
from collections import defaultdict, deque

class SpeedEstimator:
    def __init__(
        self,
        model_path: str,
        max_age: int = 120,
        n_init: int = 3,
        conf_threshold: float = 0.2,
        max_inactive_frames: int = 60,
        fps: int = 30
    ):
        self.model = YOLO(model_path)
        self.deep_sort = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=0.5,
            nn_budget=100,
            nms_max_overlap=0.5,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        self.conf_threshold = conf_threshold
        self.max_inactive_frames = max_inactive_frames
        self.max_inactive_frames_for_speed_estimation = 10
        self.pixel_to_meter_ratio = 0.05
        self.fps = fps

        self.class_map = {
            2: "car",
            7: "truck",
            5: "bus",
            3: "motorbike",
            1: "bicycle"
        }

        self.coordinates = defaultdict(lambda: deque(maxlen=self.fps))
        self.track_info = {}  # track_id -> (class_name, conf_value)

    def calculate_speed(self, first_bbox, last_bbox, time_diff):
        x1, y1, x2, y2 = map(int, first_bbox)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        x1, y1, x2, y2 = map(int, last_bbox)
        center_x_last = (x1 + x2) / 2
        center_y_last = (y1 + y2) / 2

        distance_x = (center_x_last - center_x) * self.pixel_to_meter_ratio
        distance_y = (center_y_last - center_y) * self.pixel_to_meter_ratio

        speed_x = distance_x / time_diff
        speed_y = distance_y / time_diff

        speed = (speed_x ** 2 + speed_y ** 2) ** 0.5 * 3.6  # m/s to km/h
        return round(speed)

    def _get_detections(self, results) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        detections = []

        for frame_result in results:
            for obj in frame_result.boxes:
                obj_conf = obj.conf.tolist()
                xyxy = obj.xyxy.tolist()
                obj_cls = obj.cls.tolist()
                for i in range(len(xyxy)):
                    if obj_conf[i] < self.conf_threshold:
                        continue
                    cls_id = int(obj_cls[i])
                    if cls_id not in self.class_map:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    w, h = x2 - x1, y2 - y1
                    detections.append(((x1, y1, w, h), obj_conf[i], cls_id))
        return detections

    def _process_tracks(self, tracks) -> List[Tuple[int, Tuple[int, int, int, int], str, float, float]]:
        processed_tracks = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > self.max_inactive_frames_for_speed_estimation:
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            self.coordinates[track_id].append(bbox)

            if track_id in self.track_info:
                class_name, conf = self.track_info[track_id]
            else:
                continue  # No class info yet

            if len(self.coordinates[track_id]) < self.fps / 2:
                continue

            first_bbox = self.coordinates[track_id][0]
            last_bbox = self.coordinates[track_id][-1]
            time_diff = len(self.coordinates[track_id]) / self.fps
            speed = self.calculate_speed(first_bbox, last_bbox, time_diff)

            processed_tracks.append((track_id, last_bbox, class_name, speed, conf))

        return processed_tracks

    def _draw_annotations(self, frame, processed_tracks):
        for track_id, bbox, class_name, speed, conf_value in processed_tracks:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{track_id} {class_name} {speed}km/h {conf_value:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return frame

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(source=frame, conf=self.conf_threshold, imgsz=640, iou=0.5)

        detections = self._get_detections(results)

        ds_detections = []
        for (tlwh, conf, cls_id) in detections:
            ds_detections.append((tlwh, conf, str(cls_id)))

        tracks = self.deep_sort.update_tracks(ds_detections, frame=frame)

        for track, (tlwh, conf, cls_id) in zip(tracks, detections):
            if track.is_confirmed():
                self.track_info[track.track_id] = (self.class_map[cls_id], conf)

        processed_tracks = self._process_tracks(tracks)
        frame = self._draw_annotations(frame, processed_tracks)
        return frame

def main():
    video_path = "/Users/egeardaozturk/speed_estimation/Vehicle Speed Estimation Video.avi"
    output_path = "output.mp4"

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    estimator = SpeedEstimator(model_path="yolo11n.pt", fps=fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = estimator._process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
