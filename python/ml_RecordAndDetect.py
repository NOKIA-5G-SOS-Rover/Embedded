#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Camera pipeline for the rover: capture, person detection, live video relay to
the website, and person-detected alarms.

main.py owns the motors and the dashboard connection; this file owns the
cameras. main.py starts it as a child process, so a single
`arduino-app-cli app start ~/ArduinoApps/castel` brings up both halves.

Each camera gets its OWN local HTTP server on its own port (cam1 -> 8081,
cam2 -> 8082 by default), rather than sharing one port with the camera id in
the URL path. main.py's snapshot_url()/detections_url() must agree with the
ports set here - see CAM1_PORT/CAM2_PORT below.

Per camera it runs three threads:
  capture   - pulls frames off the V4L2 device
  inference - person detection, and the rising-edge alarm to POST /events
  relay     - POSTs annotated JPEGs to the backend, which re-serves them to the
              site as MJPEG at /stream/<camera id>

Each camera's local server exposes:
  GET /                 - camera info + endpoint list
  GET /api/detections   - latest detections as JSON
  GET /snapshot         - latest annotated frame as JPEG
  GET /video_feed       - MJPEG stream

Detection runs YOLOv8n through onnxruntime rather than ultralytics, because
ultralytics pulls in PyTorch and the board has nowhere near the memory for it.
OpenCV on the board is used only for camera capture; its DNN module is not
available in the stripped-down build shipped with the Arduino container.

The model has to be an ONNX export. Produce it once on a machine that can spare
the disk:

    pip install ultralytics
    yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12

then copy yolov8n.onnx next to this file. Keep the export's imgsz and
INFERENCE_SIZE below in agreement.

Requires (see requirements.txt): numpy, onnxruntime, opencv-python-headless, requests
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import onnxruntime as ort
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ml-record-detect")

# COCO class 0. The decoder reads only this column out of the model output.
PERSON_CLASS_ID = 0

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Config - every value overridable from the environment so main.py can pass
# things down without this file needing to be edited.
# ============================================================
API_BASE = os.getenv("API_BASE", "http://92.87.91.146:5000")
ROVER_ID = os.getenv("ROVER_ID", "ROVER-Q1")
SESSION_ID = os.getenv(
    "SESSION_ID",
    "session-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
)

# "device" is the V4L2 index behind /dev/videoN - check with `ls /dev/video*`.
# "port" is this camera's OWN local HTTP server port (not shared).
CAMERAS = [
    {
        "id": "cam1",
        "device": int(os.getenv("CAM1_DEVICE", "2")),
        "port": int(os.getenv("CAM1_PORT", "8081")),
    },
    {
        "id": "cam2",
        "device": int(os.getenv("CAM2_DEVICE", "0")),
        "port": int(os.getenv("CAM2_PORT", "8082")),
    },
]

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "30"))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(HERE, "yolo26n_img480_int8.onnx"))
INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE", "480"))  # must match the export
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))

JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))

# ~5fps per camera. Modest on purpose: this goes ou t over the 5G uplink.
PUSH_INTERVAL_SECONDS = float(os.getenv("PUSH_INTERVAL_SECONDS", "0.2"))

RELAY_REPORT_INTERVAL_SECONDS = 30.0


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


# ============================================================
# Detection
# ============================================================
def letterbox(
    frame: np.ndarray, size: int
) -> Tuple[np.ndarray, float, int, int]:
    """Resize into a square canvas without distorting aspect ratio, which
    plain blobFromImage resizing would do to a 4:3 camera frame."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)

    scaled_width = int(round(width * scale))
    scaled_height = int(round(height * scale))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - scaled_width) // 2
    pad_y = (size - scaled_height) // 2

    canvas[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width] = cv2.resize(
        frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
    )

    return canvas, scale, pad_x, pad_y


def preprocess(canvas: np.ndarray) -> np.ndarray:
    """BGR uint8 frame -> NCHW float32 tensor for the ONNX model."""
    rgb = canvas[:, :, ::-1]
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Pure-numpy NMS. boxes are (x, y, w, h)."""
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        width = np.maximum(0.0, xx2 - xx1)
        height = np.maximum(0.0, yy2 - yy1)
        intersection = width * height
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        order = rest[iou <= iou_threshold]

    return keep


class PersonDetector:
    """YOLOv8n via onnxruntime.

    One session shared by every camera behind a lock. The board has neither the
    memory for a second copy nor the CPU to run two inferences at once, so
    serializing here costs nothing that contention wouldn't cost anyway.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Export it with "
                f"`yolo export model=yolov8n.pt format=onnx imgsz={INFERENCE_SIZE} "
                "opset=12` on a machine that can install ultralytics, then copy "
                "yolov8n.onnx onto the board next to this script."
            )

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.lock = threading.Lock()

        log.info("Loaded %s (input %dx%d)", model_path, INFERENCE_SIZE, INFERENCE_SIZE)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        canvas, scale, pad_x, pad_y = letterbox(frame, INFERENCE_SIZE)
        blob = preprocess(canvas)

        with self.lock:
            raw = self.session.run(None, {self.input_name: blob})[0]

        return self.decode(raw, frame.shape, scale, pad_x, pad_y)

    @staticmethod
    def decode(
        raw: np.ndarray,
        frame_shape: Tuple[int, ...],
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[Detection]:
        predictions = np.squeeze(raw)

        # A v8 export is (84, anchors); transpose to one row per anchor.
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        # v8 has no objectness column: 0-3 are the box, 4 onward are class
        # scores, so the person score sits at 4 + its class id.
        scores = predictions[:, 4 + PERSON_CLASS_ID]
        keep = scores >= CONFIDENCE_THRESHOLD

        if not np.any(keep):
            return []

        boxes = predictions[keep, :4]
        scores = scores[keep]

        # Undo the letterbox: strip the padding, then the scale.
        centre_x, centre_y = boxes[:, 0], boxes[:, 1]
        box_width, box_height = boxes[:, 2], boxes[:, 3]

        candidates = np.stack(
            [
                (centre_x - box_width / 2 - pad_x) / scale,
                (centre_y - box_height / 2 - pad_y) / scale,
                box_width / scale,
                box_height / scale,
            ],
            axis=1,
        )

        indices = nms_indices(candidates, scores, NMS_IOU_THRESHOLD)

        if not indices:
            return []

        height, width = frame_shape[:2]
        detections: List[Detection] = []

        for index in indices:
            x, y, box_w, box_h = candidates[index]

            detections.append(
                Detection(
                    x1=int(max(0, x)),
                    y1=int(max(0, y)),
                    x2=int(min(width, x + box_w)),
                    y2=int(min(height, y + box_h)),
                    confidence=float(scores[index]),
                )
            )

        return detections


class CameraPipeline:
    """Everything that happens to one camera's frames."""

    def __init__(self, camera_id: str, device: int):
        self.camera_id = camera_id
        self.device = device

        self.lock = threading.Lock()
        self.running = True

        self.latest_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.detections: List[Detection] = []
        self.person_present = False

        self.fps_capture = 0.0
        self.fps_inference = 0.0

    # ---------- capture ----------
    def open_camera(self) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(self.device)

        if not capture.isOpened():
            log.error("%s: cannot open /dev/video%d", self.camera_id, self.device)
            return None

        ###
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

        log.info(
            "%s: opened /dev/video%d at %dx%d",
            self.camera_id,
            self.device,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return capture

    def capture_loop(self) -> None:
        capture = None
        frames = 0
        window_start = time.perf_counter()

        while self.running:
            if capture is None:
                capture = self.open_camera()
                if capture is None:
                    time.sleep(2.0)
                    continue

            ok, frame = capture.read()

            if not ok:
                log.warning("%s: camera read failed, reopening", self.camera_id)
                capture.release()
                capture = None
                time.sleep(0.5)
                continue

            with self.lock:
                self.latest_frame = frame

            frames += 1
            now = time.perf_counter()

            if now - window_start >= 1.0:
                with self.lock:
                    self.fps_capture = frames / (now - window_start)
                frames = 0
                window_start = now

    # ---------- inference ----------
    def inference_loop(self, detector: PersonDetector) -> None:
        frames = 0
        window_start = time.perf_counter()

        while self.running:
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                detections = detector.detect(frame)
            except Exception as e:
                log.exception("%s: inference failed (%s)", self.camera_id, e)
                time.sleep(1.0)
                continue

            annotated = self.draw_overlay(frame, detections)

            with self.lock:
                self.detections = detections
                self.annotated_frame = annotated

            self.handle_detection_edge(detections, frame)

            frames += 1
            now = time.perf_counter()

            if now - window_start >= 1.0:
                with self.lock:
                    self.fps_inference = frames / (now - window_start)
                frames = 0
                window_start = now

    def draw_overlay(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        annotated = frame.copy()

        for detection in detections:
            cv2.rectangle(
                annotated,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 255, 0),
                2,
            )

            label = f"person {detection.confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            baseline = max(text_height + 8, detection.y1)

            cv2.rectangle(
                annotated,
                (detection.x1, baseline - text_height - 8),
                (detection.x1 + text_width + 4, baseline),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (detection.x1 + 2, baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

        with self.lock:
            hud = (
                f"{self.camera_id} | capture {self.fps_capture:.1f} fps | "
                f"infer {self.fps_inference:.1f} fps | persons {len(detections)}"
            )

        cv2.putText(
            annotated, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        return annotated

    # ---------- alarm ----------
    def handle_detection_edge(
        self, detections: List[Detection], frame: np.ndarray
    ) -> None:
        """Fires one alert when a person first appears, not once per frame."""
        present = len(detections) > 0

        with self.lock:
            was_present = self.person_present
            self.person_present = present

        if not present or was_present:
            return

        best = max(detections, key=lambda detection: detection.confidence)

        # Off the inference thread: a slow upload must not stall detection.
        threading.Thread(
            target=self.report_person,
            args=(best, frame.copy()),
            daemon=True,
        ).start()

    def report_person(self, detection: Detection, frame: np.ndarray) -> None:
        payload = {
            "roverId": ROVER_ID,
            "sessionId": SESSION_ID,
            "alertType": "Human Detected",
            "source": "YOLOv8-Camera",
            "detectedAt": datetime.now(timezone.utc).isoformat(),
            "locationX": float(detection.x1),
            "locationY": float(detection.y1),
            "boundingBoxWidth": max(float(detection.x2 - detection.x1), 0.1),
            "boundingBoxHeight": max(float(detection.y2 - detection.y1), 1.0),
            "confidenceScore": float(detection.confidence),
            "motorHaltRequested": True,
            "injuryClass": "Unknown",
            "cameraId": self.camera_id,
            "status": "NEW",
        }

        try:
            response = requests.post(f"{API_BASE}/events", json=payload, timeout=5)
            response.raise_for_status()
            event_id = response.json().get("id")
        except requests.RequestException as e:
            log.error("%s: alert POST failed (%s)", self.camera_id, e)
            return

        if event_id is None:
            log.error("%s: backend returned no event id", self.camera_id)
            return

        log.info("%s: person detected, event id=%s", self.camera_id, event_id)

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        if not ok:
            log.error("%s: could not encode alert image", self.camera_id)
            return

        try:
            files = {
                "image": ("snapshot.jpg", io.BytesIO(buffer.tobytes()), "image/jpeg")
            }
            upload = requests.post(
                f"{API_BASE}/events/{event_id}/image", files=files, timeout=10
            )
            upload.raise_for_status()
        except requests.RequestException as e:
            log.error("%s: alert image upload failed (%s)", self.camera_id, e)

    # ---------- relay to the website ----------
    def encode_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            frame = (
                self.annotated_frame
                if self.annotated_frame is not None
                else self.latest_frame
            )
            if frame is None:
                return None
            frame = frame.copy()

        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        return buffer.tobytes() if ok else None

    def relay_loop(self) -> None:
        push_url = f"{API_BASE}/stream/{self.camera_id}/frame"
        log.info("%s: relaying frames to %s", self.camera_id, push_url)

        pushed = 0
        failed = 0
        last_report = time.monotonic()

        while self.running:
            time.sleep(PUSH_INTERVAL_SECONDS)

            jpeg = self.encode_latest_jpeg()

            if jpeg is None:
                continue

            try:
                response = requests.post(
                    push_url,
                    data=jpeg,
                    headers={"Content-Type": "image/jpeg"},
                    timeout=3,
                )
                response.raise_for_status()
                pushed += 1
            except requests.RequestException as e:
                failed += 1
                # Loud on the first failure, then rate limited - this runs 5x a
                # second and would otherwise bury every other log line.
                if failed == 1 or failed % 50 == 0:
                    log.warning("%s: frame push failed (%s)", self.camera_id, e)

            now = time.monotonic()

            if now - last_report >= RELAY_REPORT_INTERVAL_SECONDS:
                log.info(
                    "%s: %d frames pushed, %d failed in the last %.0fs",
                    self.camera_id,
                    pushed,
                    failed,
                    now - last_report,
                )
                pushed = 0
                failed = 0
                last_report = now

    # ---------- state for the local API ----------
    def state(self) -> dict:
        with self.lock:
            return {
                "cameraId": self.camera_id,
                "device": self.device,
                "fps_capture": self.fps_capture,
                "fps_inference": self.fps_inference,
                "person_present": self.person_present,
                "person_count": len(self.detections),
                "persons": [
                    {
                        "x1": detection.x1,
                        "y1": detection.y1,
                        "x2": detection.x2,
                        "y2": detection.y2,
                        "confidence": detection.confidence,
                    }
                    for detection in self.detections
                ],
            }

    def start(self, detector: PersonDetector) -> None:
        for target, name in (
            (self.capture_loop, "capture"),
            (self.relay_loop, "relay"),
        ):
            threading.Thread(
                target=target, name=f"{self.camera_id}-{name}", daemon=True
            ).start()

        threading.Thread(
            target=self.inference_loop,
            args=(detector,),
            name=f"{self.camera_id}-inference",
            daemon=True,
        ).start()


# ============================================================
# Local HTTP API - one server PER CAMERA, each on its own port.
# main.py's autonomy loop polls http://localhost:<camera port>/api/detections
# for whichever camera it cares about. Built on http.server so the board
# doesn't need Flask installed.
# ============================================================
def make_handler(pipeline: CameraPipeline):
    """Returns a BaseHTTPRequestHandler subclass bound to one camera's
    pipeline. http.server wants a class, not an instance, per server - this
    closes over `pipeline` so each camera's server only ever serves itself,
    no camera id needed in the URL."""

    class ApiHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            # main.py polls detections continuously; the default per-request
            # line would bury every other log entry.
            pass

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path

            if path in ("", "/"):
                self.send_json(
                    200,
                    {
                        "cameraId": pipeline.camera_id,
                        "device": pipeline.device,
                        "backend": API_BASE,
                        "endpoints": [
                            "/api/detections",
                            "/snapshot",
                            "/video_feed",
                        ],
                    },
                )
                return

            if path == "/api/detections":
                self.send_json(200, pipeline.state())
                return

            if path == "/snapshot":
                jpeg = pipeline.encode_latest_jpeg()
                if jpeg is None:
                    self.send_json(503, {"error": "No frame captured yet"})
                    return
                self.send_payload(200, "image/jpeg", jpeg)
                return

            if path == "/video_feed":
                self.stream_mjpeg(pipeline)
                return

            self.send_json(404, {"error": f"No route for {self.path}"})

        def send_json(self, status: int, body: dict) -> None:
            self.send_payload(
                status, "application/json", json.dumps(body).encode("utf-8")
            )

        def send_payload(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def stream_mjpeg(self, pipeline: CameraPipeline) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            # An endless body has no Content-Length, so the connection cannot
            # be reused afterwards.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            try:
                while pipeline.running:
                    jpeg = pipeline.encode_latest_jpeg()

                    if jpeg is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                    time.sleep(1.0 / 15.0)
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab

    return ApiHandler


def main() -> None:
    detector = PersonDetector(MODEL_PATH)

    servers: List[ThreadingHTTPServer] = []

    for camera in CAMERAS:
        pipeline = CameraPipeline(camera["id"], camera["device"])
        pipeline.start(detector)

        handler_cls = make_handler(pipeline)
        server = ThreadingHTTPServer(("0.0.0.0", camera["port"]), handler_cls)
        server.daemon_threads = True
        servers.append(server)

        log.info(
            "%s: local API on http://0.0.0.0:%d, pushing frames to %s",
            camera["id"],
            camera["port"],
            API_BASE,
        )
        threading.Thread(
            target=server.serve_forever,
            name=f"{camera['id']}-http",
            daemon=True,
        ).start()

    # Both servers run on background threads; keep the process alive.
    threading.Event().wait()


if __name__ == "__main__":
    main()