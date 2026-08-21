#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Optimized dual-camera rover pipeline.

Keeps the same external behavior as the current ml_RecordAndDetect.py:
  - cam1 local API on port 8081 by default
  - cam2 local API on port 8082 by default
  - POST annotated JPEGs to /stream/<camera id>/frame
  - POST person alerts to /events and upload an event snapshot

Main optimizations:
  1. ONE inference thread services both cameras in round-robin order.
  2. A frame sequence number prevents inference on the same frame twice.
  3. Web video is decoupled from inference: the encoder always uses the newest
     captured frame and overlays the latest known detections.
  4. JPEGs are encoded once per camera and cached for the backend relay,
     /snapshot, and /video_feed.
  5. requests.Session() is reused for backend HTTP traffic.
  6. Camera capture defaults to 25 FPS with V4L2 + MJPG to reduce USB load.

Trade-off of decoupled video:
The browser/backend sees fresher video than the inference rate, but bounding
boxes can be a few frames old between YOLO runs. That is intentional. The
/api/detections endpoint still reports the latest inference result.

Environment variables of interest:
  API_BASE=http://92.87.91.146:5000
  ROVER_ID=ROVER-Q1
  CAM1_DEVICE=2
  CAM2_DEVICE=0
  CAM1_PORT=8081
  CAM2_PORT=8082
  FRAME_WIDTH=640
  FRAME_HEIGHT=480
  CAPTURE_FPS=25
  CAMERA_FOURCC=MJPG
  MODEL_PATH=/path/to/yolo26n_img480_int8.onnx
  INFERENCE_SIZE=480
  CONFIDENCE_THRESHOLD=0.45
  NMS_IOU_THRESHOLD=0.45
  WEB_FPS=8
  PUSH_FPS=5
  JPEG_QUALITY=70

Requires: numpy, onnxruntime, opencv-python-headless, requests
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

PERSON_CLASS_ID = 0
HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Configuration
# ============================================================
API_BASE = os.getenv("API_BASE", "http://92.87.91.146:5000")
ROVER_ID = os.getenv("ROVER_ID", "ROVER-Q1")
SESSION_ID = os.getenv(
    "SESSION_ID",
    "session-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
)

# Preserve the current script's camera identity defaults. Override these with
# CAM1_DEVICE / CAM2_DEVICE if your working mapping is the opposite way around.
CAMERAS = [
    {
        "id": "cam1",
        "device": os.getenv(
            "CAM1_DEVICE",
            "/dev/v4l/by-path/"
            "platform-xhci-hcd.2.auto-usb-0:1.2:1.0-video-index0",
        ),
        "port": int(os.getenv("CAM1_PORT", "8081")),
    },
    {
        "id": "cam2",
        "device": os.getenv(
            "CAM2_DEVICE",
            "/dev/v4l/by-path/"
            "platform-xhci-hcd.2.auto-usb-0:1.4.2:1.0-video-index0",
        ),
        "port": int(os.getenv("CAM2_PORT", "8082")),
    },
]

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))

# Both of your capture devices support 640x480 at 25 FPS. The value remains
# configurable; 15 FPS is a useful lower-load fallback.
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "25"))
CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "MJPG").upper()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(HERE, "yolo26n_img480_int8.onnx"),
)
INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE", "480"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))

JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "35"))
ALERT_JPEG_QUALITY = int(os.getenv("ALERT_JPEG_QUALITY", "55"))

# Web rendering/encoding is independent of YOLO. 8 FPS is a good initial
# compromise on the board; it can be raised without increasing YOLO load.
WEB_FPS = max(1.0, float(os.getenv("WEB_FPS", "8")))

# Backend upload rate is independently configurable. 5 FPS matches the
# previous script's 0.2 s push interval.
PUSH_FPS = max(0.2, float(os.getenv("PUSH_FPS", "5")))
PUSH_INTERVAL_SECONDS = 1.0 / PUSH_FPS

RELAY_REPORT_INTERVAL_SECONDS = float(
    os.getenv("RELAY_REPORT_INTERVAL_SECONDS", "30")
)

# Small sleep only when neither camera has a new frame waiting for inference.
INFERENCE_IDLE_SLEEP_SECONDS = float(
    os.getenv("INFERENCE_IDLE_SLEEP_SECONDS", "0.002")
)


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
    frame: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, float, int, int]:
    """Resize into a square canvas without distorting aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)

    scaled_width = int(round(width * scale))
    scaled_height = int(round(height * scale))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - scaled_width) // 2
    pad_y = (size - scaled_height) // 2

    canvas[pad_y : pad_y + scaled_height, pad_x : pad_x + scaled_width] = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    )

    return canvas, scale, pad_x, pad_y


def preprocess(canvas: np.ndarray) -> np.ndarray:
    """BGR uint8 image -> RGB NCHW float32 tensor."""
    rgb = canvas[:, :, ::-1]
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)


def nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    """Pure NumPy NMS. boxes are (x, y, w, h)."""
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
    """YOLO ONNX detector used by one global inference thread."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # There is only one inference thread, so no session lock is necessary.
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name

        # Catch 320-vs-480 mistakes immediately instead of failing on every frame.
        shape = model_input.shape
        if len(shape) >= 4:
            model_h = shape[-2]
            model_w = shape[-1]
            if isinstance(model_h, int) and isinstance(model_w, int):
                if model_h != INFERENCE_SIZE or model_w != INFERENCE_SIZE:
                    raise ValueError(
                        "INFERENCE_SIZE does not match the ONNX model: "
                        f"configured {INFERENCE_SIZE}x{INFERENCE_SIZE}, "
                        f"model expects {model_w}x{model_h}"
                    )

        log.info(
            "Loaded %s (input %dx%d, provider=CPUExecutionProvider)",
            model_path,
            INFERENCE_SIZE,
            INFERENCE_SIZE,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        canvas, scale, pad_x, pad_y = letterbox(frame, INFERENCE_SIZE)
        blob = preprocess(canvas)
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

        if predictions.ndim != 2:
            raise ValueError(f"Unexpected YOLO output shape: {raw.shape}")

        # This model outputs:
        # (300, 6)
        # [x1, y1, x2, y2, confidence, class_id]
        if predictions.shape[1] != 6:
            raise ValueError(
                f"Expected end-to-end YOLO output (N, 6), got {raw.shape}"
            )

        height, width = frame_shape[:2]
        detections: List[Detection] = []

        for row in predictions:
            x1, y1, x2, y2, confidence, class_id = row

            confidence = float(confidence)
            class_id = int(round(float(class_id)))

            # Keep PERSON only
            if class_id != PERSON_CLASS_ID:
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Coordinates refer to the 480x480 letterboxed image.
            # Convert them back to the original camera frame.
            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale

            # Clip to original frame boundaries
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(0.0, min(float(width - 1), x2))
            y2 = max(0.0, min(float(height - 1), y2))

            # Reject invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    confidence=confidence,
                )
            )

        return detections


# ============================================================
# Per-camera pipeline
# ============================================================
network_lock = threading.Lock()
class CameraPipeline:
    """Capture, cached rendering/JPEG, backend relay, and camera state."""

    def __init__(self, camera_id: str, device: int):
        self.camera_id = camera_id
        self.device = device

        self.lock = threading.Lock()
        self.running = True

        # Capture state.
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_seq = 0
        self.last_inferred_seq = -1

        # Latest inference result. These boxes are deliberately overlaid onto
        # newer video frames until the next inference result arrives.
        self.detections: List[Detection] = []
        self.person_present = False

        # Cached JPEG used by ALL consumers; no per-request re-encoding.
        self.latest_jpeg: Optional[bytes] = None
        self.latest_jpeg_seq = -1

        # Performance stats.
        self.fps_capture = 0.0
        self.fps_inference = 0.0
        self.fps_encode = 0.0
        self.fps_relay = 0.0

        # Persistent HTTP connections. Separate sessions avoid sharing one
        # Session concurrently between the high-rate relay and alert thread.
        self.relay_session = requests.Session()
        self.alert_session = requests.Session()
        self.alert_http_lock = threading.Lock()

        self.threads: List[threading.Thread] = []

    # ---------- capture ----------
    def open_camera(self) -> Optional[cv2.VideoCapture]:
        # This code runs on the Linux side of the rover, so V4L2 is explicit.
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if not capture.isOpened():
            log.error(
                "%s: cannot open camera %s",
                self.camera_id,
                self.device,
            )
            return None
                            
            

        if CAMERA_FOURCC:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*CAMERA_FOURCC),
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(
            chr((actual_fourcc_value >> (8 * i)) & 0xFF) for i in range(4)
        ).strip("\x00")

        log.info(
            "%s: opened %s at %dx%d @ %.1f fps, FOURCC=%s",
            self.camera_id,
            self.device,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS),
            actual_fourcc or "unknown",
        )
        return capture

    def capture_loop(self) -> None:
        capture: Optional[cv2.VideoCapture] = None
        frames = 0
        window_start = time.perf_counter()

        try:
            while self.running:
                if capture is None:
                    capture = self.open_camera()
                    if capture is None:
                        time.sleep(2.0)
                        continue

                ok, frame = capture.read()

                if not ok or frame is None or frame.size == 0:
                    log.warning("%s: camera read failed, reopening", self.camera_id)
                    capture.release()
                    capture = None
                    time.sleep(0.5)
                    continue

                with self.lock:
                    self.latest_frame = frame
                    self.frame_seq += 1

                frames += 1
                now = time.perf_counter()
                if now - window_start >= 1.0:
                    with self.lock:
                        self.fps_capture = frames / (now - window_start)
                    frames = 0
                    window_start = now
        finally:
            # Release from the same thread that performs cap.read(). This is
            # safer with some OpenCV/V4L2 builds during Ctrl+C shutdown.
            if capture is not None:
                capture.release()

    # ---------- interface used by the ONE inference worker ----------
    def newest_uninferred_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        """Return only a frame that has not already been inferred."""
        with self.lock:
            if self.latest_frame is None:
                return None
            if self.frame_seq == self.last_inferred_seq:
                return None
            return self.frame_seq, self.latest_frame.copy()

    def set_inference_fps(self, fps: float) -> None:
        with self.lock:
            self.fps_inference = fps

    def apply_inference_result(
        self,
        frame_seq: int,
        detections: List[Detection],
        inference_frame: np.ndarray,
    ) -> None:
        """Store detections and trigger a rising-edge alert if needed."""
        with self.lock:
            # A newer capture may already exist; that is fine. We only mark the
            # exact frame that was processed so the scheduler will next take
            # the newest available frame and skip stale intermediates.
            self.last_inferred_seq = frame_seq
            self.detections = detections

        self.handle_detection_edge(detections, inference_frame)

    # ---------- drawing + cached JPEG encoder ----------
    def draw_overlay(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps_capture: float,
        fps_inference: float,
        fps_encode: float,
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
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
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

        hud = (
            f"{self.camera_id} | capture {fps_capture:.1f} | "
            f"infer {fps_inference:.1f} | web {fps_encode:.1f} | "
            f"persons {len(detections)}"
        )
        cv2.putText(
            annotated,
            hud,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )

        return annotated

    def encoder_loop(self) -> None:
        """Encode the newest video frame once and cache the JPEG bytes."""
        interval = 1.0 / WEB_FPS
        encoded = 0
        window_start = time.perf_counter()
        next_deadline = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            next_deadline = max(next_deadline + interval, time.perf_counter())

            with self.lock:
                if self.latest_frame is None:
                    continue

                frame_seq = self.frame_seq
                frame = self.latest_frame.copy()
                detections = list(self.detections)
                fps_capture = self.fps_capture
                fps_inference = self.fps_inference
                fps_encode = self.fps_encode

            # If capture is stalled and nothing changed, reuse the existing JPEG.
            with self.lock:
                already_encoded = frame_seq == self.latest_jpeg_seq
            if already_encoded:
                continue

            annotated = self.draw_overlay(
                frame,
                detections,
                fps_capture,
                fps_inference,
                fps_encode,
            )

            ok, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not ok:
                log.warning("%s: JPEG encoding failed", self.camera_id)
                continue

            jpeg = buffer.tobytes()
            with self.lock:
                self.latest_jpeg = jpeg
                self.latest_jpeg_seq = frame_seq

            encoded += 1
            now = time.perf_counter()
            if now - window_start >= 1.0:
                with self.lock:
                    self.fps_encode = encoded / (now - window_start)
                encoded = 0
                window_start = now

    def get_cached_jpeg(self) -> Optional[bytes]:
        with self.lock:
            # bytes are immutable, so returning the reference is safe and avoids
            # another allocation/copy for every HTTP request.
            return self.latest_jpeg

    # ---------- alerts ----------
    def handle_detection_edge(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> None:
        present = len(detections) > 0

        with self.lock:
            was_present = self.person_present
            self.person_present = present

        if not present or was_present:
            return

        best = max(detections, key=lambda detection: detection.confidence)

        threading.Thread(
            target=self.report_person,
            args=(best, frame.copy()),
            name=f"{self.camera_id}-alert",
            daemon=True,
        ).start()

    def report_person(self, detection: Detection, frame: np.ndarray) -> None:
        payload = {
            "roverId": ROVER_ID,
            "sessionId": SESSION_ID,
            "alertType": "Human Detected",
            "source": "YOLOv26-Camera",
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

        # Keep event creation + its image upload on one persistent session.
        with self.alert_http_lock:
            try:
                with network_lock:
                    response = self.alert_session.post(
                        f"{API_BASE}/events",
                        json=payload,
                        timeout=(5, 10),
                    )
                    response.raise_for_status()
                    event_id = response.json().get("id")
            except (requests.RequestException, ValueError) as exc:
                log.error("%s: alert POST failed (%s)", self.camera_id, exc)
                return

            if event_id is None:
                log.error("%s: backend returned no event id", self.camera_id)
                return

            log.info("%s: person detected, event id=%s", self.camera_id, event_id)

            # Alert snapshot remains tied to the exact inference frame.
            annotated = self.draw_overlay(
                frame,
                [detection],
                fps_capture=self.fps_capture,
                fps_inference=self.fps_inference,
                fps_encode=self.fps_encode,
            )
            ok, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), ALERT_JPEG_QUALITY],
            )
            if not ok:
                log.error("%s: could not encode alert image", self.camera_id)
                return

            files = {
                "image": (
                    "snapshot.jpg",
                    io.BytesIO(buffer.tobytes()),
                    "image/jpeg",
                )
            }

            try:
                with network_lock:
                    upload = self.alert_session.post(
                        f"{API_BASE}/events/{event_id}/image",
                        files=files,
                        timeout=(5, 10),
                    )
                    upload.raise_for_status()
            except requests.RequestException as exc:
                log.error(
                    "%s: alert image upload failed (%s)",
                    self.camera_id,
                    exc,
                )

    # ---------- backend relay ----------
    def relay_loop(self) -> None:
        push_url = f"{API_BASE}/stream/{self.camera_id}/frame"
        log.info(
            "%s: relaying cached frames to %s at up to %.1f fps",
            self.camera_id,
            push_url,
            PUSH_FPS,
        )

        pushed = 0
        failed = 0
        last_report = time.monotonic()
        relay_window_start = time.perf_counter()
        relay_window_count = 0

        while self.running:
            started = time.perf_counter()
            jpeg = self.get_cached_jpeg()

            if jpeg is not None:
                try:
                    with network_lock:
                        response = self.relay_session.post(
                            push_url,
                            data=jpeg,
                            headers={"Content-Type": "image/jpeg"},
                            timeout=(5, 8),
                        )
                        response.raise_for_status()
                    pushed += 1
                    relay_window_count += 1
                except requests.RequestException as exc:
                    failed += 1
                    if failed == 1 or failed % 50 == 0:
                        log.warning(
                            "%s: frame push failed (%s)",
                            self.camera_id,
                            exc,
                        )

            now_perf = time.perf_counter()
            if now_perf - relay_window_start >= 1.0:
                with self.lock:
                    self.fps_relay = relay_window_count / (
                        now_perf - relay_window_start
                    )
                relay_window_count = 0
                relay_window_start = now_perf

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

            elapsed = time.perf_counter() - started
            sleep_for = PUSH_INTERVAL_SECONDS - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ---------- state / lifecycle ----------
    def state(self) -> dict:
        with self.lock:
            return {
                "cameraId": self.camera_id,
                "device": self.device,
                "fps_capture": self.fps_capture,
                "fps_inference": self.fps_inference,
                "fps_web_encode": self.fps_encode,
                "fps_relay": self.fps_relay,
                "person_present": self.person_present,
                "person_count": len(self.detections),
                "frame_seq": self.frame_seq,
                "last_inferred_seq": self.last_inferred_seq,
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

    def start(self) -> None:
        for target, name in (
            (self.capture_loop, "capture"),
            (self.encoder_loop, "encoder"),
            (self.relay_loop, "relay"),
        ):
            thread = threading.Thread(
                target=target,
                name=f"{self.camera_id}-{name}",
                daemon=False,
            )
            self.threads.append(thread)
            thread.start()
            log.info("Started thread: %s", thread.name)

    def stop(self) -> None:
        self.running = False

    def join(self, timeout: float = 3.0) -> None:
        for thread in self.threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.warning("Thread %s did not stop within %.1fs", thread.name, timeout)

        self.relay_session.close()
        self.alert_session.close()


# ============================================================
# ONE inference thread for ALL cameras
# ============================================================
def inference_worker(
    detector: PersonDetector,
    pipelines: List[CameraPipeline],
    stop_event: threading.Event,
) -> None:
    """Round-robin inference using only the newest frame from each camera."""
    stats: Dict[str, Dict[str, float]] = {
        pipeline.camera_id: {
            "count": 0.0,
            "start": time.perf_counter(),
        }
        for pipeline in pipelines
    }

    while not stop_event.is_set():
        processed_any = False

        for pipeline in pipelines:
            if stop_event.is_set() or not pipeline.running:
                break

            item = pipeline.newest_uninferred_frame()
            if item is None:
                continue

            frame_seq, frame = item
            processed_any = True

            try:
                detections = detector.detect(frame)
            except Exception as exc:
                log.exception(
                    "%s: inference failed (%s)",
                    pipeline.camera_id,
                    exc,
                )
                # Mark this frame handled so a permanently bad frame is not
                # retried forever. A newer capture will be picked next.
                with pipeline.lock:
                    pipeline.last_inferred_seq = frame_seq
                continue

            pipeline.apply_inference_result(frame_seq, detections, frame)

            entry = stats[pipeline.camera_id]
            entry["count"] += 1.0
            now = time.perf_counter()
            elapsed = now - entry["start"]
            if elapsed >= 1.0:
                pipeline.set_inference_fps(entry["count"] / elapsed)
                entry["count"] = 0.0
                entry["start"] = now

        if not processed_any:
            time.sleep(INFERENCE_IDLE_SLEEP_SECONDS)


# ============================================================
# Local HTTP API - one server per camera, same layout as current code
# ============================================================
def make_handler(pipeline: CameraPipeline):
    class ApiHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
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
                jpeg = pipeline.get_cached_jpeg()
                if jpeg is None:
                    self.send_json(503, {"error": "No frame captured yet"})
                    return
                self.send_payload(200, "image/jpeg", jpeg)
                return

            if path == "/video_feed":
                self.stream_mjpeg()
                return

            self.send_json(404, {"error": f"No route for {self.path}"})

        def send_json(self, status: int, body: dict) -> None:
            self.send_payload(
                status,
                "application/json",
                json.dumps(body).encode("utf-8"),
            )

        def send_payload(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def stream_mjpeg(self) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            interval = 1.0 / WEB_FPS
            try:
                while pipeline.running:
                    jpeg = pipeline.get_cached_jpeg()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: "
                        + str(len(jpeg)).encode("ascii")
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ApiHandler


# ============================================================
# Main
# ============================================================
def main() -> None:
    if len({camera["port"] for camera in CAMERAS}) != len(CAMERAS):
        raise ValueError("Each camera must use a different local HTTP port")

    if len({camera["device"] for camera in CAMERAS}) != len(CAMERAS):
        raise ValueError("Each camera must use a different /dev/video device")

    log.info(
        "Config: capture=%dx%d@%d fps %s, inference=%dx%d, web=%.1f fps, push=%.1f fps",
        FRAME_WIDTH,
        FRAME_HEIGHT,
        CAPTURE_FPS,
        CAMERA_FOURCC,
        INFERENCE_SIZE,
        INFERENCE_SIZE,
        WEB_FPS,
        PUSH_FPS,
    )

    detector = PersonDetector(MODEL_PATH)
    stop_event = threading.Event()

    pipelines = [
        CameraPipeline(camera["id"], camera["device"])
        for camera in CAMERAS
    ]

    # Bind both HTTP ports before starting camera threads. If a port is already
    # occupied, the program fails cleanly without leaving cameras open.
    servers: List[ThreadingHTTPServer] = []
    server_threads: List[threading.Thread] = []

    try:
        for camera, pipeline in zip(CAMERAS, pipelines):
            handler_cls = make_handler(pipeline)
            server = ThreadingHTTPServer(("0.0.0.0", camera["port"]), handler_cls)
            server.daemon_threads = True
            servers.append(server)

        for pipeline in pipelines:
            pipeline.start()

        inference_thread = threading.Thread(
            target=inference_worker,
            args=(detector, pipelines, stop_event),
            name="inference-all-cameras",
            daemon=False,
        )
        inference_thread.start()
        log.info("Started thread: %s", inference_thread.name)

        for camera, server in zip(CAMERAS, servers):
            thread = threading.Thread(
                target=server.serve_forever,
                name=f"{camera['id']}-http",
                daemon=False,
            )
            server_threads.append(thread)
            thread.start()
            log.info(
                "%s: local API on http://0.0.0.0:%d; backend=%s",
                camera["id"],
                camera["port"],
                API_BASE,
            )

        log.info("Pipeline running. Press Ctrl+C to stop.")

        while not stop_event.wait(1.0):
            pass

    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        stop_event.set()

        for pipeline in pipelines:
            pipeline.stop()

        # shutdown() must only be called after serve_forever() has started;
        # calling it on a merely-bound server can deadlock.
        for index, server in enumerate(servers):
            if index < len(server_threads):
                try:
                    server.shutdown()
                except Exception:
                    pass
            try:
                server.server_close()
            except Exception:
                pass

        # inference_thread is only defined if startup got that far.
        if "inference_thread" in locals():
            inference_thread.join(timeout=5.0)
            if inference_thread.is_alive():
                log.warning("Inference thread did not stop within 5 seconds")

        for pipeline in pipelines:
            pipeline.join(timeout=3.0)

        for thread in server_threads:
            thread.join(timeout=2.0)

        log.info("Stopped.")


if __name__ == "__main__":
    main()
