# SPDX-License-Identifier: MIT
"""
Robot app - Uno Q Linux side. Rover control only: dashboard commands, motor
drive, the autonomy state machine and battery telemetry.

Cameras, YOLO, the live video relay and the person-detected alarms live in
ml_RecordAndDetect.py, which this file starts as a child process. Anything
touching a camera belongs there; anything touching a motor belongs here, since
Bridge is only available inside the Arduino app process.

Each camera runs its own local HTTP server (see ML_API_BASES below) rather
than sharing one port - cam1 on 8081, cam2 on 8082 by default. Keep these
ports in agreement with CAM1_PORT/CAM2_PORT in ml_RecordAndDetect.py.

Requires (see requirements.txt : signalrcore, requests
"""

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import requests
from signalrcore.hub_connection_builder import HubConnectionBuilder

from arduino.app_utils import App, Bridge

# ============================================================
# Config
# ============================================================
API_BASE = os.getenv("API_BASE", "http://92.87.91.146:5000")
HUB_URL = f"{API_BASE}/dashboardHub"
COMMAND_EVENT = "ReceiveCommand"
TELEMETRY_URL = f"{API_BASE}/telemetry"        # TODO: confirm once backend dev deploys the endpoint

# CONFIRMED via browser DevTools Network tab: the live frontend sends
# roverId: "ROVER-Q1". Must match exactly for RegisterRobot's group to line
# up with SendCommandToRobot's.
ROBOT_GROUP_ID = "ROVER-Q1"
ROVER_ID = "ROVER-Q1"  # aligned across the whole system - also set ROVER_ID=ROVER-Q1
                       # as an env var when running comms_manager.py (defaults to "ROVER-01")
SESSION_ID = "session-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

DEFAULT_SPEED = 150  # PWM scale (0-255), used only if the dashboard omits Speed

# ---------- Battery voltage divider conversion ----------
BATTERY_VREF = 3.3        # TODO: confirm - the ADC reference voltage the sketch used when scaling to 0-255
BATTERY_R1_OHMS = 10000   # TODO: your actual top resistor (battery+ side)
BATTERY_R2_OHMS = 10000   # TODO: your actual bottom resistor (GND side)
BATTERY_V_EMPTY = 3.0     # TODO: your battery's empty voltage (single Li-ion/LiPo cell ~3.0V)
BATTERY_V_FULL = 4.2      # TODO: your battery's full voltage (single Li-ion/LiPo cell ~4.2V)
# Multi-cell pack: multiply BOTH of the above by cell count (e.g. 2S: 6.0 / 8.4)
ENABLE_TELEMETRY = False  # /telemetry isn't deployed on the backend yet (404s) - re-enable once it is
TELEMETRY_INTERVAL_SECONDS = 5

# ---------- Command handling / watchdog ----------
MOVING_COMMANDS = {"forward", "backward", "turn-left", "turn-right", "arc-left", "arc-right"}
# Backstop for a truly hung/dropped connection, not a tight liveness check -
# see earlier discussion re: whether the frontend resends while held. NOTE:
# the frontend only sends a drive command once on button-down and "stop" on
# button-up (confirmed, no repeat-while-held), so this watchdog CANNOT be
# tightened to catch a dropped connection quickly - a held button and a dead
# socket look identical from here. The real disconnect handling now lives in
# on_hub_error()/on_hub_closed() below.
WATCHDOG_TIMEOUT_SECONDS = 10.0

# ---------- Autonomy ----------
# SAFETY: no distance/obstacle sensor - only a camera. See detailed note
# further down near autonomy_loop().
DEFAULT_CRUISE_SPEED = 130
SCAN_ROTATE_SPEED = 90
SCAN_ROTATE_BURST_SECONDS = 0.6
SCAN_IDLE_SECONDS = 4.0

APPROACH_SPEED_CAP = 110
APPROACH_PULSE_SECONDS = 0.4
TURN_CORRECTION_SECONDS = 0.25
CENTER_DEADZONE_FRACTION = 0.15
CLOSE_ENOUGH_AREA_FRACTION = 0.45
PERSON_CONFIDENCE_THRESHOLD = 0.6
LOST_TARGET_TIMEOUT_SECONDS = 3.0
HOLD_COOLDOWN_SECONDS = 120.0

# ---------- Camera pipeline (ml_RecordAndDetect.py) ----------
# That file owns the cameras, YOLO, the frame relay to the website and the
# person-detected alarms. This file only reads its detections to steer the
# rover. It runs as a child of this process (see ml_supervisor_loop), so it
# shares this container's network namespace and localhost is correct.
ML_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ml.py"
)

# Each camera now runs its OWN local server (ml_RecordAndDetect.py's
# CAM1_PORT/CAM2_PORT) instead of sharing one port with the camera id in the
# path. Keep these two dicts in sync with that file's defaults.
CAM_PORTS = {
    "cam1": int(os.getenv("CAM1_PORT", "8081")),
    "cam2": int(os.getenv("CAM2_PORT", "8082")),
}
# V4L2 device indices - confirmed working values for this board's camera
# enumeration (NOT 0/1 - this board registers extra /dev/videoN nodes per
# camera, e.g. metadata streams, so the lowest indices aren't the capture
# devices). Set here as the single source of truth instead of letting
# ml_RecordAndDetect.py's own defaults drift out of sync with this file.
CAM_DEVICES = {
    "cam1": os.getenv(
        "CAM1_DEVICE",
        "/dev/v4l/by-path/"
        "platform-xhci-hcd.2.auto-usb-0:1.2:1.0-video-index0",
    ),
    "cam2": os.getenv(
        "CAM2_DEVICE",
        "/dev/v4l/by-path/"
        "platform-xhci-hcd.2.auto-usb-0:1.4.2:1.0-video-index0",
    ),
}
ML_API_BASES = {
    camera_id: os.getenv(f"{camera_id.upper()}_API_BASE", f"http://localhost:{port}")
    for camera_id, port in CAM_PORTS.items()
}

# `arduino-app-cli app start` only launches main.py, so nothing else would ever
# start the camera side. Set START_ML=0 to run it by hand instead.
START_ML = os.getenv("START_ML", "1") != "0"
ML_RESTART_INTERVAL_SECONDS = 5.0

AUTONOMY_CAMERA_ID = "cam1"  # which camera the scan/approach behavior uses

SNAPSHOT_TEMP_PATH_TEMPLATE = "/tmp/last_person_snapshot_{camera_id}.jpg"
FRAME_WIDTH = 640       # matches ml_RecordAndDetect.py's FRAME_WIDTH default
FRAME_HEIGHT = 480      # matches ml_RecordAndDetect.py's FRAME_HEIGHT default


# ============================================================
# ml_RecordAndDetect.py's local API - one base URL per camera now
# ============================================================
def snapshot_url(camera_id: str) -> str:
    return f"{ML_API_BASES[camera_id]}/snapshot"


def detections_url(camera_id: str) -> str:
    return f"{ML_API_BASES[camera_id]}/api/detections"


# ============================================================
# Bridge access, serialized (multiple threads touch this now)
# ============================================================
bridge_lock = threading.Lock()


def safe_bridge_call(*args, **kwargs):
    with bridge_lock:
        return Bridge.call(*args, **kwargs)


# ============================================================
# Mode state (Manual / Autonomous)
# ============================================================
mode_lock = threading.Lock()
current_mode = "MANUAL"  # always boot safe - never auto-drive on power-up
autonomy_reset_event = threading.Event()


def set_mode(new_mode: str):
    global current_mode
    with mode_lock:
        if new_mode not in ("MANUAL", "AUTONOMOUS"):
            print(f"Ignoring unknown mode: {new_mode}")
            return
        changed = current_mode != new_mode
        current_mode = new_mode
    if changed:
        print(f"Mode switched to {new_mode}")
        if new_mode == "MANUAL":
            try:
                safe_bridge_call("cmd_stop")
            except Exception as e:
                print("Stop-on-mode-switch failed:", e)
        else:
            autonomy_reset_event.set()


def get_mode() -> str:
    with mode_lock:
        return current_mode


def is_manual() -> bool:
    return get_mode() == "MANUAL"


# ============================================================
# Alert reporting (POST /events)
# ============================================================
def report_alert(
    alert_type: str,
    source: str,
    location_x: float,
    location_y: float,
    bbox_width: float,
    bbox_height: float,
    confidence_score: float,
    injury_class: str,
    camera_id: str,
    motor_halt_requested: bool = False,
    status: str = "NEW",
    image_path: str | None = None,
):
    payload = {
        "roverId": ROVER_ID,
        "sessionId": SESSION_ID,
        "alertType": alert_type,
        "source": source,
        "detectedAt": datetime.now(timezone.utc).isoformat(),
        "locationX": location_x,
        "locationY": location_y,
        "boundingBoxWidth": bbox_width,
        "boundingBoxHeight": bbox_height,
        "confidenceScore": confidence_score,
        "motorHaltRequested": motor_halt_requested,
        "injuryClass": injury_class,
        "cameraId": camera_id,
        "status": status,
    }
    try:
        response = requests.post(f"{API_BASE}/events", json=payload, timeout=5)
        response.raise_for_status()
        event = response.json()
        event_id = event.get("id")
        print(f"Alert reported, event id={event_id}")
        if image_path and event_id is not None:
            upload_event_image(event_id, image_path)
        return event_id
    except requests.RequestException as e:
        print("Failed to report alert:", e)
        return None


def upload_event_image(event_id: int, image_path: str):
    try:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.split("/")[-1], f, "image/jpeg")}
            response = requests.post(f"{API_BASE}/events/{event_id}/image", files=files, timeout=10)
        response.raise_for_status()
        print("Image uploaded:", response.json().get("imageUrl"))
    except requests.RequestException as e:
        print("Failed to upload image:", e)


# ============================================================
# Battery telemetry
# ============================================================
def battery_raw_to_percent(raw: float) -> float:
    v_adc = (raw / 255.0) * BATTERY_VREF
    v_battery = v_adc * (BATTERY_R1_OHMS + BATTERY_R2_OHMS) / BATTERY_R2_OHMS
    percent = (v_battery - BATTERY_V_EMPTY) / (BATTERY_V_FULL - BATTERY_V_EMPTY) * 100
    return round(max(0, min(100, percent)))


def send_telemetry():
    raw = safe_bridge_call("get_battery_raw")
    percent = battery_raw_to_percent(raw)
    print(f"Battery: raw={raw} -> {percent}%")
    try:
        response = requests.post(
            TELEMETRY_URL,
            json={"roverId": ROBOT_GROUP_ID, "battery": percent},
            timeout=5,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print("Telemetry POST failed (endpoint may not be deployed yet):", e)


def telemetry_loop():
    while True:
        try:
            send_telemetry()
        except Exception as e:
            print("Telemetry read/send failed:", e)
        time.sleep(TELEMETRY_INTERVAL_SECONDS)


# ============================================================
# Manual command handling: latest-state-wins, not a queue
# ============================================================
def scale_speed(dto_speed) -> int:
    """SendCommandDto.Speed is 0-100 (percent of max, confirmed). Rover's
    motor driver expects 0-255 (PWM scale)."""
    if dto_speed is None:
        return DEFAULT_SPEED
    return round(max(0, min(100, dto_speed)) * 255 / 100)


desired_state = {"command": "stop", "speed": None, "degrees": None}
desired_state_lock = threading.Lock()
new_command_event = threading.Event()

last_command_time = time.time()
last_command_was_moving = False

autonomous_cruise_speed = DEFAULT_CRUISE_SPEED  # updated by "set-speed" commands


def handle_command(args):
    """
    args[0] matches SendCommandDto exactly:
      { roverId, command, speed, degrees }
    """
    global autonomous_cruise_speed
    try:
        if not args:
            return
        payload = args[0] if isinstance(args, list) else args
        command = payload.get("command")
        speed = payload.get("speed")
        degrees = payload.get("degrees", 90)

        print(f"Received: {command} (speed={speed}, degrees={degrees})")

        if command == "set-mode-manual":
            set_mode("MANUAL")
            return
        if command == "set-mode-autonomous":
            set_mode("AUTONOMOUS")
            return
        if command == "set-speed":
            autonomous_cruise_speed = scale_speed(speed)
            return

        if not is_manual():
            print(f"Ignoring drive command '{command}' - not in MANUAL mode")
            return

        with desired_state_lock:
            desired_state["command"] = command
            desired_state["speed"] = speed
            desired_state["degrees"] = degrees
        new_command_event.set()
    except Exception as e:
        print("handle_command FAILED:", e)


ACTIONS = {
    "forward": lambda speed, degrees: safe_bridge_call("cmd_forward", speed),
    "backward": lambda speed, degrees: safe_bridge_call("cmd_backward", speed),
    "stop": lambda speed, degrees: safe_bridge_call("cmd_stop"),
    "turn-left": lambda speed, degrees: safe_bridge_call("cmd_turn_left", speed),
    "turn-right": lambda speed, degrees: safe_bridge_call("cmd_turn_right", speed),
    "arc-left": lambda speed, degrees: safe_bridge_call("cmd_arc_left", speed),
    "arc-right": lambda speed, degrees: safe_bridge_call("cmd_arc_right", speed),
    "turn-degrees": lambda speed, degrees: safe_bridge_call("cmd_turn_degrees", degrees, speed),
}


def apply_command_loop():
    global last_command_time, last_command_was_moving
    last_applied = None

    while True:
        new_command_event.wait(timeout=1.0)
        new_command_event.clear()

        with desired_state_lock:
            command = desired_state["command"]
            speed = scale_speed(desired_state["speed"])
            degrees = desired_state["degrees"]

        signature = (command, speed, degrees)
        if signature == last_applied:
            continue

        action = ACTIONS.get(command)
        if not action:
            print("Unknown command:", command)
            continue

        try:
            print(f"Executing: {command} (speed={speed}, degrees={degrees})")
            action(speed, degrees)
            last_applied = signature
            last_command_time = time.time()
            last_command_was_moving = command in MOVING_COMMANDS
        except Exception as e:
            print("Command execution FAILED, stopping motors as a precaution:", e)
            try:
                safe_bridge_call("cmd_stop")
            except Exception:
                pass


def watchdog_loop():
    global last_command_was_moving
    while True:
        time.sleep(0.2)
        if last_command_was_moving and (time.time() - last_command_time) > WATCHDOG_TIMEOUT_SECONDS:
            print(f"WATCHDOG: no command for {WATCHDOG_TIMEOUT_SECONDS}s while moving - forcing stop")
            try:
                safe_bridge_call("cmd_stop")
            except Exception as e:
                print("WATCHDOG stop call failed:", e)
            last_command_was_moving = False


# ============================================================
# Autonomy: scan, detect via ml_RecordAndDetect.py, approach, alert, hold
# ============================================================
# SAFETY: no distance/obstacle sensor exists on this robot, only a camera.
# This state machine can only "see" what its detection model recognizes -
# no wall, step, or furniture awareness. Movement is pulsed (not
# continuous) and re-checks the camera between every pulse. This is a
# mitigation, not a substitute for a real safety sensor. Test only
# supervised, in open space, with Manual mode one button-press away.

AUTO_STATE_SCANNING = "SCANNING"
AUTO_STATE_APPROACHING = "APPROACHING"
AUTO_STATE_HOLDING = "HOLDING"


def capture_snapshot(camera_id: str = AUTONOMY_CAMERA_ID) -> str | None:
    path = SNAPSHOT_TEMP_PATH_TEMPLATE.format(camera_id=camera_id)
    try:
        response = requests.get(snapshot_url(camera_id), timeout=3)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        return path
    except requests.RequestException as e:
        print(f"Autonomy: snapshot capture failed for {camera_id} ({e}) - alert will go out without a photo")
        return None


def detect_objects_in_frame(camera_id: str = AUTONOMY_CAMERA_ID):
    """Polls ml_RecordAndDetect.py for that camera's own port and converts its
    response to [{"class": "person", "confidence": c, "bbox": (x,y,w,h)}, ...]"""
    try:
        response = requests.get(detections_url(camera_id), timeout=2)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Autonomy: detector API unreachable ({e}) - treating as no detections")
        return []

    detections = []
    for person in data.get("persons", []):
        x1, y1, x2, y2 = person["x1"], person["y1"], person["x2"], person["y2"]
        detections.append({
            "class": "person",
            "confidence": person["confidence"],
            "bbox": (x1, y1, x2 - x1, y2 - y1),
        })
    return detections


def best_person_detection(detections):
    people = [d for d in detections if d.get("class") == "person" and d.get("confidence", 0) >= PERSON_CONFIDENCE_THRESHOLD]
    if not people:
        return None
    return max(people, key=lambda d: d["confidence"])


def autonomy_scan_step():
    detections = detect_objects_in_frame()
    person = best_person_detection(detections)
    if person:
        return person

    safe_bridge_call("cmd_turn_left", SCAN_ROTATE_SPEED)
    time.sleep(SCAN_ROTATE_BURST_SECONDS)
    safe_bridge_call("cmd_stop")

    detections = detect_objects_in_frame()
    person = best_person_detection(detections)
    if person:
        return person

    time.sleep(SCAN_IDLE_SECONDS)
    return None


def autonomy_approach_step(last_seen_person, last_seen_time):
    detections = detect_objects_in_frame()
    person = best_person_detection(detections)

    if person is None:
        if time.time() - last_seen_time > LOST_TARGET_TIMEOUT_SECONDS:
            print("Autonomy: lost the person, returning to scanning")
            safe_bridge_call("cmd_stop")
            return False, False, None
        return False, True, last_seen_person

    x, y, w, h = person["bbox"]
    box_center_x = x + w / 2
    frame_center_x = FRAME_WIDTH / 2
    offset_fraction = (box_center_x - frame_center_x) / FRAME_WIDTH
    area_fraction = (w * h) / (FRAME_WIDTH * FRAME_HEIGHT)

    if area_fraction >= CLOSE_ENOUGH_AREA_FRACTION:
        print(f"Autonomy: close enough to person (area={area_fraction:.2f}), stopping")
        safe_bridge_call("cmd_stop")
        return True, True, person

    approach_speed = min(autonomous_cruise_speed, APPROACH_SPEED_CAP)

    if abs(offset_fraction) > CENTER_DEADZONE_FRACTION:
        if offset_fraction > 0:
            safe_bridge_call("cmd_turn_right", SCAN_ROTATE_SPEED)
        else:
            safe_bridge_call("cmd_turn_left", SCAN_ROTATE_SPEED)
        time.sleep(TURN_CORRECTION_SECONDS)
        safe_bridge_call("cmd_stop")
    else:
        safe_bridge_call("cmd_forward", approach_speed)
        time.sleep(APPROACH_PULSE_SECONDS)
        safe_bridge_call("cmd_stop")

    return False, True, person


def autonomy_loop():
    state = AUTO_STATE_SCANNING
    hold_started_at = None
    last_seen_person = None
    last_seen_time = 0

    while True:
        if get_mode() != "AUTONOMOUS":
            time.sleep(0.3)
            continue

        if autonomy_reset_event.is_set():
            autonomy_reset_event.clear()
            state = AUTO_STATE_SCANNING
            hold_started_at = None
            print("Autonomy: reset to SCANNING")

        try:
            if state == AUTO_STATE_SCANNING:
                person = autonomy_scan_step()
                if person:
                    print("Autonomy: person detected, switching to APPROACHING")
                    state = AUTO_STATE_APPROACHING
                    last_seen_person = person
                    last_seen_time = time.time()

            elif state == AUTO_STATE_APPROACHING:
                arrived, still_tracking, person = autonomy_approach_step(last_seen_person, last_seen_time)
                if person:
                    last_seen_person = person
                    last_seen_time = time.time()

                if arrived:
                    # NOTE: comms_manager.py already sends its own "Human
                    # Detected" alert the moment a person first appears in
                    # frame, independent of this robot's approach behavior.
                    # This is a different, later event (arrival), tagged
                    # distinctly so the dashboard shows both as separate
                    # signals rather than looking like a duplicate.
                    report_alert(
                        alert_type="PERSON_REACHED",
                        source="autonomy",
                        location_x=0,  # TODO: no localization system
                        location_y=0,
                        bbox_width=person["bbox"][2],
                        bbox_height=person["bbox"][3],
                        confidence_score=person["confidence"],
                        injury_class="UNKNOWN",  # TODO: confirm allowed values with backend team
                        camera_id=AUTONOMY_CAMERA_ID,
                        motor_halt_requested=True,
                        status="NEW",
                        image_path=capture_snapshot(),
                    )
                    state = AUTO_STATE_HOLDING
                    hold_started_at = time.time()
                elif not still_tracking:
                    state = AUTO_STATE_SCANNING

            elif state == AUTO_STATE_HOLDING:
                if hold_started_at and (time.time() - hold_started_at) > HOLD_COOLDOWN_SECONDS:
                    print("Autonomy: hold cooldown elapsed, resuming scanning")
                    state = AUTO_STATE_SCANNING
                    hold_started_at = None
                else:
                    time.sleep(1.0)

        except Exception as e:
            print("Autonomy loop error, stopping for safety:", e)
            try:
                safe_bridge_call("cmd_stop")
            except Exception:
                pass
            state = AUTO_STATE_SCANNING
            time.sleep(1.0)


# ============================================================
# Camera pipeline process: started here so a single
# `arduino-app-cli app start ~/ArduinoApps/castel` brings up both halves.
# ============================================================
def start_ml_process():
    environment = {
        **os.environ,
        "API_BASE": API_BASE,
        "ROVER_ID": ROVER_ID,
        "SESSION_ID": SESSION_ID,
        # Keep ml_RecordAndDetect.py's per-camera ports in sync with what
        # this file is polling.
        "CAM1_PORT": str(CAM_PORTS["cam1"]),
        "CAM2_PORT": str(CAM_PORTS["cam2"]),
        "CAM1_DEVICE": CAM_DEVICES["cam1"],
        "CAM2_DEVICE": CAM_DEVICES["cam2"],
    }

    print(f"ml: starting {os.path.basename(ML_SCRIPT)}")
    return subprocess.Popen([sys.executable, ML_SCRIPT], env=environment)


def ml_supervisor_loop():
    if not os.path.exists(ML_SCRIPT):
        print(
            f"ml: {ML_SCRIPT} not found, cameras will not run. Check that "
            "arduino-app-cli synced the whole python/ folder and not just main.py."
        )
        return

    process = None

    while True:
        if process is None or process.poll() is not None:
            if process is not None:
                print(f"ml: exited with code {process.returncode}, restarting")

            try:
                process = start_ml_process()
            except Exception as e:
                print(f"ml: could not start ({e})")

        time.sleep(ML_RESTART_INTERVAL_SECONDS)


# ============================================================
# SignalR connection
# ============================================================
# Tracks when the hub connection was lost so on_reconnected() can report
# exactly how long the rover was uncontrollable, instead of guessing from
# log timestamps by hand. Set by on_hub_closed()/on_hub_error(), cleared
# once a reconnect is confirmed.
connection_lost_at = None
connection_lost_lock = threading.Lock()


def _mark_connection_lost():
    global connection_lost_at
    with connection_lost_lock:
        if connection_lost_at is None:
            connection_lost_at = time.time()


def register_robot():
    print("Registering as", repr(ROBOT_GROUP_ID))
    try:
        hub_connection.send("RegisterRobot", [ROBOT_GROUP_ID])
        print("RegisterRobot invocation sent (no immediate exception)")
    except Exception as e:
        print("RegisterRobot send FAILED:", e)


def on_connected():
    print("Connected to DashboardHub (initial connection)")
    register_robot()


def on_reconnected():
    # signalrcore fires on_open only for the very first connection, and
    # on_reconnect for every subsequent automatic reconnect - group
    # membership does not survive a reconnect, so re-register here too.
    global connection_lost_at
    with connection_lost_lock:
        lost_at = connection_lost_at
        connection_lost_at = None

    if lost_at is not None:
        downtime = time.time() - lost_at
        print(f"Reconnected to DashboardHub after {downtime:.1f}s downtime - re-registering")
    else:
        print("Reconnected to DashboardHub (no recorded disconnect time) - re-registering")
    register_robot()


def on_registered(args):
    print("Hub confirmed registration:", args)


def on_hub_closed():
    _mark_connection_lost()
    print(f"Disconnected from DashboardHub at {datetime.now(timezone.utc).isoformat()} - stopping motors as a precaution")
    try:
        safe_bridge_call("cmd_stop")
    except Exception as e:
        print("Stop-on-disconnect failed:", e)


def on_hub_error(data):
    # signalrcore surfaces transport-level failures (dropped socket, failed
    # send, etc.) here, and this often fires before/instead of on_close for
    # a connection that just goes silent rather than closing cleanly. This
    # is the main defense against "button pressed, wifi dies, robot keeps
    # driving": the frontend only sends a drive command once on button-down
    # and "stop" once on button-up (no repeat-while-held), so the watchdog
    # can't distinguish a held button from a dead connection and stays as a
    # 10s last-resort backstop only. This handler is what actually reacts
    # quickly to a lost connection.
    _mark_connection_lost()
    print("SignalR ERROR:", data, "- stopping motors as a precaution")
    try:
        safe_bridge_call("cmd_stop")
    except Exception as e:
        print("Stop-on-error failed:", e)


hub_connection = (
    HubConnectionBuilder()
    .with_url(HUB_URL, options={"verify_ssl": False})
    .configure_logging(logging.DEBUG, socket_trace=True)
    .with_automatic_reconnect({
        "type": "raw",
        "keep_alive_interval": 10,
        "reconnect_interval": 5,
        "max_attempts": None,
    })
    .build()
)

hub_connection.on(COMMAND_EVENT, handle_command)
hub_connection.on("RobotRegistered", on_registered)
hub_connection.on_open(on_connected)
hub_connection.on_reconnect(on_reconnected)
hub_connection.on_close(on_hub_closed)
hub_connection.on_error(on_hub_error)
hub_connection.start()

# ============================================================
# Background threads
# ============================================================
threading.Thread(target=apply_command_loop, daemon=True).start()
threading.Thread(target=watchdog_loop, daemon=True).start()
threading.Thread(target=autonomy_loop, daemon=True).start()

# Autonomy will report the detector as unreachable for the first minute or so
# while ml_RecordAndDetect.py loads its YOLO model. It recovers on its own.
if START_ML:
    threading.Thread(target=ml_supervisor_loop, daemon=True).start()

if ENABLE_TELEMETRY:
    threading.Thread(target=telemetry_loop, daemon=True).start()

App.run()
