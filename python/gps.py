# SPDX-License-Identifier: MIT
"""
GPS localization for the SIM8262E-M2 GNSS receiver on Sânzi.

Polls AT+CGPSINFO directly over the modem's AT port. This is deliberately
simpler than the previous NMEA-port-autodiscovery / mmcli-fallback version:
main.py now runs inside a Docker container (see app.yaml), which does not
have host device-node access to every /dev/ttyUSB* port or D-Bus access to
the host's ModemManager - both of those approaches silently found nothing
and spun forever. Polling one explicitly-configured AT port only requires
that single device node to be passed into the container, nothing else.

main.py owns the reader thread and writes /tmp/rover_gps.json; ml.py reads
that file when attaching lat/lon to alert payloads.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import serial

# ---------- Config (all overridable via environment) ----------
# Must point at the modem's AT-capable port INSIDE the container - check
# `mmcli -m <id>` on the HOST for the current "(at)" port (it moves between
# ttyUSB2/ttyUSB3 depending on USB composition/enumeration order), then make
# sure that exact device node is passed through in app.yaml.
GPS_AT_PORT = os.getenv("GPS_AT_PORT", "/dev/ttyUSB2")
GPS_BAUD = int(os.getenv("GPS_BAUD", "115200"))
GPS_SHARED_FILE = os.getenv("GPS_SHARED_FILE", "/tmp/rover_gps.json")
GPS_POLL_INTERVAL = float(os.getenv("GPS_POLL_INTERVAL", "2.0"))
GPS_RETRY_SECONDS = float(os.getenv("GPS_RETRY_SECONDS", "5"))

_lock = threading.Lock()
_latest: dict[str, Any] = {
    "lat": None,
    "lon": None,
    "fix": False,
    "updated_at": None,
    "backend": "at-cgpsinfo",
}

# +CGPSINFO: <lat>,<N/S>,<lon>,<E/W>,<date>,<utc>,<alt>,<speed>,<course>
# An unfixed reading returns all-empty fields: +CGPSINFO: ,,,,,,,,
_CGPSINFO_RE = re.compile(
    r"\+CGPSINFO:\s*"
    r"(?P<lat>[\d.]*),(?P<ns>[NS]?),"
    r"(?P<lon>[\d.]*),(?P<ew>[EW]?),"
)


def get_gps() -> dict:
    with _lock:
        return dict(_latest)


def _write_shared(snapshot: dict) -> None:
    try:
        with open(GPS_SHARED_FILE, "w") as f:
            json.dump(snapshot, f)
    except OSError as exc:
        print(f"GPS: failed to write {GPS_SHARED_FILE} ({exc})")


def _publish(lat: float | None, lon: float | None, fix: bool) -> None:
    with _lock:
        _latest["fix"] = fix
        if fix and lat is not None and lon is not None:
            _latest["lat"] = lat
            _latest["lon"] = lon
            _latest["updated_at"] = time.time()
        _write_shared(dict(_latest))


def _nmea_coord_to_decimal(raw: str, hemisphere: str) -> float | None:
    """CGPSINFO reports ddmm.mmmmmm (lat) / dddmm.mmmmmm (lon), NOT plain
    decimal degrees - convert to signed decimal degrees."""
    if not raw:
        return None
    dot = raw.find(".")
    if dot < 2:
        return None
    degrees = float(raw[: dot - 2])
    minutes = float(raw[dot - 2 :])
    decimal = degrees + minutes / 60.0
    if hemisphere in ("S", "W"):
        decimal = -decimal
    return decimal


def _at_exchange(ser: serial.Serial, command: str, wait: float = 1.0) -> str:
    ser.reset_input_buffer()
    ser.write(f"{command}\r\n".encode())
    time.sleep(wait)
    return ser.read(1024).decode("ascii", errors="replace")


def _enable_gnss(ser: serial.Serial) -> None:
    response = _at_exchange(ser, "AT+CGPS=1", wait=1.0)
    print(f"GPS: AT+CGPS=1 -> {response.strip()[:120] or '(no response)'}")


def _poll_once(ser: serial.Serial) -> None:
    response = _at_exchange(ser, "AT+CGPSINFO", wait=1.0)
    match = _CGPSINFO_RE.search(response)
    if not match:
        _publish(None, None, False)
        return

    lat_raw, ns, lon_raw, ew = (
        match.group("lat"),
        match.group("ns"),
        match.group("lon"),
        match.group("ew"),
    )
    if not lat_raw or not lon_raw:
        # No fix yet - AT+CGPSINFO returns empty fields until one is acquired.
        _publish(None, None, False)
        return

    lat = _nmea_coord_to_decimal(lat_raw, ns)
    lon = _nmea_coord_to_decimal(lon_raw, ew)
    if lat is None or lon is None:
        _publish(None, None, False)
        return

    _publish(lat, lon, True)


def _reader_loop() -> None:
    gnss_enabled = False

    while True:
        try:
            with serial.Serial(GPS_AT_PORT, GPS_BAUD, timeout=2) as ser:
                if not gnss_enabled:
                    _enable_gnss(ser)
                    gnss_enabled = True
                    # Give the GNSS engine a moment before the first poll.
                    time.sleep(2.0)

                print(f"GPS: polling {GPS_AT_PORT} via AT+CGPSINFO every {GPS_POLL_INTERVAL}s")
                while True:
                    _poll_once(ser)
                    time.sleep(GPS_POLL_INTERVAL)

        except (serial.SerialException, OSError) as exc:
            print(
                f"GPS: cannot open {GPS_AT_PORT} ({exc}) - is it passed through "
                f"to the container? Retrying in {GPS_RETRY_SECONDS}s"
            )
            _publish(None, None, False)
            gnss_enabled = False
            time.sleep(GPS_RETRY_SECONDS)


def start_gps_thread() -> None:
    threading.Thread(target=_reader_loop, daemon=True, name="gps").start()