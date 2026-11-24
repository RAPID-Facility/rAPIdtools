# Copyright (c) 2025 The University of Washington
#
# This file is part of rapidtools.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# rapidtools. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 11-12-2026

"""
Simulate a drone flight over real buildings with Google Earth's 3D imagery.

Test imagery for the drone workflow (``building_inference_from_drone.py``)
without owning a drone: this script flies a virtual camera over a block of
buildings in the Google Earth web app, which renders textured 3D
photogrammetry in a browser, and saves what it sees as photos that carry
their camera pose exactly like the photos of a DJI aircraft do (EXIF GPS
tags plus the ``drone-dji`` XMP block with gimbal angles). The flight is the
usual oblique mapping pattern: a grid of shots looking north, east, south
and west with the gimbal tilted down, plus a sparser straight-down grid.

How it works:

- Headless Chrome is driven over the DevTools protocol (the ``websocket``
  module from the ``websocket-client`` package; nothing else is needed).
- Google Earth takes its camera from the address bar:
  ``@lat,lon,<a>a,<d>d,<fov>y,<heading>h,<tilt>t,0r`` where ``lat, lon, a``
  is the point looked at (``a`` in meters above sea level), ``d`` the
  distance from the camera to it, ``fov`` the VERTICAL field of view of the
  whole window, ``heading`` the compass direction of the view and ``tilt``
  the angle from straight down (0 = nadir, 90 = horizon). Verified by
  projecting building footprints into renders; the camera altitude shown in
  the status bar is checked against every pose.
- The page reports its streaming progress and the elevation under the
  cursor through its accessibility tree, which is how the script knows a
  view has finished loading.
- Each screenshot is cropped to remove the toolbar and the status bar. The
  crop is symmetric, so the principal point stays at the image center.

The imagery belongs to Google and its data providers: use the renders to
test software, not to redistribute.

Usage:
    python examples/drone_flight_from_google_earth.py \\
        --footprints examples/eaton_patch1_buildings_final.geojson \\
        --output out/drone_flight --launch-chrome
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path

import requests
import websocket  # pip install websocket-client
from PIL import Image

from rapidtools.data_sources import DroneCamera, write_pose_metadata

EARTH_URL = 'https://earth.google.com/web/'
EARTH_RADIUS_M = 6378137.0
WINDOW = (2000, 1500)  # browser viewport in pixels
CROP_ROWS = 110  # removed from the top and from the bottom
LABELS_JS = (
    "JSON.stringify([...document.querySelectorAll('flt-semantics')]"
    ".filter(e=>!e.querySelector('flt-semantics'))"
    ".map(e=>(e.getAttribute('aria-label')||e.innerText||'')))"
)


class EarthBrowser:
    """A Google Earth tab driven through the Chrome DevTools protocol."""

    def __init__(self, port: int) -> None:
        tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
        page = next(t for t in tabs if t['type'] == 'page')
        self.ws = websocket.create_connection(
            page['webSocketDebuggerUrl'], suppress_origin=True
        )
        self._id = 0
        self.send('Page.enable')
        self.send('Runtime.enable')

    def send(self, method: str, params: dict | None = None, timeout: float = 60):
        self._id += 1
        self.ws.send(
            json.dumps({'id': self._id, 'method': method, 'params': params or {}})
        )
        deadline = time.time() + timeout
        while True:
            self.ws.settimeout(max(0.1, deadline - time.time()))
            message = json.loads(self.ws.recv())
            if message.get('id') == self._id:
                return message.get('result', message)

    def evaluate(self, expression: str):
        result = self.send(
            'Runtime.evaluate',
            {'expression': expression, 'awaitPromise': True, 'returnByValue': True},
        )
        return result.get('result', {}).get('value')

    def click(self, x: float, y: float) -> None:
        for kind in ('mousePressed', 'mouseReleased'):
            self.send(
                'Input.dispatchMouseEvent',
                {'type': kind, 'x': x, 'y': y, 'button': 'left', 'clickCount': 1},
            )

    def labels(self) -> list[str]:
        """Text of every leaf of the page's accessibility tree."""
        return json.loads(self.evaluate(LABELS_JS) or '[]')

    def click_label(self, prefix: str) -> bool:
        """Click the first control whose label starts with ``prefix``."""
        box = self.evaluate(
            "(()=>{const e=[...document.querySelectorAll('flt-semantics')]"
            ".filter(e=>!e.querySelector('flt-semantics'))"
            ".find(e=>(e.getAttribute('aria-label')||e.innerText||'')"
            f'.startsWith({json.dumps(prefix)}));'
            'if(!e)return null;const r=e.getBoundingClientRect();'
            'return [r.x+r.width/2,r.y+r.height/2]})()'
        )
        if not box:
            return False
        self.click(*box)
        time.sleep(1.0)
        return True

    def open_earth(self) -> None:
        """Load the app, switch to the clean basemap and close the pop-ups."""
        width, height = WINDOW
        self.send(
            'Emulation.setDeviceMetricsOverride',
            {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': False},
        )
        self.send('Page.navigate', {'url': EARTH_URL})
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(2)
            # The app draws on a canvas; clicking this placeholder makes it
            # publish its controls as an accessibility tree we can read.
            self.evaluate(
                "(()=>{const p=document.querySelector('flt-semantics-placeholder');"
                'if(p)p.click()})()'
            )
            if any(text.startswith('Progress toward') for text in self.labels()):
                break
        else:
            raise RuntimeError('Google Earth did not finish loading.')
        for _ in range(6):  # announcement dialogs and feature tips
            if not (self.click_label('Dismiss') or self.click_label('Close')):
                break
        if self.click_label('Basemap settings'):
            self.click_label('Clean')  # no labels, borders or roads
            self.click_label('Basemap settings')
        for _ in range(3):
            if not self.click_label('Dismiss'):
                break

    def look_at(self, lat, lon, alt, distance, fov, heading, tilt) -> None:
        url = (
            f'{EARTH_URL}@{lat:.7f},{lon:.7f},{alt:.2f}a,{distance:.2f}d,'
            f'{fov:.2f}y,{heading:.2f}h,{tilt:.2f}t,0r'
        )
        self.evaluate(
            f"history.pushState({{}}, '', {json.dumps(url)});"
            "window.dispatchEvent(new PopStateEvent('popstate', {state: {}})); 1"
        )

    def status(self) -> tuple[int | None, int | None]:
        """Streaming progress in percent and the camera altitude in meters."""
        progress = altitude = None
        for text in self.labels():
            if text.startswith('Progress toward'):
                digits = ''.join(c for c in text.split('imagery')[-1] if c.isdigit())
                progress = int(digits[:3]) if digits else None
                if progress and progress > 100:  # "70% 70" while loading
                    progress = int(digits[:2])
            elif text.startswith('Altitude of the camera'):
                digits = ''.join(
                    c for c in text.split('Camera:')[-1] if c.isdigit() or c == ','
                )
                altitude = int(digits.replace(',', '')) if digits else None
        return progress, altitude

    def screenshot(self, scale: float = 1.0) -> Image.Image:
        width, height = WINDOW
        result = self.send(
            'Page.captureScreenshot',
            {
                'format': 'jpeg',
                'quality': 95 if scale == 1.0 else 60,
                'clip': {
                    'x': 0,
                    'y': 0,
                    'width': width,
                    'height': height,
                    'scale': scale,
                },
            },
        )
        return Image.open(io.BytesIO(base64.b64decode(result['data']))).convert('RGB')

    def wait_until_rendered(self, timeout: float = 40.0) -> bool:
        """Wait for streaming to finish and the picture to stop changing."""
        start = time.time()
        time.sleep(1.5)  # the fly-to animation and the first tile requests
        previous, stable = None, 0
        while time.time() - start < timeout:
            progress, _ = self.status()
            thumb = self.screenshot(0.1).convert('L')
            if previous is not None and progress == 100:
                diff = sum(
                    abs(a - b)
                    for a, b in zip(thumb.getdata(), previous.getdata(), strict=True)
                ) / (thumb.width * thumb.height)
                stable = stable + 1 if diff < 0.6 else 0
                if stable >= 2:
                    return True
            previous = thumb
            time.sleep(0.6)
        return False


def offset(
    lat: float, lon: float, east_m: float, north_m: float
) -> tuple[float, float]:
    """Latitude and longitude ``east_m``/``north_m`` meters from a point."""
    return (
        lat + math.degrees(north_m / EARTH_RADIUS_M),
        lon + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))),
    )


def ground_elevation(lon: float, lat: float) -> float:
    """Ground elevation in meters from the USGS point query service (US only)."""
    reply = requests.get(
        'https://epqs.nationalmap.gov/v1/json',
        params={'x': lon, 'y': lat, 'units': 'Meters', 'wkid': 4326},
        timeout=60,
    )
    return float(reply.json()['value'])


def plan_flight(bounds, altitude_agl, pitch, headings, spacing, nadir_spacing, pad):
    """
    Camera poses of an oblique grid flight.

    For every heading the camera looks at a grid of ground points covering
    the area, from ``altitude_agl / tan(-pitch)`` meters behind each point.
    Returns ``(lat, lon, yaw, pitch)`` tuples in flying order.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    lat0, lon0 = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    half_x = (
        math.radians(max_lon - min_lon)
        / 2
        * EARTH_RADIUS_M
        * math.cos(math.radians(lat0))
        + pad
    )
    half_y = math.radians(max_lat - min_lat) / 2 * EARTH_RADIUS_M + pad

    def grid(step):
        nx = max(1, math.ceil(2 * half_x / step))
        ny = max(1, math.ceil(2 * half_y / step))
        rows = []
        for j in range(ny + 1):
            row = [
                (-half_x + i * 2 * half_x / nx, -half_y + j * 2 * half_y / ny)
                for i in range(nx + 1)
            ]
            rows.append(row if j % 2 == 0 else row[::-1])  # serpentine
        return [point for row in rows for point in row]

    poses = []
    back = altitude_agl / math.tan(math.radians(-pitch)) if pitch > -89.9 else 0.0
    for heading in headings:
        for x, y in grid(spacing):
            east = x - back * math.sin(math.radians(heading))
            north = y - back * math.cos(math.radians(heading))
            poses.append(
                (*offset(lat0, lon0, east, north), float(heading), float(pitch))
            )
    if nadir_spacing > 0:
        for x, y in grid(nadir_spacing):
            poses.append((*offset(lat0, lon0, x, y), 0.0, -90.0))
    return poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--footprints', help='GeoJSON whose extent is flown over')
    parser.add_argument(
        '--bounds',
        nargs=4,
        type=float,
        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
    )
    parser.add_argument('--output', default='out/drone_flight')
    parser.add_argument(
        '--altitude-agl',
        type=float,
        default=70.0,
        help='flight height above the take-off point, meters',
    )
    parser.add_argument(
        '--pitch', type=float, default=-45.0, help='gimbal pitch of the oblique shots'
    )
    parser.add_argument('--headings', type=float, nargs='*', default=[0, 90, 180, 270])
    parser.add_argument(
        '--spacing',
        type=float,
        default=50.0,
        help='distance between oblique shots, meters',
    )
    parser.add_argument(
        '--nadir-spacing', type=float, default=70.0, help='0 skips the nadir grid'
    )
    parser.add_argument(
        '--pad', type=float, default=15.0, help='margin flown around the area, meters'
    )
    parser.add_argument(
        '--fov',
        type=float,
        default=35.0,
        help='vertical field of view of the browser window',
    )
    parser.add_argument(
        '--takeoff-elevation',
        type=float,
        help='meters above sea level; looked up from USGS when omitted',
    )
    parser.add_argument(
        '--port', type=int, default=9444, help='Chrome remote debugging port'
    )
    parser.add_argument(
        '--launch-chrome',
        action='store_true',
        help='start headless Chrome instead of attaching to one',
    )
    parser.add_argument(
        '--limit', type=int, default=0, help='stop after this many photos'
    )
    args = parser.parse_args()

    if args.footprints:
        xs, ys = [], []
        for feature in json.load(open(args.footprints))['features']:
            for ring in feature['geometry']['coordinates']:
                ring = (
                    ring[0] if feature['geometry']['type'] == 'MultiPolygon' else ring
                )
                xs += [p[0] for p in ring]
                ys += [p[1] for p in ring]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    elif args.bounds:
        bounds = tuple(args.bounds)
    else:
        parser.error('give --footprints or --bounds')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    takeoff = args.takeoff_elevation
    if takeoff is None:
        takeoff = ground_elevation(
            (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        )
    print(
        f'Take-off elevation {takeoff:.1f} m; '
        f'flying {args.altitude_agl:.0f} m above it.'
    )
    poses = plan_flight(
        bounds,
        args.altitude_agl,
        args.pitch,
        args.headings,
        args.spacing,
        args.nadir_spacing,
        args.pad,
    )
    if args.limit:
        poses = poses[: args.limit]
    print(f'{len(poses)} photos planned.')

    chrome = None
    if args.launch_chrome:
        chrome = subprocess.Popen(
            [
                'google-chrome',
                '--headless=new',
                f'--remote-debugging-port={args.port}',
                '--remote-allow-origins=*',
                f'--user-data-dir={output / "chrome_profile"}',
                '--use-gl=angle',
                '--use-angle=gl',
                '--enable-gpu',
                '--ignore-gpu-blocklist',
                '--hide-scrollbars',
                'about:blank',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(4)
    try:
        browser = EarthBrowser(args.port)
        browser.open_earth()
        width, height = WINDOW
        focal_px = (height / 2) / math.tan(math.radians(args.fov) / 2)
        camera_altitude = takeoff + args.altitude_agl
        log = []
        for number, (lat, lon, yaw, pitch) in enumerate(poses, start=1):
            name = f'DJI_{number:04d}.JPG'
            if (output / name).exists():
                continue
            # Google Earth wants the point looked at: put it where the lens
            # axis meets the take-off level.
            tilt = 90.0 + pitch
            distance = args.altitude_agl / math.cos(math.radians(tilt))
            reach = distance * math.sin(math.radians(tilt))
            target_lat, target_lon = offset(
                lat,
                lon,
                reach * math.sin(math.radians(yaw)),
                reach * math.cos(math.radians(yaw)),
            )
            browser.look_at(
                target_lat, target_lon, takeoff, distance, args.fov, yaw, tilt
            )
            rendered = browser.wait_until_rendered()
            _, shown_altitude = browser.status()
            if (
                shown_altitude is not None
                and abs(shown_altitude - camera_altitude) > 1.5
            ):
                print(
                    f'{name}: camera at {shown_altitude} m, expected '
                    f'{camera_altitude:.0f} m; skipped.'
                )
                continue
            frame = browser.screenshot().crop((0, CROP_ROWS, width, height - CROP_ROWS))
            camera = DroneCamera(
                image_id=Path(name).stem,
                lon=lon,
                lat=lat,
                elevation_m=args.altitude_agl,
                yaw_deg=yaw,
                pitch_deg=pitch,
                width=frame.width,
                height=frame.height,
                focal_px=focal_px,
                relative_altitude_m=args.altitude_agl,
                absolute_altitude_m=camera_altitude,
                captured_at=time.strftime(
                    '%Y:%m:%d %H:%M:%S', time.gmtime(1_780_000_000 + 2 * number)
                ),
            )
            write_pose_metadata(frame, output / name, camera)
            log.append(
                {
                    'image': name,
                    'lon': lon,
                    'lat': lat,
                    'altitude': args.altitude_agl,
                    'yaw': yaw,
                    'pitch': pitch,
                    'focal_px': round(focal_px, 2),
                    'width': frame.width,
                    'height': frame.height,
                }
            )
            print(
                f'{name}  yaw {yaw:5.1f}  pitch {pitch:5.1f}  '
                f'{"ok" if rendered else "timed out, saved anyway"}',
                flush=True,
            )
        if log:
            # The same poses as a table: the sidecar route of read_drone_images.
            with open(output / 'flight_log.csv', 'a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(log[0]))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerows(log)
        (output / 'flight.json').write_text(
            json.dumps(
                {
                    'takeoff_elevation_m': takeoff,
                    'altitude_agl_m': args.altitude_agl,
                    'bounds': bounds,
                    'source': 'Google Earth web 3D imagery',
                },
                indent=1,
            )
        )
    finally:
        if chrome is not None:
            chrome.terminate()


if __name__ == '__main__':
    main()
