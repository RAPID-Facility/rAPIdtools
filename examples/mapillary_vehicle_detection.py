"""
This script demonstrates how to:
1. Fetch Mapillary imagery along specified routes within a target bounding box.
2. Extract vehicle segmentation detection polygons from image metadata.
3. Estimate 3D ground locations for each vehicle observation using camera geometry.
4. Cluster nearby spatial observations to consolidate duplicate views of the same vehicle.
5. Optionally crop detected vehicles and classify their condition (intact, damaged, debris) using Gemini.
6. Export consolidated vehicle locations and attributes as a GeoJSON file.
"""

from __future__ import annotations

import base64
import json
import math
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import mapbox_vector_tile
import requests

# --------------------------------------------------------------- Configuration
# User inputs & parameters
BBOX = (-117.535812, 47.688456, -117.442589, 47.737830)  # (min_lon, min_lat, max_lon, max_lat)
OUTPUT_PATH = Path('old_trails_fire_buildings_vehicles.geojson')

# Credentials are never written into this file. Provide them through the
# environment (MAPILLARY_TOKEN, GOOGLE_API_KEY) or through a text file whose
# path the *_FILE variables give. The defaults match the files the other
# examples use: ``rapidtools.download_dataset('mapillary_token')`` writes
# ``mapillary_token.txt`` into the working directory.
MAPILLARY_TOKEN = os.environ.get('MAPILLARY_TOKEN', '')
MAPILLARY_TOKEN_FILE = os.environ.get('MAPILLARY_TOKEN_FILE', 'mapillary_token.txt')
GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
GEMINI_API_KEY_FILE = os.environ.get('GOOGLE_API_KEY_FILE', 'api_key.txt')

# Detection and clustering configuration
MIN_OBSERVATIONS = 1  # Minimum views required (>=2 filters out single-frame noise)
SAVE_OBSERVATIONS = False  # Save per-image observation points to a separate GeoJSON
CLASSIFY_CONDITION = True  # Set to True to classify condition via Gemini

# API Endpoints & Constants
IMAGES_URL = 'https://graph.mapillary.com/images'
DETECTIONS_URL = 'https://graph.mapillary.com/{image_id}/detections'
IMAGE_URL = 'https://graph.mapillary.com/{image_id}'
UW_RAPID_CREATOR_ID = '107708041466249'

VEHICLE_VALUES = (
    'object--vehicle--car',
    'object--vehicle--truck',
    'object--vehicle--other-vehicle',
    'object--vehicle--vehicle-group',
    'object--vehicle--trailer',
    'object--vehicle--caravan',
)

MAX_BBOX_SIZE = 0.002  # degrees; keeps each Graph API request small
MAX_WORKERS = 10
CAMERA_HEIGHT_M = 2.4  # roof-mounted 360 camera on the uw-rapid vehicle
MIN_RANGE_M, MAX_RANGE_M = 2.0, 30.0  # ignore ego-vehicle and far-away blobs
MIN_AREA_FRACTION = 0.0004  # skip specks (fraction of the panorama area)
CLUSTER_RADIUS_M = 4.0  # observations closer than this are the same vehicle
EARTH_RADIUS_M = 6371000.0

CONDITION_PROMPT = (
    'This is a crop from a post-wildfire street-view image showing a vehicle or '
    'what remains of one. Classify its condition as exactly one word: intact '
    '(undamaged or lightly dirty), damaged (visible fire or heat damage but the '
    'body is mostly whole), debris (burned-out shell, collapsed, only the frame '
    'or ash remains), or unclear. Reply with the single word only.'
)


# ----------------------------------------------------------------- helpers
def read_credential(value: str, file: str | Path, name: str) -> str:
    """Return ``value`` if set, otherwise the stripped contents of ``file``."""
    if value.strip():
        return value.strip()
    path = Path(file)
    if path.is_file():
        return path.read_text().strip()
    raise ValueError(
        f'{name} is required: set the environment variable or create {path}.'
    )


def generate_grid(min_lon, min_lat, max_lon, max_lat, step):
    """Split a bounding box into sub-boxes of at most ``step`` degrees."""
    boxes = []
    lon = min_lon
    while lon < max_lon:
        next_lon = min(lon + step, max_lon)
        lat = min_lat
        while lat < max_lat:
            next_lat = min(lat + step, max_lat)
            boxes.append(f'{lon},{lat},{next_lon},{next_lat}')
            lat = next_lat
        lon = next_lon
    return boxes


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def destination(lon, lat, bearing_deg, distance_m):
    """Point reached from (lon, lat) travelling ``distance_m`` along ``bearing_deg``."""
    delta = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    phi1, lam1 = math.radians(lat), math.radians(lon)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(lam2), math.degrees(phi2)


def request_with_retries(url, params, max_retries=6, timeout=60):
    """GET with exponential back-off on rate limits and server errors."""
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            time.sleep((2**attempt) + random.uniform(0, 1))
            continue
        if response.status_code == 200:
            return response
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep((2**attempt) + random.uniform(0, 1))
            continue
        return response
    return response


# --------------------------------------------------------------- step 1
def fetch_images_worker(bbox, token):
    """Return uw-rapid image records (with camera geometry) inside ``bbox``."""
    params = {
        'access_token': token,
        'bbox': bbox,
        'creator_id': UW_RAPID_CREATOR_ID,
        'fields': (
            'id,geometry,captured_at,compass_angle,sequence,camera_type,'
            'camera_parameters,width,height,is_pano'
        ),
    }
    response = request_with_retries(IMAGES_URL, params)
    if response is None or response.status_code != 200:
        return []
    images = []
    for item in response.json().get('data', []):
        coords = (item.get('geometry') or {}).get('coordinates')
        if not coords or item.get('compass_angle') is None:
            continue
        images.append(
            {
                'id': str(item['id']),
                'lon': coords[0],
                'lat': coords[1],
                'compass_angle': float(item['compass_angle']),
                'captured_at': item.get('captured_at'),
                'sequence': item.get('sequence'),
                'camera_type': item.get('camera_type'),
                'camera_parameters': item.get('camera_parameters') or [],
                'width': item.get('width'),
                'height': item.get('height'),
                'is_pano': bool(item.get('is_pano')),
            }
        )
    return images


# --------------------------------------------------------------- step 2
def decode_detection_polygons(b64_geometry, width, height):
    """Decode a Mapillary MVT detection into pixel-space polygons (x right, y down)."""
    tile = mapbox_vector_tile.decode(base64.decodebytes(b64_geometry.encode('utf-8')))
    polygons = []
    for layer in tile.values():
        extent = layer.get('extent', 4096)
        sx, sy = width / extent, height / extent
        for feature in layer.get('features', []):
            geom = feature['geometry']
            rings = geom['coordinates'] if geom['type'] == 'Polygon' else []
            if geom['type'] == 'MultiPolygon':
                rings = [ring for poly in geom['coordinates'] for ring in poly]
            for ring in rings[:1]:  # exterior ring only
                polygons.append([(x * sx, height - y * sy) for x, y in ring])
    return polygons


def polygon_area(points):
    """Shoelace area of a pixel polygon."""
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def locate_detection(image, polygon):
    """
    Estimate where a detected vehicle stands on the ground.

    Bearing comes from the polygon's horizontal centre; distance from the
    depression angle of its lowest point (ground contact) and the camera
    height. Returns ``None`` when the geometry is implausible.
    """
    width, height = image['width'], image['height']
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    cx = sum(xs) / len(xs)
    y_bottom = max(ys)
    if polygon_area(polygon) < MIN_AREA_FRACTION * width * height:
        return None

    if image['is_pano'] or image['camera_type'] in ('spherical', 'equirectangular'):
        # Equirectangular: 360 deg across the width, 180 deg down the height,
        # image centre column points along the compass angle.
        rel_bearing = (cx / width - 0.5) * 360.0
        depression = (y_bottom / height - 0.5) * 180.0
    else:
        params = image['camera_parameters']
        focal_px = (params[0] if params else 0.85) * max(width, height)
        rel_bearing = math.degrees(math.atan((cx - width / 2) / focal_px))
        depression = math.degrees(math.atan((y_bottom - height / 2) / focal_px))

    if depression <= 1.0:  # bottom edge above the horizon: no ground contact
        return None
    distance = CAMERA_HEIGHT_M / math.tan(math.radians(depression))
    if not MIN_RANGE_M <= distance <= MAX_RANGE_M:
        return None
    bearing = (image['compass_angle'] + rel_bearing) % 360.0
    lon, lat = destination(image['lon'], image['lat'], bearing, distance)
    return {
        'lon': lon,
        'lat': lat,
        'bearing': round(bearing, 1),
        'distance_m': round(distance, 1),
        'pixel_bbox': [round(min(xs)), round(min(ys)), round(max(xs)), round(y_bottom)],
    }


def fetch_vehicle_detections_worker(image, token, vehicle_values):
    """Return geolocated vehicle observations for one image."""
    if not image['width'] or not image['height']:
        return []
    params = {'access_token': token, 'fields': 'id,value,geometry'}
    response = request_with_retries(DETECTIONS_URL.format(image_id=image['id']), params)
    if response is None or response.status_code != 200:
        return []
    observations = []
    for det in response.json().get('data', []):
        value = det.get('value')
        if value not in vehicle_values or not det.get('geometry'):
            continue
        try:
            polygons = decode_detection_polygons(
                det['geometry'], image['width'], image['height']
            )
        except Exception:  # noqa: BLE001 - skip undecodable geometry
            continue
        for polygon in polygons:
            located = locate_detection(image, polygon)
            if located is None:
                continue
            located.update(
                {
                    'detection_id': str(det['id']),
                    'image_id': image['id'],
                    'sequence': image['sequence'],
                    'captured_at': image['captured_at'],
                    'value': value,
                }
            )
            observations.append(located)
    return observations


# --------------------------------------------------------------- step 3
def cluster_observations(observations, radius_m):
    """Greedy spatial clustering: observations within ``radius_m`` share a vehicle."""
    cell = radius_m / 111_000.0  # ~degrees per metre at the equator (fine for grouping)
    grid = defaultdict(list)
    clusters = []
    for obs in sorted(observations, key=lambda o: o['distance_m']):  # near views first
        gx, gy = math.floor(obs['lon'] / cell), math.floor(obs['lat'] / cell)
        target = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cluster in grid.get((gx + dx, gy + dy), []):
                    if (
                        haversine_m(
                            obs['lon'], obs['lat'], cluster['lon'], cluster['lat']
                        )
                        <= radius_m
                    ):
                        target = cluster
                        break
                if target:
                    break
            if target:
                break
        if target is None:
            target = {'lon': obs['lon'], 'lat': obs['lat'], 'members': []}
            clusters.append(target)
            grid[(gx, gy)].append(target)
        target['members'].append(obs)
        n = len(target['members'])
        target['lon'] += (obs['lon'] - target['lon']) / n  # running mean
        target['lat'] += (obs['lat'] - target['lat']) / n
    return clusters


# --------------------------------------------------------------- step 4
def classify_condition(clusters, token, gemini_key, work_dir):
    """Ask Gemini whether each clustered vehicle is intact, damaged or debris."""
    from PIL import Image
    from rapidtools.models import GenerationConfig, load

    model = load('gemini', api_key=gemini_key, model_id='gemini-3.8-flash')
    work_dir.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, Image.Image] = {}

    def panorama(image_id):
        if image_id not in image_cache:
            meta = request_with_retries(
                IMAGE_URL.format(image_id=image_id),
                {'access_token': token, 'fields': 'thumb_2048_url,width,height'},
            ).json()
            img = Image.open(
                BytesIO(requests.get(meta['thumb_2048_url'], timeout=60).content)
            )
            image_cache[image_id] = (img.convert('RGB'), meta['width'], meta['height'])
        return image_cache[image_id]

    for index, cluster in enumerate(clusters):
        best = cluster['members'][0]  # closest view of the vehicle
        try:
            img, full_w, full_h = panorama(best['image_id'])
            sx, sy = img.size[0] / full_w, img.size[1] / full_h
            x0, y0, x1, y1 = best['pixel_bbox']
            pad_x, pad_y = (x1 - x0) * 0.25 + 20, (y1 - y0) * 0.25 + 20
            crop = img.crop(
                (
                    max(0, int((x0 - pad_x) * sx)),
                    max(0, int((y0 - pad_y) * sy)),
                    min(img.size[0], int((x1 + pad_x) * sx)),
                    min(img.size[1], int((y1 + pad_y) * sy)),
                )
            )
            crop_path = work_dir / f'vehicle_{index:04d}_{best["image_id"]}.jpg'
            crop.save(crop_path, quality=90)
            
            out = model.run_inference(
                crop_path,
                CONDITION_PROMPT,
                config=GenerationConfig(temperature=0.0, max_tokens=5),
            )
            condition = (
                (out.text or 'unclear').strip().lower().split()[0] if out else 'unclear'
            )
            if condition not in ('intact', 'damaged', 'debris'):
                condition = 'unclear'
            cluster['condition'] = condition
            cluster['crop'] = str(crop_path)
        except Exception as exc:  # noqa: BLE001 - keep going on a bad image
            print(f'  Could not classify vehicle {index}: {exc}')
            cluster['condition'] = 'unclear'


# --------------------------------------------------------------- pipeline

token = read_credential(
    MAPILLARY_TOKEN, MAPILLARY_TOKEN_FILE, 'A Mapillary access token (MAPILLARY_TOKEN)'
)

min_lon, min_lat, max_lon, max_lat = BBOX
output = Path(OUTPUT_PATH)
output.parent.mkdir(parents=True, exist_ok=True)
vehicle_values = VEHICLE_VALUES

sub_bboxes = generate_grid(min_lon, min_lat, max_lon, max_lat, MAX_BBOX_SIZE)
print(f'--- STEP 1: Fetching uw-rapid images in {len(sub_bboxes)} chunks ---')
images = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(fetch_images_worker, b, token) for b in sub_bboxes]
    for done, future in enumerate(as_completed(futures), 1):
        for image in future.result():
            images[image['id']] = image
        if done % 100 == 0 or done == len(futures):
            print(
                f'  Progress: {done}/{len(futures)} chunks. '
                f'Images found: {len(images)}'
            )
if not images:
    print('No images found. Exiting.')
images_path = output.with_name(output.stem + '_images.geojson')
with open(images_path, 'w', encoding='utf-8') as f:
    json.dump(
        {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [im['lon'], im['lat']],
                    },
                    'properties': {
                        k: v for k, v in im.items() if k not in ('lon', 'lat')
                    },
                }
                for im in images.values()
            ],
        },
        f,
    )
print(f'Step 1 complete: {len(images)} images saved to {images_path}\n')

print(f'--- STEP 2: Fetching vehicle detections for {len(images)} images ---')
observations = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [
        pool.submit(fetch_vehicle_detections_worker, im, token, vehicle_values)
        for im in images.values()
    ]
    for done, future in enumerate(as_completed(futures), 1):
        observations.extend(future.result())
        if done % 100 == 0 or done == len(futures):
            print(
                f'  Progress: {done}/{len(futures)} images. '
                f'Vehicle observations: {len(observations)}'
            )
if SAVE_OBSERVATIONS:
    obs_path = output.with_name(output.stem + '_observations.geojson')
    with open(obs_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [o['lon'], o['lat']],
                        },
                        'properties': {
                            k: v for k, v in o.items() if k not in ('lon', 'lat')
                        },
                    }
                    for o in observations
                ],
            },
            f,
        )
    print(f'  Per-image observations saved to {obs_path}')

print(f'\n--- STEP 3: Clustering {len(observations)} observations ---')
clusters = [
    c
    for c in cluster_observations(observations, CLUSTER_RADIUS_M)
    if len(c['members']) >= MIN_OBSERVATIONS
]
print(
    f'  {len(clusters)} distinct vehicles (>= {MIN_OBSERVATIONS} observations)'
)

if CLASSIFY_CONDITION:
    print('\n--- STEP 4: Classifying vehicle condition with Gemini ---')
    gemini_key = read_credential(
        GEMINI_API_KEY, GEMINI_API_KEY_FILE, 'A Google API key (GOOGLE_API_KEY)'
    )

    classify_condition(
        clusters, token, gemini_key, output.parent / 'vehicle_crops'
    )

features = []
for index, cluster in enumerate(clusters):
    members = cluster['members']
    features.append(
        {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [cluster['lon'], cluster['lat']],
            },
            'properties': {
                'id': f'vehicle_{index:04d}',
                'n_observations': len(members),
                'values': sorted({m['value'] for m in members}),
                'image_ids': sorted({m['image_id'] for m in members}),
                'sequences': sorted(
                    {m['sequence'] for m in members if m['sequence']}
                ),
                'min_distance_m': min(m['distance_m'] for m in members),
                'first_captured_at': min(
                    m['captured_at'] for m in members if m['captured_at']
                ),
                'condition': cluster.get('condition'),
                'crop': cluster.get('crop'),
                'position_method': 'ground-plane estimate from panorama',
            },
        }
    )
with open(output, 'w', encoding='utf-8') as f:
    json.dump({'type': 'FeatureCollection', 'features': features}, f, indent=2)
print(f'\nSaved {len(features)} vehicle locations to {output}')
if CLASSIFY_CONDITION:
    counts = defaultdict(int)
    for c in clusters:
        counts[c.get('condition')] += 1
    print('Condition summary:', dict(counts))
