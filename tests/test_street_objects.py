"""Tests for street-level object discovery and cropping on Mapillary imagery."""

import base64
import logging
import math
import threading
from io import BytesIO

import mapbox_vector_tile
import numpy as np
import pytest
from PIL import Image

from rapidtools.core import (
    BoundingBox,
    ImageAsset,
    ImageCollection,
    Observation,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.models.base import ModelOutput
from rapidtools.processing import (
    MapillaryFeatureExtractor,
    MapillaryObjectImageExtractor,
    Pipeline,
)
from rapidtools.processing.street_localization import destination_point, haversine_m
from rapidtools.processing.street_objects import (
    VEHICLE_LABELS,
    mapillary_vocabulary,
    resolve_classes,
)

CAMERA_HEIGHT = 2.4
ROAD_LON, ROAD_LAT = -117.45, 47.70
CAR = 'object--vehicle--car'
WIDTH, HEIGHT = 2048, 1024


# ==========================================
# Synthetic survey
# ==========================================
def _b64_box(x0, y0, x1, y1):
    """Encode a normalised box as the base64 MVT Mapillary uses (tile y up)."""
    extent = 4096
    pts = [(x0, 1 - y0), (x1, 1 - y0), (x1, 1 - y1), (x0, 1 - y1), (x0, 1 - y0)]
    wkt = (
        'POLYGON(('
        + ', '.join(f'{x * extent:.0f} {y * extent:.0f}' for x, y in pts)
        + '))'
    )
    layer = {'name': 'mpy-or', 'features': [{'geometry': wkt, 'properties': {}}]}
    return base64.encodebytes(mapbox_vector_tile.encode([layer])).decode()


def _box_for(camera, compass, target, width_frac=0.05):
    """Where a ground object at ``target`` appears in a pano taken at ``camera``."""
    dist = haversine_m(*camera, *target)
    project_x = math.degrees(
        math.atan2(
            math.radians(target[0] - camera[0]) * math.cos(math.radians(camera[1])),
            math.radians(target[1] - camera[1]),
        )
    )
    rel = (project_x - compass + 180) % 360 - 180
    cx = 0.5 + rel / 360.0
    depression = math.degrees(math.atan(CAMERA_HEIGHT / dist))
    y_bottom = 0.5 + depression / 180.0
    y_top = y_bottom - 0.06
    return (cx - width_frac / 2, y_top, cx + width_frac / 2, y_bottom)


def _frame(
    image_id, camera, compass, detections, sequence='seq1', captured='2025-08-20'
):
    return ImageAsset(
        id=image_id,
        path=f'/virtual/{image_id}.jpg',
        allow_missing_file=True,
        properties={
            'computed_geometry': {'type': 'Point', 'coordinates': list(camera)},
            'computed_compass_angle': compass,
            'is_pano': True,
            'camera_type': 'spherical',
            'width': WIDTH,
            'height': HEIGHT,
            'sequence': sequence,
            'capture_date': captured,
            'thumb_1024_url': f'https://img/{image_id}_1024.jpg',
            'thumb_2048_url': f'https://img/{image_id}_2048.jpg',
            'detections': {'data': detections},
        },
    )


EGO_BOX = (0.40, 0.80, 0.60, 1.00)  # the survey vehicle: same place in every frame
# True car position: 6 m north of the road, level with the second frame.
CAR_POSITION = destination_point(*destination_point(ROAD_LON, ROAD_LAT, 90, 8), 0, 6)


def _survey(n_frames=5, spacing_m=4.0, with_ego=True, with_car=True):
    """Camera driving east along the road, one frame every ``spacing_m``."""
    frames = []
    for i in range(n_frames):
        camera = destination_point(ROAD_LON, ROAD_LAT, 90, spacing_m * i)
        detections = []
        if with_ego:
            detections.append({'value': CAR, 'geometry': _b64_box(*EGO_BOX)})
        if with_car:
            detections.append(
                {
                    'value': CAR,
                    'geometry': _b64_box(*_box_for(camera, 90.0, CAR_POSITION)),
                }
            )
        frames.append(
            _frame(f'img{i}', camera, 90.0, detections, captured=f'2025-08-2{i}')
        )
    return ImageCollection(frames)


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, image_bytes):
        self.image_bytes = image_bytes
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return FakeResponse(self.image_bytes)


class FakeClient:
    """Stands in for MapillaryClient: canned images, detections and pixels."""

    def __init__(self, images, image_size=(WIDTH, HEIGHT)):
        self.images = images
        self.calls = []
        buf = BytesIO()
        img = Image.new('RGB', image_size, (40, 120, 40))
        # A bright block where the (non-ego) car sits in frame 2, for crop checks:
        img.paste(
            (250, 250, 250),
            (
                int(0.62 * WIDTH),
                int(0.55 * HEIGHT),
                int(0.70 * WIDTH),
                int(0.63 * HEIGHT),
            ),
        )
        img.save(buf, format='JPEG')
        self.session = FakeSession(buf.getvalue())
        self.downloaded = []

    def fetch_images_in_bbox(self, bbox, **kwargs):
        """Tile-level listing: positions, heading, date and sequence only."""
        self.calls.append(('bbox', bbox, kwargs))
        listed = []
        for image in self.images:
            p = image.properties
            listed.append(
                ImageAsset(
                    id=image.id,
                    path=image.path,
                    allow_missing_file=True,
                    properties={
                        'longitude': p['computed_geometry']['coordinates'][0],
                        'latitude': p['computed_geometry']['coordinates'][1],
                        'compass_angle': p['computed_compass_angle'],
                        'is_pano': p['is_pano'],
                        'capture_date': p['capture_date'],
                        'sequence': p['sequence'],
                    },
                )
            )
        return ImageCollection(listed)

    def fetch_images_by_ids(self, image_ids, fields, **kwargs):
        """Per-image metadata, only for the requested IDs."""
        self.calls.append(('by_ids', list(image_ids), fields))
        wanted = set(image_ids)
        return ImageCollection([img for img in self.images if img.id in wanted])

    def fetch_detections(self, image_id, values=None):
        self.calls.append(('detections', image_id))
        return []

    def get_image_url(self, image_id, size='2048'):
        return f'https://img/{image_id}_{size}.jpg'

    def _download_image(self, url, destination):
        self.downloaded.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.session.image_bytes)
        return True

    decode_detection_polygons = staticmethod(
        __import__(
            'rapidtools.data_sources', fromlist=['MapillaryClient']
        ).MapillaryClient.decode_detection_polygons
    )


@pytest.fixture
def region():
    return BoundingBox(
        ROAD_LON - 0.001, ROAD_LAT - 0.001, ROAD_LON + 0.002, ROAD_LAT + 0.001
    )


# ==========================================
# Class resolution
# ==========================================
def test_resolve_classes_aliases_and_labels():
    resolved, unresolved = resolve_classes(
        ['vehicles', 'Cars', 'utility poles', 'object--fire-hydrant', 'debris pile']
    )
    assert resolved['vehicles'] == VEHICLE_LABELS
    assert resolved['Cars'] == ('object--vehicle--car',)
    assert resolved['utility poles'] == ('object--support--utility-pole',)
    assert resolved['object--fire-hydrant'] == ('object--fire-hydrant',)
    assert unresolved == ['debris pile']
    assert all(label in mapillary_vocabulary() for label in VEHICLE_LABELS)


def test_vehicles_means_motor_vehicles_and_trailers_are_their_own_class():
    # A trailer, caravan, boat or jet ski is not a vehicle for a damage survey.
    assert 'object--vehicle--trailer' not in VEHICLE_LABELS
    assert 'object--vehicle--caravan' not in VEHICLE_LABELS
    assert 'object--vehicle--other-vehicle' not in VEHICLE_LABELS
    assert 'object--vehicle--motorcycle' in VEHICLE_LABELS
    resolved, unresolved = resolve_classes(['trailers', 'boats', 'all vehicles'])
    assert unresolved == []
    assert resolved['trailers'] == (
        'object--vehicle--trailer',
        'object--vehicle--caravan',
    )
    assert resolved['boats'] == ('object--vehicle--boat',)
    assert set(VEHICLE_LABELS) < set(resolved['all vehicles'])
    assert 'object--vehicle--trailer' in resolved['all vehicles']


def test_resolve_classes_uses_label_mapper_for_the_rest():
    class Mapper:
        def map_classes(self, classes):
            return ['object--bike-rack', 'not-a-real-label']

    resolved, unresolved = resolve_classes(['bike parking'], label_mapper=Mapper())
    assert resolved == {'bike parking': ('object--bike-rack',)}
    assert unresolved == []


# ==========================================
# Constructor validation
# ==========================================
def test_feature_extractor_validation():
    with pytest.raises(ValueError):
        MapillaryFeatureExtractor(classes=[], access_token='t')
    with pytest.raises(ValueError):
        MapillaryFeatureExtractor(classes=['cars'])  # no token, no client
    with pytest.raises(ValueError):
        MapillaryFeatureExtractor(
            classes=['cars'], access_token='t', detection_source='x'
        )
    with pytest.raises(ValueError):
        MapillaryFeatureExtractor(
            classes=['cars'], access_token='t', min_observations=0
        )
    ext = MapillaryFeatureExtractor(
        classes=['cars'], client=FakeClient(ImageCollection())
    )
    with pytest.raises(ValueError):
        ext()  # no region anywhere


def test_object_image_extractor_validation(tmp_path):
    with pytest.raises(ValueError):
        MapillaryObjectImageExtractor(tmp_path)
    with pytest.raises(ValueError):
        MapillaryObjectImageExtractor(
            tmp_path, access_token='t', max_images_per_asset=0
        )
    with pytest.raises(ValueError):
        MapillaryObjectImageExtractor(tmp_path, access_token='t', outline_shape='oval')
    with pytest.raises(ValueError):
        MapillaryObjectImageExtractor(tmp_path, access_token='t', crop_buffer='lots')


# ==========================================
# Detection end to end (Mapillary source)
# ==========================================
def test_detects_and_triangulates_parked_car_and_drops_ego(region, tmp_path):
    client = FakeClient(_survey())
    extractor = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        start_date='2025-08-01',
        camera_height_m=CAMERA_HEIGHT,
        save_directory=tmp_path,
    )
    result = extractor()

    assert len(result) == 1
    asset = result[0]
    assert asset.id == 'street_cars_00001'
    assert asset.attributes['asset_type'] == 'cars'
    assert asset.attributes['label'] == CAR
    assert asset.attributes['n_observations'] == 5
    assert asset.attributes['n_images'] == 5
    assert asset.attributes['localization'] == 'triangulated'
    assert asset.attributes['position_rms_m'] < 0.5
    assert asset.attributes['first_seen'] == '2025-08-20'
    assert asset.attributes['last_seen'] == '2025-08-24'
    error_m = haversine_m(asset.geometry.x, asset.geometry.y, *CAR_POSITION)
    assert error_m < 0.5

    observations = [Observation.from_dict(o) for o in asset.attributes['observations']]
    assert {o.image_id for o in observations} == {f'img{i}' for i in range(5)}
    assert all(o.has_location and o.source == 'mapillary' for o in observations)

    # The client was asked for RAPID imagery in the window, with detections:
    kind, bbox, kwargs = client.calls[0]
    assert kind == 'bbox' and bbox is region
    assert kwargs['start_date'] == '2025-08-01' and kwargs['filter_rapid_only'] is True
    assert 'fields' not in kwargs  # the tile pass fetches no per-image metadata
    kind, ids, fields = client.calls[1]
    assert kind == 'by_ids' and sorted(ids) == [f'img{i}' for i in range(5)]
    assert 'detections.value' in fields
    # Detections were in the metadata, so no per-image detection calls:
    assert all(c[0] != 'detections' for c in client.calls)


def test_ego_filter_can_be_disabled(region, tmp_path):
    client = FakeClient(_survey(with_car=False))
    ext = MapillaryFeatureExtractor(
        classes=['cars'], client=client, region=region, save_directory=tmp_path
    )
    assert len(ext()) == 0  # ego vehicle removed
    ext = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        ego_filter=False,
        save_directory=tmp_path,
    )
    # Without the filter the ego box sits under the camera, inside min_range,
    # so it still produces no located object:
    assert len(ext(region)) == 0


def test_min_observations_and_frame_spacing(region, tmp_path):
    client = FakeClient(_survey(n_frames=2, spacing_m=1.0))
    ext = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        min_observations=3,
        save_directory=tmp_path,
    )
    assert len(ext()) == 0
    # Frames 1 m apart collapse to one with spacing 10 m: a single sighting.
    ext = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        frame_spacing_m=10.0,
        save_directory=tmp_path,
    )
    result = ext()
    assert len(result) == 1
    assert result[0].attributes['localization'] == 'single_view'
    assert result[0].attributes['n_observations'] == 1


def test_frames_are_fetched_in_batches_and_payloads_dropped(region, tmp_path):
    client = FakeClient(_survey())  # five frames
    ext = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        frame_batch_size=2,
        save_directory=tmp_path,
    )
    result = ext()
    assert len(result) == 1
    batches = [ids for kind, ids, _ in client.calls if kind == 'by_ids']
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sorted(sum(batches, [])) == [f'img{i}' for i in range(5)]
    # Nothing was refetched per image: the batch payloads carried the detections.
    assert not any(kind == 'detections' for kind, *_ in client.calls)


def test_detection_outlines_are_simplified_on_arrival(region, tmp_path):
    client = FakeClient(_survey())
    frame = {
        'id': 'img0',
        'lon': ROAD_LON,
        'lat': ROAD_LAT,
        'compass': 90.0,
        'is_pano': True,
        'sequence': 'seq1',
        'captured_at': '2025-08-20',
        'width': WIDTH,
        'height': HEIGHT,
        'camera_parameters': None,
    }
    # A box whose edges are traced pixel by pixel, as Mapillary masks are:
    jagged = (
        [(0.6 + i / 4096, 0.5 + (i % 2) / 4096) for i in range(200)]
        + [(0.6 + 199 / 4096, 0.6)]
        + [(0.6, 0.6)]
    )
    ext = MapillaryFeatureExtractor(
        classes=['cars'], client=client, region=region, save_directory=tmp_path
    )
    obs = ext._observation(frame, CAR, jagged, 'mapillary')
    assert len(obs.polygon) <= 6
    raw = Observation('img0', CAR, jagged, ROAD_LON, ROAD_LAT, 90.0)
    assert all(abs(a - b) < 0.003 for a, b in zip(obs.bbox, raw.bbox, strict=True))
    # SAM 3 boxes and a zero tolerance are left alone:
    assert len(ext._observation(frame, CAR, jagged, 'sam3').polygon) == len(jagged)
    ext_raw = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        simplify_tolerance=0,
        save_directory=tmp_path,
    )
    assert len(ext_raw._observation(frame, CAR, jagged, 'mapillary').polygon) == len(
        jagged
    )
    with pytest.raises(ValueError):
        MapillaryFeatureExtractor(
            classes=['cars'], access_token='t', frame_batch_size=0
        )
    with pytest.raises(ValueError):
        MapillaryFeatureExtractor(
            classes=['cars'], access_token='t', simplify_tolerance=-1
        )


def test_source_collection_is_kept_and_pipeline_orders_stages(region, tmp_path):
    client = FakeClient(_survey())
    existing = PhysicalAssetCollection(
        [
            PhysicalAsset(
                id='bldg', geometry=__import__('shapely.geometry').geometry.Point(0, 0)
            )
        ]
    )
    detector = MapillaryFeatureExtractor(
        classes=['cars'], client=client, region=region, save_directory=tmp_path
    )
    cropper = MapillaryObjectImageExtractor(
        tmp_path / 'crops', client=client, max_images_per_asset=1
    )
    # Steps added in the wrong order; the pipeline sorts DETECT before EXTRACT.
    result = Pipeline([cropper, detector]).run(existing)
    assert 'bldg' in result and len(result) == 2
    car = result['street_cars_00001']
    assert len(car.image_assets) == 1


def test_unresolved_classes_are_skipped_or_sent_to_sam3(region, tmp_path, caplog):
    client = FakeClient(_survey())
    with caplog.at_level(logging.WARNING):
        ext = MapillaryFeatureExtractor(
            classes=['debris pile'],
            client=client,
            region=region,
            detection_source='mapillary',
            save_directory=tmp_path,
        )
        assert len(ext()) == 0
    assert 'No Mapillary label' in caplog.text


class FakeSAM3:
    """Returns one mask covering a fixed block in every image."""

    def __init__(self):
        self.calls = []

    def run_inference(self, image_inputs, prompt, threshold, mask_threshold, **kw):
        self.calls.append((list(image_inputs), prompt, threshold))
        masks = []
        for _ in image_inputs:
            m = np.zeros((1, 64, 128), bool)
            m[0, 40:48, 70:80] = True  # right of centre, below the horizon
            masks.append(m)
        return ModelOutput(masks=masks, bounding_boxes=None)


def test_sam3_source_detects_classes_without_labels(region, tmp_path):
    client = FakeClient(_survey(n_frames=3, with_car=False, with_ego=False))
    sam = FakeSAM3()
    ext = MapillaryFeatureExtractor(
        classes=['debris pile'],
        client=client,
        region=region,
        detection_source='auto',
        sam3_model=sam,
        sam3_batch_size=2,
        save_directory=tmp_path,
        ego_filter=False,
    )
    result = ext()
    assert [c[1] for c in sam.calls] == [
        'debris pile',
        'debris pile',
    ]  # 3 frames in batches of 2
    assert len(client.downloaded) == 3
    assert len(result) >= 1
    assert all(a.attributes['source'] == 'sam3' for a in result)
    assert all(a.attributes['asset_type'] == 'debris_pile' for a in result)


def test_cancellation_stops_detection(region, tmp_path):
    client = FakeClient(_survey())
    stop = threading.Event()
    stop.set()
    ext = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        cancel_event=stop,
        save_directory=tmp_path,
    )
    with pytest.raises(OperationCancelled):
        ext()


# ==========================================
# Cropping
# ==========================================
@pytest.fixture
def detected(region, tmp_path):
    client = FakeClient(_survey())
    ext = MapillaryFeatureExtractor(
        classes=['cars'], client=client, region=region, save_directory=tmp_path
    )
    return client, ext()


def test_crops_closest_views_with_outline(detected, tmp_path):
    client, collection = detected
    cropper = MapillaryObjectImageExtractor(
        tmp_path / 'crops',
        client=client,
        max_images_per_asset=2,
        image_size='1024',
        min_crop_px=200,
        overlay_asset_outline=True,
        outline_shape='corners',
        outline_color='red',
    )
    result = cropper(collection)
    car = result[0]
    assert len(car.image_assets) == 2
    # The closest two frames are the ones level with the car (img2) and its neighbours:
    ranges = [img.properties['range_m'] for img in car.image_assets]
    assert ranges == sorted(ranges) and ranges[0] < 7
    for img in car.image_assets:
        assert img.path.is_file()
        arr = np.asarray(Image.open(img.path).convert('RGB')).astype(int)
        assert min(arr.shape[:2]) >= 200
        red = (arr[..., 0] > 150) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
        assert red.sum() > 0
        assert img.properties['label'] == CAR
        assert img.properties['source_image_size'] == [WIDTH, HEIGHT]
        assert len(img.properties['crop_box']) == 4
    # 1024 thumbnails were requested:
    assert all('_1024.jpg' in url for url in client.session.urls)
    # Each source image is downloaded once even if several assets share it:
    assert len(client.session.urls) == len(set(client.session.urls))


def test_assets_sharing_an_image_are_cropped_from_one_download(detected, tmp_path):
    client, collection = detected
    car = collection[0]
    twin = PhysicalAsset(
        id='street_cars_00002',
        geometry=car.geometry,
        attributes=dict(car.attributes),
    )
    collection.add(twin)
    client.session.urls.clear()
    cropper = MapillaryObjectImageExtractor(
        tmp_path / 'crops', client=client, max_images_per_asset=2, image_size='1024'
    )
    cropper(collection)
    assert len(car.image_assets) == 2 and len(twin.image_assets) == 2
    # Two assets, two views each, but only two source downloads:
    assert len(client.session.urls) == 2
    assert len(set(client.session.urls)) == 2
    # Crops are attached closest view first regardless of download order:
    for asset in (car, twin):
        ranges = [img.properties['range_m'] for img in asset.image_assets]
        assert ranges == sorted(ranges)
    assert not hasattr(cropper, '_image_cache')


def test_crop_handles_seam_straddling_detection(tmp_path):
    client = FakeClient(ImageCollection())
    cropper = MapillaryObjectImageExtractor(
        tmp_path, client=client, min_crop_px=50, crop_buffer=0
    )
    obs = Observation(
        image_id='x',
        label=CAR,
        polygon=[(0.97, 0.5), (0.03, 0.5), (0.03, 0.6), (0.97, 0.6)],
        camera_lon=0,
        camera_lat=0,
        compass_angle=0,
        is_pano=True,
    )
    image = Image.new('RGB', (400, 200))
    crop, polygon, box, mpp = cropper._crop(image, obs)
    # The object spans 6% of the width: after the roll the crop is ~24 px wide,
    # grown to the 50 px minimum, not the whole panorama:
    assert crop.width <= 60 and crop.height == 50
    assert mpp == 0.0  # no range known


def test_select_observations_prefers_close_distinct_images():
    def obs(i, rng, conf=None):
        o = Observation(str(i), CAR, [(0, 0), (1, 0), (1, 1)], 0, 0, 0)
        o.range_m, o.confidence = rng, conf
        return o

    chosen = MapillaryObjectImageExtractor.select_observations(
        [obs(1, 9.0), obs(2, 4.0, 0.2), obs(2, 4.0, 0.9), obs(3, 6.0), obs(4, None)], 3
    )
    assert [o.image_id for o in chosen] == ['2', '3', '1']
    assert chosen[0].confidence == 0.9


def test_cropper_skips_assets_without_observations(tmp_path, caplog):
    client = FakeClient(ImageCollection())
    cropper = MapillaryObjectImageExtractor(tmp_path, client=client)
    col = PhysicalAssetCollection(
        [
            PhysicalAsset(
                id='a', geometry=__import__('shapely.geometry').geometry.Point(0, 0)
            )
        ]
    )
    with caplog.at_level(logging.WARNING):
        assert cropper(col) is col
    assert 'No assets with street-level observations' in caplog.text


# ==========================================
# Localisation methods, map features and re-identification
# ==========================================
def test_feature_extractor_rejects_unknown_localization_method():
    with pytest.raises(ValueError):
        MapillaryFeatureExtractor(
            classes=['cars'], access_token='t', localization_method='magic'
        )


@pytest.mark.parametrize(
    'method,expected',
    [('tracks', 'triangulated'), ('voting', 'voted'), ('cluster', 'triangulated')],
)
def test_localization_methods_agree_on_the_parked_car(
    region, tmp_path, method, expected
):
    client = FakeClient(_survey())
    extractor = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        camera_height_m=CAMERA_HEIGHT,
        localization_method=method,
        save_directory=tmp_path,
    )
    result = extractor()
    assert len(result) == 1
    asset = result[0]
    assert asset.attributes['localization'] == expected
    assert asset.attributes['n_observations'] == 5
    tolerance = 1.5 if method == 'voting' else 0.5
    assert haversine_m(asset.geometry.x, asset.geometry.y, *CAR_POSITION) < tolerance
    if method != 'cluster':
        assert 'position_sigma_m' in asset.attributes
        assert asset.attributes['n_images'] == 5


class MapFeatureClient(FakeClient):
    """Fake client that also serves Mapillary map features."""

    def __init__(self, images, features):
        super().__init__(images)
        self.features = features

    def fetch_map_features(self, bbox, object_values, **kwargs):
        self.calls.append(('map_features', bbox, list(object_values), kwargs))
        return [f for f in self.features if f['object_value'] in object_values]


POLE = {
    'id': '2519029251914551',
    'object_value': 'object--support--utility-pole',
    'lon': ROAD_LON + 0.0005,
    'lat': ROAD_LAT + 0.0003,
    'aligned_direction': 69.79,
    'first_seen_at': '2025-08-27T03:59:39+0000',
    'last_seen_at': '2025-09-15T23:37:17+0000',
    'image_ids': ['111', '222'],
}


def test_static_classes_come_from_map_features(region, tmp_path):
    client = MapFeatureClient(_survey(), [POLE])
    extractor = MapillaryFeatureExtractor(
        classes=['utility poles', 'cars'],
        client=client,
        region=region,
        camera_height_m=CAMERA_HEIGHT,
        start_date='2025-08-01',
        end_date='2025-09-30',
        save_directory=tmp_path,
    )
    result = extractor()
    kinds = [c[0] for c in client.calls]
    assert kinds[0] == 'map_features'
    _, bbox, values, kwargs = client.calls[0]
    assert bbox is region and values == ['object--support--utility-pole']
    assert kwargs == {'start_date': '2025-08-01', 'end_date': '2025-09-30'}
    assert len(result) == 2
    pole = result['street_utility_poles_00001']
    assert pole.attributes['localization'] == 'map_feature'
    assert pole.attributes['source'] == 'mapillary_map_features'
    assert pole.attributes['label'] == 'object--support--utility-pole'
    assert pole.attributes['n_images'] == 2
    assert pole.attributes['aligned_direction'] == 69.8
    assert pole.attributes['first_seen'] == '2025-08-27T03:59:39+0000'
    assert pole.geometry.x == pytest.approx(POLE['lon'])
    # Cars still come from the per-image detections:
    car = result['street_cars_00001']
    assert car.attributes['localization'] == 'triangulated'
    # Objects without observations are skipped by the image extractor:
    assert not pole.attributes.get('observations')


def test_map_features_only_mode_skips_other_classes(region, tmp_path, caplog):
    client = MapFeatureClient(_survey(), [POLE])
    extractor = MapillaryFeatureExtractor(
        classes=['utility poles', 'cars'],
        client=client,
        region=region,
        detection_source='map_features',
        save_directory=tmp_path,
    )
    with caplog.at_level('WARNING'):
        result = extractor()
    assert len(result) == 1
    assert result[0].attributes['localization'] == 'map_feature'
    assert all(c[0] == 'map_features' for c in client.calls)  # no image listing
    assert "'cars'" in caplog.text


def test_detections_mode_keeps_poles_on_the_image_route(region, tmp_path):
    client = MapFeatureClient(_survey(), [POLE])
    extractor = MapillaryFeatureExtractor(
        classes=['utility poles'],
        client=client,
        region=region,
        detection_source='mapillary',
        save_directory=tmp_path,
    )
    extractor()
    assert all(c[0] != 'map_features' for c in client.calls)


def _second_pass(dx_m, sequence='seq2', prefix='p2_'):
    """
    The same street driven again. The car is seen ``dx_m`` further east than
    in the first pass, as a GPS offset between surveys would make it appear.
    """
    shifted = destination_point(*CAR_POSITION, 90, dx_m)
    frames = []
    for i in range(5):
        camera = destination_point(ROAD_LON, ROAD_LAT, 90, 4.0 * i)
        detections = [
            {'value': CAR, 'geometry': _b64_box(*_box_for(camera, 90.0, shifted))}
        ]
        frames.append(
            _frame(
                f'{prefix}img{i}',
                camera,
                90.0,
                detections,
                sequence=sequence,
                captured=f'2025-09-0{i + 1}',
            )
        )
    return frames


class SameLookEmbedder:
    """Every crop embeds to the same vector: everything looks alike."""

    def __init__(self):
        self.calls = 0

    def embed(self, images):
        self.calls += 1
        import numpy as np

        return np.ones((len(images), 4), dtype='float32') / 2.0


class DifferentLookEmbedder:
    def embed(self, images):
        import numpy as np

        return np.eye(len(images), 4, dtype='float32')


@pytest.mark.parametrize('alike,expected_objects', [(True, 1), (False, 2)])
def test_reid_merges_duplicates_across_passes(
    region, tmp_path, alike, expected_objects
):
    first = list(_survey(with_ego=False))
    # The second pass places the car 4 m away (a GPS offset between surveys):
    # beyond the geometric merge gate, inside the re-identification distance.
    frames = first + _second_pass(dx_m=4.0)
    client = FakeClient(ImageCollection(frames))
    embedder = SameLookEmbedder() if alike else DifferentLookEmbedder()
    without = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        camera_height_m=CAMERA_HEIGHT,
        save_directory=tmp_path,
    )()
    assert len(without) == 2  # two passes, two objects before re-identification
    with_reid = MapillaryFeatureExtractor(
        classes=['cars'],
        client=client,
        region=region,
        camera_height_m=CAMERA_HEIGHT,
        reid=True,
        reid_embedder=embedder,
        reid_max_distance_m=8.0,
        save_directory=tmp_path,
    )()
    assert len(with_reid) == expected_objects
    if alike:
        merged = with_reid[0]
        assert merged.attributes['reid_merged'] == 2
        assert merged.attributes['n_observations'] == 10
        assert sorted(merged.attributes['sequence_ids']) == ['seq1', 'seq2']
        assert embedder.calls >= 1
        assert client.downloaded  # one thumbnail per compared object


def test_min_area_fraction_is_deprecated_but_honoured():
    with pytest.warns(DeprecationWarning):
        ext = MapillaryFeatureExtractor(
            classes=['cars'], access_token='t', min_area_fraction=0.0004
        )
    assert ext.min_area_fraction == 0.0004
    ext = MapillaryFeatureExtractor(classes=['cars'], access_token='t')
    assert ext.min_area_fraction < 1e-4
    assert ext.max_range_m == 60.0
