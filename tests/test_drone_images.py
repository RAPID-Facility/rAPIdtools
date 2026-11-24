"""Tests for the drone photo pose reader (rapidtools.data_sources.drone_images)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from rapidtools.data_sources import (
    DroneCamera,
    read_drone_image,
    read_drone_images,
    read_pose_table,
    write_pose_metadata,
)
from rapidtools.data_sources.drone_images import (
    _float,
    _insert_xmp,
    dji_xmp_packet,
    read_xmp_tags,
    rotation_from_angles,
)
from rapidtools.processing.street_localization import local_projection


def make_camera(**overrides) -> DroneCamera:
    values = {
        'image_id': 'DJI_0001',
        'lon': -118.1412345,
        'lat': 34.1861234,
        'elevation_m': 80.0,
        'yaw_deg': 271.3,
        'pitch_deg': -47.25,
        'width': 640,
        'height': 480,
        'focal_px': 500.0,
    }
    values.update(overrides)
    return DroneCamera(**values)


# ----------------------------------------------------------------- geometry
def test_rotation_is_orthonormal_and_right_handed():
    for yaw, pitch, roll in [(0, 0, 0), (37, -52, 0), (200, -90, 0), (310, -30, 12)]:
        rotation = rotation_from_angles(yaw, pitch, roll)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0)


def test_lens_axis_follows_yaw_and_pitch():
    forward = rotation_from_angles(90.0, -45.0)[2]
    assert np.allclose(forward, [math.sqrt(0.5), 0.0, -math.sqrt(0.5)])
    # A level camera looking north has image-right to the east, image-down down.
    right, down, _ = rotation_from_angles(0.0, 0.0)
    assert np.allclose(right, [1, 0, 0]) and np.allclose(down, [0, 0, -1])


def test_nadir_projection_puts_north_up():
    camera = make_camera(yaw_deg=0.0, pitch_deg=-90.0, lon=0.0, lat=0.0)
    project, _ = local_projection(0.0, 0.0)
    pixels, depth = camera.project_points([[0, 0, 0], [10, 0, 0], [0, 10, 0]], project)
    assert np.allclose(depth, 80.0)
    assert np.allclose(pixels[0], [320, 240])
    assert pixels[1][0] > 320 and np.isclose(pixels[1][1], 240)  # east is right
    assert pixels[2][1] < 240 and np.isclose(pixels[2][0], 320)  # north is up


def test_oblique_view_centre_hits_the_ground_ahead():
    camera = make_camera(yaw_deg=90.0, pitch_deg=-45.0, lon=0.0, lat=0.0)
    lon, lat = camera.pixel_to_ground(320, 240)
    project, _ = local_projection(0.0, 0.0)
    x, y = project(lon, lat)
    assert x == pytest.approx(80.0, abs=0.01) and y == pytest.approx(0.0, abs=0.01)
    # ... and that ground point projects back to the image centre.
    pixels, depth = camera.project_lonlat([(lon, lat, 0.0)])
    assert np.allclose(pixels[0], [320, 240], atol=1e-6)
    assert depth[0] == pytest.approx(80.0 * math.sqrt(2), rel=1e-6)


def test_roll_rotates_the_image():
    camera = make_camera(yaw_deg=0.0, pitch_deg=-90.0, roll_deg=90.0, lon=0.0, lat=0.0)
    project, _ = local_projection(0.0, 0.0)
    pixels, _ = camera.project_points([[10, 0, 0]], project)
    # Rolled a quarter turn clockwise, what was to the right appears at the top.
    assert np.isclose(pixels[0][0], 320) and pixels[0][1] < 240


def test_pixel_to_ground_none_above_horizon_and_below_ground():
    camera = make_camera(pitch_deg=10.0)
    assert camera.pixel_to_ground(320, 240) is None
    assert make_camera(pitch_deg=-45.0).pixel_to_ground(320, 240, 500.0) is None


def test_ground_footprint_caps_rays_that_miss_the_ground():
    project, _ = local_projection(0.0, 0.0)
    camera = make_camera(yaw_deg=0.0, pitch_deg=-20.0, lon=0.0, lat=0.0)
    corners = camera.ground_footprint(max_range_m=300.0)
    ranges = [math.hypot(*project(lon, lat)) for lon, lat in corners]
    assert len(corners) == 4 and max(ranges) == pytest.approx(300.0, rel=1e-3)
    assert min(ranges) < 300.0


def test_fields_of_view_and_validation():
    camera = make_camera(width=4000, height=3000, focal_px=3000.0)
    assert camera.hfov_deg == pytest.approx(67.38, abs=0.01)
    assert camera.vfov_deg == pytest.approx(53.13, abs=0.01)
    assert (camera.cx, camera.cy) == (2000.0, 1500.0)
    with pytest.raises(ValueError, match='focal_px'):
        make_camera(focal_px=0)
    with pytest.raises(ValueError, match='width'):
        make_camera(width=0)


def test_dict_round_trip(tmp_path):
    camera = make_camera(path=tmp_path / 'a.jpg', properties={'k': 1})
    again = DroneCamera.from_dict(json.loads(json.dumps(camera.to_dict())))
    assert again == camera


# ------------------------------------------------------------------ parsing
def test_float_parses_camera_spellings():
    assert _float('+12.50') == 12.5
    assert _float(b'-3.5') == -3.5
    assert _float('7/2') == 3.5
    assert _float('1/0') is None
    assert _float('') is None and _float(None) is None and _float('abc') is None
    assert _float(float('nan')) is None


def test_write_then_read_round_trip(tmp_path):
    camera = make_camera(
        relative_altitude_m=80.0,
        absolute_altitude_m=445.3,
        captured_at='2026:11:12 10:00:00',
    )
    path = write_pose_metadata(
        Image.new('RGB', (640, 480), 'gray'), tmp_path / 'DJI_0001.JPG', camera
    )
    read = read_drone_image(path)
    assert read.image_id == 'DJI_0001'
    assert read.lon == pytest.approx(camera.lon, abs=1e-7)
    assert read.lat == pytest.approx(camera.lat, abs=1e-7)
    assert read.elevation_m == 80.0  # relative altitude preferred
    assert read.yaw_deg == pytest.approx(271.3, abs=0.01)
    assert read.pitch_deg == pytest.approx(-47.25, abs=0.01)
    assert read.focal_px == pytest.approx(500.0)
    assert read.captured_at == '2026:11:12 10:00:00'
    assert read.properties == {
        'pose_source': 'gimbal',
        'focal_source': 'xmp_calibrated',
    }
    # The take-off elevation places the flight in a real datum.
    assert read_drone_image(path, takeoff_elevation_m=365.0).elevation_m == 445.0
    assert read_drone_image(path, altitude_source='absolute').elevation_m == 445.3
    assert read_drone_image(path, yaw_offset_deg=100.0).yaw_deg == pytest.approx(11.3)
    with pytest.raises(ValueError, match='altitude_source'):
        read_drone_image(path, altitude_source='agl')


def test_write_accepts_a_path_and_southern_western_positions(tmp_path):
    source = tmp_path / 'src.png'
    Image.new('RGB', (64, 48)).save(source)
    camera = make_camera(lon=-70.5, lat=-33.4, width=64, height=48, focal_px=60.0)
    read = read_drone_image(write_pose_metadata(source, tmp_path / 'out.jpg', camera))
    assert read.lon == pytest.approx(-70.5) and read.lat == pytest.approx(-33.4)


def _exif_only_jpeg(path, with_altitude=True, with_direction=True, film=24):
    """A JPEG with EXIF GPS tags and no XMP, like a non-DJI camera writes."""
    camera = make_camera(lon=10.25, lat=45.5, absolute_altitude_m=312.0)
    full = write_pose_metadata(Image.new('RGB', (640, 480)), path, camera)
    with Image.open(full) as image:
        exif = image.getexif()
        gps = exif.get_ifd(0x8825)
        if not with_altitude:
            del gps[6]
        if not with_direction:
            del gps[17]
        exif[0x8825] = gps
        if film:
            exif.get_ifd(0x8769)[0xA405] = film
        image.save(path, exif=exif)  # re-saving drops the XMP segment
    return path


def test_exif_only_photo_falls_back_to_gps_tags(tmp_path):
    path = _exif_only_jpeg(tmp_path / 'plain.jpg')
    assert read_xmp_tags(path) == {}
    camera = read_drone_image(path)
    assert camera.lon == pytest.approx(10.25, abs=1e-6)
    assert camera.lat == pytest.approx(45.5, abs=1e-6)
    assert camera.elevation_m == pytest.approx(312.0)  # GPS altitude
    assert camera.pitch_deg == -90.0  # no gimbal block: assumed straight down
    assert camera.yaw_deg == pytest.approx(271.3, abs=0.01)
    assert camera.properties['pose_source'] == 'gps_img_direction'
    assert camera.properties['focal_source'] == 'exif_35mm'
    assert camera.focal_px == pytest.approx(24 * 800 / 43.2666, rel=1e-4)


def test_photo_without_lens_data_uses_default_field_of_view(tmp_path):
    path = _exif_only_jpeg(tmp_path / 'plain.jpg', with_direction=False, film=None)
    camera = read_drone_image(path, default_hfov_deg=90.0)
    assert camera.focal_px == pytest.approx(320.0)
    assert camera.properties == {
        'pose_source': 'assumed_nadir',
        'focal_source': 'default_hfov',
    }


def test_unusable_photos_are_skipped(tmp_path, caplog):
    no_gps = tmp_path / 'no_gps.jpg'
    Image.new('RGB', (32, 32)).save(no_gps)
    assert read_drone_image(no_gps) is None
    assert 'no GPS position' in caplog.text
    no_altitude = _exif_only_jpeg(tmp_path / 'no_alt.jpg', with_altitude=False)
    assert read_drone_image(no_altitude) is None
    assert read_drone_image(no_altitude, altitude_source='relative') is None
    broken = tmp_path / 'broken.jpg'
    broken.write_bytes(b'not an image')
    assert read_drone_image(broken) is None
    assert read_drone_images([no_gps, broken]) == []
    assert 'None of the photos' in caplog.text


def test_xmp_element_spelling_and_flipped_gimbal(tmp_path):
    path = tmp_path / 'elements.jpg'
    Image.new('RGB', (100, 80)).save(path)
    packet = (
        b'<x:xmpmeta><rdf:RDF><rdf:Description>'
        b'<drone-dji:GpsLatitude>34.5</drone-dji:GpsLatitude>'
        b'<drone-dji:GpsLongtitude>-118.5</drone-dji:GpsLongtitude>'
        b'<drone-dji:RelativeAltitude>+50.0</drone-dji:RelativeAltitude>'
        b'<drone-dji:GimbalYawDegree>10.0</drone-dji:GimbalYawDegree>'
        b'<drone-dji:GimbalPitchDegree>-100.0</drone-dji:GimbalPitchDegree>'
        b'</rdf:Description></rdf:RDF></x:xmpmeta>'
    )
    _insert_xmp(path, packet)
    tags = read_xmp_tags(path)
    assert tags['GpsLongtitude'] == '-118.5'  # DJI's historical misspelling
    camera = read_drone_image(path)
    assert (camera.lon, camera.lat, camera.elevation_m) == (-118.5, 34.5, 50.0)
    # A gimbal 10 degrees past nadir looks backwards at -80 degrees.
    assert camera.pitch_deg == pytest.approx(-80.0)
    assert camera.yaw_deg == pytest.approx(190.0)


def test_flight_yaw_is_used_without_gimbal_angles(tmp_path):
    path = tmp_path / 'flight.jpg'
    Image.new('RGB', (100, 80)).save(path)
    _insert_xmp(
        path,
        b'<x:xmpmeta drone-dji:GpsLatitude="1.0" drone-dji:GpsLongitude="2.0" '
        b'drone-dji:AbsoluteAltitude="+99.5" drone-dji:FlightYawDegree="-45.0">'
        b'</x:xmpmeta>',
    )
    camera = read_drone_image(path)
    assert camera.yaw_deg == pytest.approx(315.0) and camera.pitch_deg == -90.0
    assert camera.elevation_m == 99.5
    assert camera.properties['pose_source'] == 'flight_yaw'


def test_insert_xmp_rejects_non_jpeg_and_oversized_packets(tmp_path):
    png = tmp_path / 'a.png'
    Image.new('RGB', (8, 8)).save(png)
    with pytest.raises(ValueError, match='not a JPEG'):
        _insert_xmp(png, b'<x:xmpmeta/>')
    jpeg = tmp_path / 'a.jpg'
    Image.new('RGB', (8, 8)).save(jpeg)
    with pytest.raises(ValueError, match='too large'):
        _insert_xmp(jpeg, b'x' * 70000)


def test_xmp_packet_uses_elevation_when_altitudes_are_unset():
    packet = dji_xmp_packet(make_camera()).decode()
    assert 'drone-dji:RelativeAltitude="+80.000"' in packet
    assert 'drone-dji:AbsoluteAltitude="+80.000"' in packet
    assert 'drone-dji:GimbalYawDegree="-88.70"' in packet  # 271.3 as +-180


# ------------------------------------------------------------- collections
def test_read_folder_sorts_and_recurses(tmp_path):
    (tmp_path / 'sub').mkdir()
    for name in ('b.jpg', 'a.jpg', 'sub/c.jpg'):
        write_pose_metadata(
            Image.new('RGB', (64, 48)),
            tmp_path / name,
            make_camera(width=64, height=48, focal_px=60.0),
        )
    (tmp_path / 'notes.txt').write_text('not a photo')
    assert [c.image_id for c in read_drone_images(tmp_path)] == ['a', 'b', 'c']
    assert [c.image_id for c in read_drone_images(tmp_path, recursive=False)] == [
        'a',
        'b',
    ]
    assert [c.image_id for c in read_drone_images(tmp_path / 'a.jpg')] == ['a']


def test_pose_table_csv(tmp_path, caplog):
    Image.new('RGB', (800, 600)).save(tmp_path / 'IMG_1.jpg')
    Image.new('RGB', (800, 600)).save(tmp_path / 'IMG_2.jpg')
    (tmp_path / 'poses.csv').write_text(
        'Filename,Longitude,Latitude,Altitude,Heading,Pitch,HFOV\n'
        'IMG_1.jpg,-118.1,34.1,60,90,-45,90\n'
        'IMG_2,-118.2,34.2,61,,,\n'  # no extension, no angles
        'IMG_3.jpg,-118.3,34.3,,0,-90,\n'  # no altitude
        'IMG_4.jpg,-118.4,34.4,60,0,-90,\n'  # no file and no size
    )
    cameras = read_pose_table(tmp_path / 'poses.csv', takeoff_elevation_m=100.0)
    assert [c.image_id for c in cameras] == ['IMG_1', 'IMG_2']
    first, second = cameras
    assert (first.yaw_deg, first.pitch_deg, first.elevation_m) == (90.0, -45.0, 160.0)
    assert first.focal_px == pytest.approx(400.0)  # 90 degrees across 800 px
    assert first.path == tmp_path / 'IMG_1.jpg'
    assert (second.yaw_deg, second.pitch_deg) == (0.0, -90.0)
    assert second.path == tmp_path / 'IMG_2.jpg'
    assert 'no pose' in caplog.text and 'no size' in caplog.text
    # The same table through read_drone_images, images in another folder:
    table = tmp_path / 'elsewhere.csv'
    table.write_text((tmp_path / 'poses.csv').read_text())
    assert len(read_drone_images(tmp_path, sidecar=table)) == 2


def test_pose_table_geojson(tmp_path):
    table = tmp_path / 'poses.geojson'
    table.write_text(
        json.dumps(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [8.5, 47.4, 520.0],
                        },
                        'properties': {
                            'image': 'frame_001.jpg',
                            'yaw': 180,
                            'pitch': -60,
                            'roll': 1.5,
                            'focal_px': 1200,
                            'width': 1920,
                            'height': 1080,
                            'timestamp': '2026-05-01T10:00:00Z',
                        },
                    },
                    {'type': 'Feature', 'geometry': None, 'properties': {}},
                ],
            }
        )
    )
    (camera,) = read_pose_table(table)
    assert (camera.lon, camera.lat, camera.elevation_m) == (8.5, 47.4, 520.0)
    assert (camera.yaw_deg, camera.pitch_deg, camera.roll_deg) == (180.0, -60.0, 1.5)
    assert camera.focal_px == 1200 and camera.path is None
    assert camera.captured_at == '2026-05-01T10:00:00Z'


def test_focal_plane_resolution_and_below_sea_level(tmp_path):
    from rapidtools.data_sources.drone_images import _dms_to_degrees, _focal_px

    # 8.8 mm lens on a sensor with 240 pixels per millimetre (unit 4 = mm).
    focal, source = _focal_px(
        {0x920A: 8.8, 0xA20E: 240.0, 0xA210: 4}, {}, 4000, 3000, 70
    )
    assert source == 'exif_focal_plane' and focal == pytest.approx(2112.0)
    # The same sensor described per inch (unit 2, the EXIF default).
    focal, _ = _focal_px({0x920A: 8.8, 0xA20E: 6096.0}, {}, 4000, 3000, 70)
    assert focal == pytest.approx(2112.0)
    assert _dms_to_degrees((10, 30, 0), b'S') == -10.5
    assert _dms_to_degrees((10, 30), 'N') is None
    assert _dms_to_degrees(None, 'N') is None
    # GPSAltitudeRef 1 means below sea level.
    path = _exif_only_jpeg(tmp_path / 'low.jpg')
    with Image.open(path) as image:
        exif = image.getexif()
        gps = exif.get_ifd(0x8825)
        gps[5] = b'\x01'
        exif[0x8825] = gps
        image.save(path, exif=exif)
    assert read_drone_image(path).elevation_m == pytest.approx(-312.0)
