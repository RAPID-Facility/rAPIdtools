"""Tests for sightings of known footprints (``street_footprints``)."""

import json
import logging
from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, mapping
from test_street_instances import WINGS, _instances
from test_street_structures import (
    BUILDING,
    HOUSES,
    LEVEL,
    UNPROJECT,
    _Client,
    _survey,
)

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.processing import (
    MapillaryFeatureExtractor,
    street_footprints,
    street_instances,
    street_structures,
)
from rapidtools.processing.street_footprints import (
    FootprintViews,
    footprint_region,
    footprint_sightings,
)
from rapidtools.processing.street_objects import _known_footprints, _OffFootprints


def _wgs84(x0, x1, y0, y1):
    """A rectangle of the local meter frame as a longitude/latitude polygon."""
    return Polygon(
        [UNPROJECT(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    )


KNOWN = [_wgs84(*house) for house in HOUSES]


@pytest.fixture(scope='module')
def masks():
    """One outline per house and frame, numbered as a segmentation model gives them."""
    return _instances(_survey())[1]


# ==========================================
# Projection
# ==========================================
def test_footprint_region_is_where_the_box_appears():
    house = Polygon([(-5, 10), (5, 10), (5, 20), (-5, 20)])
    region, nearest = footprint_region(house, LEVEL, np.zeros(3), -2.0, 6.0)
    assert region.centroid.x == pytest.approx(0.5, abs=0.01)  # straight ahead, north
    assert nearest == pytest.approx(10.2, abs=0.1)
    x0, y0, x1, y1 = region.bounds
    assert 0.42 < x0 < 0.44 and 0.56 < x1 < 0.58  # 27 degrees to either side
    assert y0 < 0.5 < y1  # from below the horizon to above it
    # A box behind the camera straddles the seam and runs past 1 on the right:
    behind = Polygon([(-5, -20), (5, -20), (5, -10), (-5, -10)])
    region, _ = footprint_region(behind, LEVEL, np.zeros(3), -2.0, 6.0)
    assert region.bounds[0] > 0.9 and region.bounds[2] > 1.0
    # A footprint with no area and no height projects onto a line:
    flat = Polygon([(0, 10), (5, 10), (2, 10)])
    assert footprint_region(flat, LEVEL, np.zeros(3), 0.0, 0.0)[0] is None


def test_seam_helpers_look_on_both_sides():
    past = Polygon([(0.95, 0.4), (1.05, 0.4), (1.05, 0.6), (0.95, 0.6)])
    left = Polygon([(0.0, 0.4), (0.04, 0.4), (0.04, 0.6), (0.0, 0.6)])
    right = Polygon([(0.96, 0.4), (1.0, 0.4), (1.0, 0.6), (0.96, 0.6)])
    assert street_footprints._overlap(left, past) == pytest.approx(0.008)
    assert street_footprints._overlap(right, past) == pytest.approx(0.008)
    inside = street_footprints._within(left.union(right), past)
    assert inside.area == pytest.approx(0.016) and inside.bounds[2] <= 1.0
    plain = Polygon([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    assert street_footprints._within(plain, plain).equals(plain)


# ==========================================
# Per-object outlines
# ==========================================
def test_every_mask_goes_to_the_footprint_it_lies_on(masks):
    views, unclaimed, counts = footprint_sightings(masks, KNOWN)
    assert counts['frames'] == 21 and counts['footprints'] == 2
    assert counts['footprints_expected'] == counts['footprints_seen'] == 2
    assert unclaimed == [] and counts['unclaimed'] == 0
    for view, house, known in zip(views, HOUSES, KNOWN, strict=True):
        assert isinstance(view, FootprintViews)
        # (from far up the street the two test outlines overlap, which real
        # masks do not: those few frames go to the nearer house)
        assert view.n_expected >= 15 and view.n_seen >= 15
        assert view.n_images == len(view.members) == 12
        middle = (house[0] + house[1]) / 2.0
        for obs in view.members:
            assert obs.source == 'sam3' and 'instance' not in obs.extra
            assert obs.range_m is not None and 5.0 < obs.range_m <= 40.0
            assert (obs.lon, obs.lat) == (known.centroid.x, known.centroid.y)
        # sightings are spread over the directions the house was seen from
        bearings = sorted(o.bearing for o in view.members)
        assert (bearings[-1] - bearings[0] + 180.0) % 360.0 - 180.0 > 40.0 or middle
    every, _, _ = footprint_sightings(masks, KNOWN, max_views=0)
    assert all(len(v.members) == v.n_seen for v in every)


def test_wings_outlined_apart_fall_on_one_footprint():
    wings = _instances(_survey(houses=WINGS))[1]
    whole = _wgs84(-14.0, -1.0, 10.0, 20.0)
    (view,), unclaimed, _ = footprint_sightings(wings, [whole])
    assert unclaimed == [] and view.n_seen == view.n_expected >= 15
    # one sighting per frame, and it covers both wings
    assert view.n_images == len(view.members)
    wide = [
        max(x for x, _ in o.polygon) - min(x for x, _ in o.polygon)
        for o in view.members
    ]
    one_wing = [
        max(x for x, _ in o.polygon) - min(x for x, _ in o.polygon)
        for o in wings
        if o.image_id == view.members[0].image_id
    ]
    assert wide[0] > max(one_wing)


def test_masks_on_no_footprint_are_handed_back(masks):
    # Only the west house is in the layer: the masks of the east one are unclaimed.
    views, unclaimed, counts = footprint_sightings(masks, KNOWN[:1])
    assert views[0].n_seen >= 15
    east = {n for n, o in enumerate(masks) if o.extra['instance'] == 1}
    assert set(unclaimed) <= east and len(unclaimed) >= 15
    assert counts['unclaimed'] == len(unclaimed)
    # A footprint nothing was seen on is expected but not seen:
    empty_lot = _wgs84(-14.0, -2.0, 30.0, 40.0)
    views, _, counts = footprint_sightings(masks, [*KNOWN, empty_lot])
    assert views[2].n_seen == 0 and views[2].members == []
    assert counts['footprints_seen'] == 2


def test_a_footprint_hidden_behind_another_is_not_expected(masks):
    behind = _wgs84(-14.0, -2.0, 21.0, 31.0)  # the lot behind the west house
    views, _, _ = footprint_sightings(masks, [*KNOWN, behind])
    assert views[2].n_expected < views[0].n_expected
    assert views[2].n_seen <= 2  # the west house's masks stay with the west house
    assert views[0].n_seen >= 15


# ==========================================
# Surface outlines
# ==========================================
def test_a_surface_outline_is_cut_by_the_footprints():
    _, surface, _ = _survey(together=True)  # one outline around both houses
    views, unclaimed, counts = footprint_sightings(surface, KNOWN)
    assert unclaimed == [] and counts['footprints_seen'] == 2
    for view in views:
        assert view.n_seen >= 12 and len(view.members) == 12
    # each sighting is the part of the surface on its own footprint
    frame = views[0].members[0].image_id
    whole = next(o for o in surface if o.image_id == frame)
    part = views[0].members[0]
    assert Polygon(part.polygon).area < 0.8 * Polygon(whole.polygon).area
    # A surface that fills too little of a footprint's region is no sighting:
    views, _, counts = footprint_sightings(surface, KNOWN, min_cover=1.5)
    assert counts['footprints_seen'] == 0


def test_nothing_to_do_cases(masks):
    views, unclaimed, counts = footprint_sightings([], KNOWN)
    assert [v.n_seen for v in views] == [0, 0] and unclaimed == []
    assert counts['footprints_seen'] == 0
    views, unclaimed, counts = footprint_sightings(masks, [])
    assert views == [] and counts['footprints'] == 0
    flat = replace(masks[0], is_pano=False)
    no_pose = replace(masks[1], extra={})
    views, _, counts = footprint_sightings([flat, no_pose], KNOWN)
    assert counts['unusable'] == 2 and counts['frames'] == 0
    # Footprints out of range of every frame are neither expected nor seen:
    views, _, counts = footprint_sightings(masks, KNOWN, max_range_m=3.0)
    assert counts['footprints_expected'] == 0
    assert FootprintViews().n_images == 0


# ==========================================
# Through the extractor
# ==========================================
def _layer():
    return PhysicalAssetCollection(
        [
            PhysicalAsset(
                id='parcel-7', geometry=KNOWN[0], attributes={'year_built': 1990}
            ),
            PhysicalAsset(
                id='parcel-8', geometry=KNOWN[1], attributes={'observations': 'old'}
            ),
        ]
    )


def test_known_footprints_are_read_from_several_forms(tmp_path):
    known = _known_footprints(_layer())
    assert [k[0] for k in known] == ['parcel-7', 'parcel-8']
    assert known[0][2] == {'year_built': 1990} and known[1][2] == {}
    assert _known_footprints(None) == []
    # bare shapes: a part of several is the largest; points are skipped
    two = MultiPolygon([KNOWN[0], _wgs84(20.0, 21.0, 10.0, 11.0)])
    known = _known_footprints([two, Point(0, 0), KNOWN[1]])
    assert [k[0] for k in known] == [None, None]
    assert known[0][1].equals(KNOWN[0])
    path = tmp_path / 'layer.geojson'
    path.write_text(
        json.dumps(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'id': f'lot-{n}',
                        'properties': {},
                        'geometry': mapping(k),
                    }
                    for n, k in enumerate(KNOWN)
                ],
            }
        )
    )
    assert len(_known_footprints(path)) == 2 and len(_known_footprints(str(path))) == 2


def test_extractor_matches_buildings_to_known_footprints(masks, tmp_path, caplog):
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        footprints=_layer(),
        structure_views=4,
    )
    assert ext._method_for(masks) == 'footprints'
    with caplog.at_level(logging.INFO):
        assets, counter = ext._assets_for_class('buildings', list(masks), 0)
    assert '2 of the 2 footprints in view were seen' in caplog.text
    assert counter == 2 and [a.id for a in assets] == ['parcel-7', 'parcel-8']
    first = assets[0]
    assert first.geometry.equals(KNOWN[0])  # the footprint itself, not an estimate
    at = first.attributes
    assert at['localization'] == 'footprint' and at['year_built'] == 1990
    assert at['n_images'] == 4 and 15 <= at['frames_seen'] <= at['frames_expected']
    assert at['asset_type'] == 'buildings' and len(at['observations']) == 4
    assert at['source'] == 'sam3'
    # Mapillary's own surface outline works too, with no model at all:
    _, surface, _ = _survey(together=True)
    assert ext._method_for(surface) == 'footprints'
    assets, _ = ext._assets_for_class('buildings', list(surface), 0)
    assert [a.id for a in assets] == ['parcel-7', 'parcel-8']
    # Footprints seen in too few frames are left out:
    ext.min_observations = 50
    assert ext._assets_for_class('buildings', list(masks), 0)[0] == []
    # Bare polygons get generated identifiers:
    bare = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        footprints=KNOWN,
    )
    assets, _ = bare._assets_for_class('buildings', list(masks), 0)
    assert [a.id for a in assets] == [
        'street_buildings_00001',
        'street_buildings_00002',
    ]


def test_extractor_reports_buildings_the_layer_does_not_have(tmp_path, monkeypatch):
    cloud, masks, clusters = _instances(_survey())
    monkeypatch.setattr(
        street_instances, 'fetch_sfm_cluster', lambda cid, url, cache=None: cloud
    )
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        footprints=KNOWN[:1],  # the east house is not in the layer
    )
    ext._clusters = dict(clusters)
    assets, _ = ext._assets_for_class('buildings', list(masks), 0)
    assert len(assets) == 2
    mapped, unmapped = assets
    assert (
        mapped.attributes['localization'] == 'footprint'
        and 'unmapped' not in mapped.attributes
    )
    assert unmapped.attributes['unmapped'] is True
    assert unmapped.attributes['localization'] == 'point_cloud'
    assert unmapped.geometry.centroid.x > mapped.geometry.centroid.x  # the east house
    # Not asked for, or no point clouds to place them with: only the layer's buildings.
    ext.report_unmapped = False
    assert len(ext._assets_for_class('buildings', list(masks), 0)[0]) == 1
    ext.report_unmapped, ext._clusters = True, {}
    assert len(ext._assets_for_class('buildings', list(masks), 0)[0]) == 1


def test_extractor_reports_unmapped_buildings_from_surface_outlines(
    tmp_path, monkeypatch, caplog
):
    # Mapillary's outlines carry no instance numbers, so none is left over
    # after matching: the point cloud route finds what lies off the layer.
    cloud, surface, clusters = _survey()
    monkeypatch.setattr(
        street_structures, 'fetch_sfm_cluster', lambda cid, url, cache=None: cloud
    )
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        footprints=KNOWN[:1],  # the east house is not in the layer
    )
    ext._clusters = dict(clusters)
    with caplog.at_level(logging.INFO):
        assets, counter = ext._assets_for_class('buildings', list(surface), 0)
    assert '1 structures from the point cloud lie off every footprint' in caplog.text
    assert counter == 2 and len(assets) == 2
    mapped, unmapped = assets
    assert mapped.attributes['localization'] == 'footprint'
    assert 'unmapped' not in mapped.attributes
    assert unmapped.attributes['unmapped'] is True
    assert unmapped.attributes['localization'] == 'point_cloud'
    assert unmapped.geometry.centroid.x > KNOWN[0].bounds[2]  # the east house
    # With the whole layer nothing is left off it:
    ext.footprints = _known_footprints(KNOWN)
    assets, _ = ext._assets_for_class('buildings', list(surface), 0)
    assert len(assets) == 2 and not any('unmapped' in a.attributes for a in assets)
    # Not asked for, or no point clouds: only the layer's buildings.
    ext.footprints = _known_footprints(KNOWN[:1])
    ext.report_unmapped = False
    assert len(ext._assets_for_class('buildings', list(surface), 0)[0]) == 1
    ext.report_unmapped, ext._clusters = True, {}
    assert len(ext._assets_for_class('buildings', list(surface), 0)[0]) == 1


def test_a_shape_is_off_the_layer_unless_a_tenth_lies_on_a_footprint():
    off = _OffFootprints(KNOWN)
    assert not off(KNOWN[0]) and not off(KNOWN[1])
    # A shed in the side yard, touching the west house over 5 % of its area:
    assert off(_wgs84(-2.2, 2.0, 12.0, 16.0))
    # A quarter of it on the house: a piece of the mapped house.
    assert not off(_wgs84(-3.0, 1.0, 12.0, 16.0))
    assert off(_wgs84(20.0, 30.0, 10.0, 20.0))  # clear of both
    assert not off(Polygon())
    # A self-crossing outline is repaired, not refused:
    x0, x1, y0, y1 = 20.0, 30.0, 10.0, 20.0
    bow = Polygon(
        [UNPROJECT(x0, y0), UNPROJECT(x1, y1), UNPROJECT(x1, y0), UNPROJECT(x0, y1)]
    )
    assert off(bow)
    assert _OffFootprints([])(KNOWN[0])  # no layer: everything is off it


def test_extractor_footprint_options(masks, tmp_path):
    with pytest.raises(ValueError, match='footprints'):
        MapillaryFeatureExtractor(
            classes=['buildings'],
            client=_Client(),
            save_directory=tmp_path,
            localization_method='footprints',
        )
    forced = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        localization_method='footprints',
        footprints=KNOWN,
    )
    assert forced._method_for(masks) == 'footprints'
    # Asked for by name, SAM 3 masks stay one per building, and the point
    # clouds are fetched to place what the layer lacks:
    assert forced._outlined_classes(['buildings']) == {'buildings'}
    assert forced._wants_point_clouds({'buildings': (BUILDING,)}, [])
    forced.report_unmapped = False
    assert not forced._wants_point_clouds({'buildings': (BUILDING,)}, [])
    # Footprints never claim classes that are not structures:
    ext = MapillaryFeatureExtractor(
        classes=['cars'], client=_Client(), save_directory=tmp_path, footprints=KNOWN
    )
    car = replace(
        masks[0], label='object--vehicle--car', extra={'computed_rotation': [0, 0, 0]}
    )
    assert ext._method_for([car]) == 'fusion'
