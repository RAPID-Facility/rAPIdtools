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
# 11-27-2026

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, box

from rapidtools.processing.instance_fusion import (
    FusedObject,
    ViewDetection,
    fuse_view_detections,
    split_into_parts,
)

# Two tiles overlapping by half, as a 50% overlap scan produces them:
WINDOWS = {'A': box(0, 0, 50, 50), 'B': box(25, 0, 75, 50)}


def _iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter)


# ------------------------------------------------------------ split_into_parts


def test_split_cuts_narrow_necks_only_when_asked():
    """Two roofs joined by a 0.4 m strip are cut by a 1 m neck width."""
    dumbbell = box(0, 0, 10, 10).union(box(10, 4.8, 12, 5.2)).union(box(12, 0, 22, 10))
    parts = split_into_parts(dumbbell, neck_width=1.0)
    assert [round(p.area) for p in parts] == [100, 100]
    assert len(split_into_parts(dumbbell, neck_width=0.0)) == 1


def test_split_keeps_wide_connections():
    """An L-shaped building is not cut: its wings meet along a full wall."""
    ell = box(0, 0, 10, 10).union(box(10, 0, 16, 4))
    assert len(split_into_parts(ell, neck_width=1.0)) == 1


def test_split_separates_pieces_and_drops_specks():
    """Separate pieces become parts; tiny and relatively small specks go."""
    pieces = MultiPolygon([box(0, 0, 10, 10), box(20, 0, 26, 6), box(40, 0, 40.5, 0.5)])
    parts = split_into_parts(pieces, neck_width=0.0, min_area=1.0)
    assert [round(p.area) for p in parts] == [100, 36]
    # A piece under 10% of the largest one is a speck:
    with_speck = MultiPolygon([box(0, 0, 10, 10), box(20, 0, 23, 3)])
    assert len(split_into_parts(with_speck, neck_width=0.0, min_area=1.0)) == 1
    assert len(split_into_parts(with_speck, 0.0, 1.0, min_fraction=0.0)) == 2


# -------------------------------------------------------- fuse_view_detections


def test_same_object_in_two_views_is_fused():
    """Two views of one roof become one object with both scores averaged."""
    detections = [
        ViewDetection(box(30, 10, 40, 20), 'A', 0.9),
        ViewDetection(box(30.3, 10.1, 40.2, 20.3), 'B', 0.7),
    ]
    (obj,) = fuse_view_detections(detections, WINDOWS)
    assert isinstance(obj, FusedObject)
    assert obj.n_views == 2
    assert obj.score == pytest.approx(0.8)
    assert obj.members == (0, 1)
    assert _iou(obj.geometry, box(30.15, 10.05, 40.1, 20.15)) > 0.95


def test_touching_neighbors_from_one_view_stay_apart():
    """Two instances of the same view are never merged, even if they touch."""
    detections = [
        ViewDetection(box(30, 10, 38, 20), 'A', 0.9),
        ViewDetection(box(38, 10, 46, 20), 'A', 0.9),
    ]
    objects = fuse_view_detections(detections, WINDOWS)
    assert len(objects) == 2


def test_object_cut_by_a_tile_edge_is_whole():
    """A view that saw only part of an object still links to the whole view."""
    whole = box(40, 10, 60, 20)  # crosses x = 50, the edge of window A
    detections = [
        ViewDetection(whole.intersection(WINDOWS['A']), 'A', 0.8),
        ViewDetection(whole, 'B', 0.8),
    ]
    (obj,) = fuse_view_detections(detections, WINDOWS)
    assert obj.n_views == 2
    assert _iou(obj.geometry, whole) > 0.97


def test_group_detection_is_dropped_for_its_parts():
    """A view that merged two houses loses to the views that kept them apart."""
    windows = {k: box(0, 0, 60, 60) for k in 'ABC'}
    house, garage = box(10, 10, 30, 25), box(32, 10, 40, 18)
    detections = [
        ViewDetection(house.union(garage).convex_hull, 'A', 0.8),
        ViewDetection(house, 'B', 0.9),
        ViewDetection(garage, 'B', 0.7),
        ViewDetection(house, 'C', 0.9),
        ViewDetection(garage, 'C', 0.7),
    ]
    objects = fuse_view_detections(detections, windows)
    assert len(objects) == 2
    areas = sorted(round(o.geometry.area) for o in objects)
    assert areas == [round(garage.area), round(house.area)]


def test_group_detection_kept_when_parts_are_poorly_supported():
    """A merged outline most views agree on wins over a one-off split."""
    windows = {k: box(0, 0, 60, 60) for k in 'ABCD'}
    left, right = box(10, 10, 25, 25), box(25, 10, 40, 25)
    merged = left.union(right)
    detections = [
        ViewDetection(merged, 'A', 0.9),
        ViewDetection(merged, 'B', 0.9),
        ViewDetection(merged, 'C', 0.9),
        ViewDetection(left, 'D', 0.6),
        ViewDetection(right, 'D', 0.6),
    ]
    objects = fuse_view_detections(detections, windows)
    assert len(objects) == 1
    assert _iou(objects[0].geometry, merged) > 0.95


def test_consensus_outline_resists_one_stray_view():
    """The median of three views ignores a view that bled onto the lawn."""
    windows = {k: box(0, 0, 60, 60) for k in 'ABC'}
    roof = box(10, 10, 30, 25)
    detections = [
        ViewDetection(roof, 'A', 0.9),
        ViewDetection(box(10.2, 10, 30, 25.1), 'B', 0.9),
        ViewDetection(box(10, 10, 36, 25), 'C', 0.9),  # 6 m of lawn
    ]
    (obj,) = fuse_view_detections(detections, windows)
    assert obj.n_views == 3
    assert _iou(obj.geometry, roof) > 0.95


def test_detection_rate_counts_views_that_missed_the_object():
    """An object seen whole by three views but detected by one has rate 1/3."""
    windows = {k: box(0, 0, 60, 60) for k in 'ABC'}
    (obj,) = fuse_view_detections(
        [ViewDetection(box(10, 10, 20, 20), 'A', 0.6)], windows
    )
    assert (obj.n_views, obj.n_seen) == (1, 3)
    assert obj.detection_rate == pytest.approx(1 / 3)


def test_partial_overlap_goes_to_the_better_supported_object():
    """Contested area goes to the object more views detected."""
    windows = {k: box(0, 0, 60, 60) for k in 'ABC'}
    strong, weak = box(10, 10, 30, 20), box(28, 10, 40, 20)
    detections = [
        ViewDetection(strong, 'A', 0.9),
        ViewDetection(strong, 'B', 0.9),
        ViewDetection(weak, 'C', 0.9),
    ]
    objects = fuse_view_detections(detections, windows)
    assert len(objects) == 2
    by_views = {o.n_views: o.geometry for o in objects}
    assert _iou(by_views[2], strong) > 0.98
    assert by_views[1].intersection(by_views[2]).area == pytest.approx(0.0, abs=1e-9)
    assert by_views[1].area == pytest.approx(100.0, rel=0.03)


def test_fusion_handles_empty_input_and_missing_windows():
    """No detections give no objects; a detection without a window is an error."""
    assert fuse_view_detections([], WINDOWS) == []
    with pytest.raises(KeyError, match='No window'):
        fuse_view_detections([ViewDetection(box(0, 0, 5, 5), 'Z', None)], WINDOWS)


def test_outlines_touching_the_shared_window_edge_still_link():
    """A view whose outline also touches the shared window's edge links fine.

    Cut to the window both tiles share, such an outline is an area plus the
    line where it touches the edge; only the area is compared.
    """
    from shapely.geometry import GeometryCollection

    from rapidtools.processing.instance_fusion import _polygonal, _polygonal_array

    # An outline crossing into the shared window (x >= 25) with an arm that
    # reaches its edge from outside:
    hooked = box(15, 10, 35, 20).union(box(15, 20, 17, 30)).union(box(15, 30, 25, 32))
    cut = hooked.intersection(box(25, 0, 50, 50))
    assert isinstance(cut, GeometryCollection)
    assert _polygonal(cut).equals(box(25, 10, 35, 20))
    fixed = _polygonal_array(np.array([cut, box(0, 0, 1, 1)], dtype=object))
    assert [g.geom_type for g in fixed] == ['Polygon', 'Polygon']

    detections = [
        ViewDetection(hooked, 'A', 0.9),
        ViewDetection(box(24.8, 10.1, 35.1, 19.9), 'B', 0.8),
    ]
    objects = fuse_view_detections(detections, WINDOWS, neck_width=0)
    assert any(o.members == (0, 1) for o in objects)


def test_unscored_detections_have_no_score():
    """Objects fused from detections without scores report ``None``."""
    (obj,) = fuse_view_detections([ViewDetection(box(5, 5, 15, 15), 'A')], WINDOWS)
    assert obj.score is None
