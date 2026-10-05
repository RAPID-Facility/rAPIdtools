"""Tests for appearance re-identification of street-level candidates."""

import ast
import math
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from rapidtools.core import Observation
from rapidtools.processing import reid
from rapidtools.processing.reid import (
    AppearanceEmbedder,
    ReidCandidate,
    crop_observation,
    merge_by_appearance,
    merged_position,
    normalise_rows,
    pairs_to_compare,
    polygon_width_px,
    recommended_min_similarity,
    views_for_reid,
)
from rapidtools.processing.street_localization import destination_point, haversine_m

LON0, LAT0 = -117.45, 47.70

RED = (255, 0, 0)
RED_ISH = (255, 40, 0)
BLUE = (0, 0, 255)


def _solid(colour, size=(32, 24)):
    return Image.new('RGB', size, colour)


def _mean_colour_backend(images):
    """Embedding = mean RGB of the crop, so colour decides similarity."""
    return np.array(
        [
            np.asarray(im.convert('RGB'), dtype=float).reshape(-1, 3).mean(0)
            for im in images
        ]
    )


def _candidate(key, east_m=0.0, north_m=0.0, sequences=('s1',), images=None):
    lon, lat = destination_point(LON0, LAT0, 90.0, east_m)
    lon, lat = destination_point(lon, lat, 0.0, north_m)
    return ReidCandidate(
        key, lon, lat, frozenset(sequences), tuple(images or (f'img-{key}',))
    )


# ------------------------------------------------------------- embedder
def test_embedder_with_fake_backend_returns_unit_rows():
    calls = []

    def backend(images):
        calls.append(len(images))
        return np.array([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]])

    embedder = AppearanceEmbedder(backend=backend)
    out = embedder.embed([_solid(RED), _solid(BLUE), _solid(RED_ISH)])
    assert out.dtype == np.float32 and out.shape == (3, 2)
    assert out[0] == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(out[:2], axis=1) == pytest.approx([1.0, 1.0])
    assert out[2] == pytest.approx([0.0, 0.0])  # zero rows stay zero, no NaN
    assert calls == [3]
    assert embedder.model_id == 'openai/clip-vit-base-patch16'
    assert not embedder.is_loaded


def test_pool_outputs_prefers_clip_projection_then_cls_then_mean():
    from types import SimpleNamespace

    class Hidden:
        def mean(self, dim):
            return ('mean', dim)

    clip = SimpleNamespace(image_embeds='proj', pooler_output='cls')
    dino = SimpleNamespace(image_embeds=None, pooler_output='cls')
    bare = SimpleNamespace(last_hidden_state=Hidden())
    assert AppearanceEmbedder.pool_outputs(clip) == 'proj'
    assert AppearanceEmbedder.pool_outputs(dino) == 'cls'
    assert AppearanceEmbedder.pool_outputs(bare) == ('mean', 1)


def test_recommended_min_similarity_per_backbone():
    assert recommended_min_similarity('openai/clip-vit-base-patch16') == 0.89
    assert recommended_min_similarity('facebook/dinov2-small') == 0.83
    assert recommended_min_similarity('unknown/backbone') == 0.89


def test_merge_threshold_defaults_to_the_embedder_backbone():
    # Two crops at similarity 0.85: merged under DINOv2's 0.83, not CLIP's 0.89.
    def backend(images):
        return np.array([[1.0, 0.0], [0.85, math.sqrt(1 - 0.85**2)]])

    a, b = _candidate('a'), _candidate('b', east_m=2.0, sequences=('s2',))
    crops = {'a': _solid(RED), 'b': _solid(RED_ISH)}
    dino = AppearanceEmbedder('facebook/dinov2-small', backend=backend)
    clip = AppearanceEmbedder('openai/clip-vit-base-patch16', backend=backend)
    assert merge_by_appearance([a, b], crops, dino) == [['a', 'b']]
    assert merge_by_appearance([a, b], crops, clip) == [['a'], ['b']]


# ----------------------------------------------------------- width gate
def _obs(image_id, x0, x1):
    return Observation(
        image_id, 'car', [(x0, 0.5), (x1, 0.5), (x1, 0.6), (x0, 0.6)], 0.0, 0.0, 0.0
    )


def test_polygon_width_px_handles_seam():
    assert polygon_width_px([(0.40, 0.5), (0.45, 0.6)], 2048) == pytest.approx(102.4)
    assert polygon_width_px([(0.98, 0.5), (0.02, 0.6)], 2048) == pytest.approx(81.92)
    assert polygon_width_px([(0.98, 0.5), (0.02, 0.6)], 2048, panorama=False) == (
        pytest.approx(0.96 * 2048)
    )
    assert polygon_width_px([], 2048) == 0.0


def test_views_for_reid_orders_widest_first_and_drops_narrow():
    narrow = _obs('narrow', 0.40, 0.42)  # 41 px at 2048
    wide = _obs('wide', 0.40, 0.50)  # 205 px
    medium = _obs('medium', 0.40, 0.45)  # 102 px, just over the default gate
    views = views_for_reid([narrow, medium, wide], 2048)
    assert [o.image_id for o in views] == ['wide', 'medium']
    # The threshold scales with the thumbnail actually in use (1024 -> 50 px):
    assert [o.image_id for o in views_for_reid([narrow, medium], 1024)] == ['medium']
    assert views_for_reid([narrow], 2048) == []
    assert [o.image_id for o in views_for_reid([narrow], 2048, min_width_px=0)] == [
        'narrow'
    ]


def test_embedder_empty_input_and_row_count_check():
    embedder = AppearanceEmbedder(backend=lambda imgs: np.ones((1, 4)))
    assert embedder.embed([]).shape == (0, 0)
    with pytest.raises(ValueError, match='returned 1 rows for 2 images'):
        embedder.embed([_solid(RED), _solid(BLUE)])


def test_normalise_rows_rejects_non_2d():
    with pytest.raises(ValueError, match='2-D'):
        normalise_rows(np.ones(3))


# ------------------------------------------------------------- candidates
def test_candidate_derives_frame_ids_and_coerces_collections():
    cand = ReidCandidate('a', LON0, LAT0, ['s1', 's1'], ['i1', 'i2'])
    assert cand.sequence_ids == frozenset({'s1'})
    assert cand.image_ids == ('i1', 'i2')
    assert cand.frame_ids == frozenset({'i1', 'i2'})


def test_candidate_from_observations():
    tri = [(0, 0), (1, 0), (1, 1)]
    obs = [
        Observation('i2', 'car', tri, 0.0, 0.0, 0.0, sequence_id='s1'),
        Observation('i1', 'car', tri, 0.0, 0.0, 0.0, sequence_id='s2'),
        Observation('i2', 'car', tri, 0.0, 0.0, 0.0),
    ]
    cand = ReidCandidate.from_observations('a', LON0, LAT0, obs)
    assert cand.sequence_ids == frozenset({'s1', 's2'})
    assert cand.image_ids == ('i2', 'i1')


# --------------------------------------------------------------- cropping
def test_crop_observation_bbox_buffer_and_clamp():
    image = Image.new('RGB', (200, 100))
    box = [(0.4, 0.5), (0.5, 0.5), (0.5, 0.7), (0.4, 0.7)]  # 20 x 20 px
    assert crop_observation(image, box, buffer=0.0).size == (20, 20)
    # 15 % of 20 px = 3 px on each side:
    assert crop_observation(image, box, buffer=0.15).size == (26, 26)
    # A box touching the image corner is clamped, not padded:
    corner = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.2), (0.0, 0.2)]
    assert crop_observation(image, corner, buffer=0.5).size == (30, 30)
    # Degenerate polygons still yield a one-pixel crop:
    assert crop_observation(image, [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]).size == (1, 1)
    with pytest.raises(ValueError):
        crop_observation(image, [])


def test_crop_observation_handles_panorama_seam():
    pano = Image.new('RGB', (200, 100), (0, 0, 0))
    # Paint the two edges of the panorama that meet at the seam:
    pano.paste(RED, (190, 40, 200, 60))
    pano.paste(BLUE, (0, 40, 10, 60))
    seam_box = [(0.95, 0.4), (0.05, 0.4), (0.05, 0.6), (0.95, 0.6)]
    crop = crop_observation(pano, seam_box, buffer=0.0)
    assert crop.size == (20, 20)  # not the 180 px wide naive bbox
    colours = {px for px in crop.getdata()}
    assert RED in colours and BLUE in colours
    # The same polygon on a non-panoramic image is treated literally:
    flat = Image.new('RGB', (200, 120))
    assert crop_observation(flat, seam_box, buffer=0.0).size == (180, 24)


# ---------------------------------------------------------------- pairing
def test_pairs_to_compare_respects_distance():
    a = _candidate('a', sequences=('s1',))
    near = _candidate('near', east_m=4.0, sequences=('s2',))
    far = _candidate('far', north_m=7.0, sequences=('s3',))  # 8.06 m from near
    assert pairs_to_compare([a, near, far], max_distance_m=6.0) == [('a', 'near')]
    assert pairs_to_compare([a, near, far], max_distance_m=10.0) == [
        ('a', 'near'),
        ('a', 'far'),
        ('near', 'far'),
    ]
    assert pairs_to_compare([a], max_distance_m=6.0) == []
    assert pairs_to_compare([a, near], max_distance_m=0.0) == []


def test_pairs_to_compare_sequence_rule():
    a = _candidate('a', sequences=('s1',))
    b = _candidate('b', east_m=2.0, sequences=('s1',))
    assert pairs_to_compare([a, b]) == []
    assert pairs_to_compare([a, b], require_different_sequence=False) == [('a', 'b')]
    # Candidates without any sequence information are still compared:
    c = _candidate('c', east_m=3.0, sequences=())
    assert pairs_to_compare([a, c]) == [('a', 'c')]


def test_pairs_to_compare_shared_frame_rule():
    a = _candidate('a', sequences=('s1',), images=('f1', 'f2'))
    b = _candidate('b', east_m=2.0, sequences=('s2',), images=('f9', 'f2'))
    assert pairs_to_compare([a, b]) == []
    assert pairs_to_compare([a, b], require_different_sequence=False) == []


def test_pairs_to_compare_grid_matches_brute_force():
    rng = np.random.default_rng(7)
    candidates = [
        _candidate(
            f'c{i}',
            east_m=float(rng.uniform(-60, 60)),
            north_m=float(rng.uniform(-60, 60)),
            sequences=(f's{i % 3}',),
        )
        for i in range(80)
    ]
    expected = {
        (a.key, b.key)
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
        if not (a.sequence_ids & b.sequence_ids)
        and haversine_m(a.lon, a.lat, b.lon, b.lat) <= 6.0
    }
    assert expected  # the fixture does produce neighbours
    assert set(pairs_to_compare(candidates, max_distance_m=6.0)) == expected


def test_pairs_to_compare_rejects_duplicate_keys():
    with pytest.raises(ValueError, match='Duplicate candidate key'):
        pairs_to_compare([_candidate('a'), _candidate('a', east_m=1.0)])


# ---------------------------------------------------------------- merging
@pytest.fixture
def embedder():
    return AppearanceEmbedder(backend=_mean_colour_backend)


def test_merge_by_appearance_merges_similar_pair(embedder):
    a = _candidate('a', sequences=('s1',))
    b = _candidate('b', east_m=3.0, sequences=('s2',))
    crops = {'a': _solid(RED), 'b': _solid(RED_ISH)}
    assert merge_by_appearance([a, b], crops, embedder) == [['a', 'b']]


def test_merge_by_appearance_keeps_dissimilar_pair_apart(embedder):
    a = _candidate('a', sequences=('s1',))
    b = _candidate('b', east_m=3.0, sequences=('s2',))
    crops = {'a': _solid(RED), 'b': _solid(BLUE)}
    assert merge_by_appearance([a, b], crops, embedder) == [['a'], ['b']]
    # The threshold is honoured: RED vs RED_ISH is ~0.988 cosine.
    crops = {'a': _solid(RED), 'b': _solid(RED_ISH)}
    assert merge_by_appearance([a, b], crops, embedder, min_similarity=0.999) == [
        ['a'],
        ['b'],
    ]


def test_merge_by_appearance_keeps_shared_frame_pair_apart(embedder):
    a = _candidate('a', sequences=('s1',), images=('f1',))
    b = _candidate('b', east_m=3.0, sequences=('s2',), images=('f1',))
    crops = {'a': _solid(RED), 'b': _solid(RED)}
    assert merge_by_appearance([a, b], crops, embedder) == [['a'], ['b']]


def test_merge_by_appearance_leaves_candidates_without_crops_alone(embedder):
    a = _candidate('a', sequences=('s1',))
    b = _candidate('b', east_m=3.0, sequences=('s2',))
    seen = []

    def backend(images):
        seen.append(len(images))
        return _mean_colour_backend(images)

    emb = AppearanceEmbedder(backend=backend)
    assert merge_by_appearance([a, b], {'a': _solid(RED)}, emb) == [['a'], ['b']]
    assert seen == []  # nothing to compare, nothing embedded
    assert merge_by_appearance([a, b], {}, emb) == [['a'], ['b']]
    assert merge_by_appearance([], {}, emb) == []


def test_merge_by_appearance_embeds_each_crop_once(embedder):
    a = _candidate('a', sequences=('s1',))
    b = _candidate('b', east_m=2.0, sequences=('s2',))
    c = _candidate('c', east_m=4.0, sequences=('s3',))
    seen = []

    def backend(images):
        seen.append(len(images))
        return _mean_colour_backend(images)

    emb = AppearanceEmbedder(backend=backend)
    crops = {'a': _solid(RED), 'b': _solid(RED), 'c': _solid(RED)}
    assert merge_by_appearance([a, b, c], crops, emb) == [['a', 'b', 'c']]
    assert seen == [3]  # three pairs, but only three crops embedded


def test_merge_by_appearance_is_transitive(embedder):
    # a-c share a sequence so they are never compared directly; b bridges them.
    a = _candidate('a', sequences=('s1',))
    b = _candidate('b', east_m=3.0, sequences=('s2',))
    c = _candidate('c', east_m=6.0, sequences=('s1',))
    assert pairs_to_compare([a, b, c]) == [('a', 'b'), ('b', 'c')]
    crops = {'a': _solid(RED), 'b': _solid(RED), 'c': _solid(RED_ISH)}
    assert merge_by_appearance([a, b, c], crops, embedder) == [['a', 'b', 'c']]


def test_merge_by_appearance_shared_frame_rule_holds_through_chains(embedder):
    # a~b and b~c, but a and c were seen in the same frame: the better match
    # wins and the chain is broken rather than merging a with c.
    a = _candidate('a', sequences=('s1',), images=('f1',))
    b = _candidate('b', east_m=3.0, sequences=('s2',), images=('f2',))
    c = _candidate('c', east_m=6.0, sequences=('s3',), images=('f1',))
    crops = {'a': _solid(RED), 'b': _solid(RED), 'c': _solid(RED_ISH)}
    assert merge_by_appearance([a, b, c], crops, embedder) == [['a', 'b'], ['c']]


def test_merged_position_is_metre_frame_mean():
    p1 = (LON0, LAT0)
    p2 = destination_point(LON0, LAT0, 90.0, 4.0)
    lon, lat = merged_position([p1, p2])
    assert haversine_m(LON0, LAT0, lon, lat) == pytest.approx(2.0, abs=0.01)
    assert haversine_m(p2[0], p2[1], lon, lat) == pytest.approx(2.0, abs=0.01)
    assert merged_position([p1]) == pytest.approx(p1)
    with pytest.raises(ValueError):
        merged_position([])


# -------------------------------------------------------------- lazy torch
def _module_level_imports(path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    names = set()
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
    return names


def test_module_does_not_import_torch_at_import_time():
    """The backbone is loaded lazily: torch/transformers appear only in methods."""
    assert not {'torch', 'transformers'} & _module_level_imports(reid.__file__)
    # ``rapidtools.processing`` itself pulls torch in through the SAM3 modules,
    # so the subprocess checks that the reid module adds no torch names of its
    # own and that nothing is loaded until an image is actually embedded.
    script = (
        'import sys, types\n'
        'import rapidtools\n'
        "assert 'torch' not in sys.modules\n"
        'import rapidtools.processing.reid as reid\n'
        'mods = [v.__name__ for v in vars(reid).values() '
        'if isinstance(v, types.ModuleType)]\n'
        "assert not [m for m in mods if m.split('.')[0] in ('torch', 'transformers')]"
        ', mods\n'
        'emb = reid.AppearanceEmbedder(backend=lambda imgs: [[1.0, 0.0]] * len(imgs))\n'
        'assert emb._torch is None and not emb.is_loaded\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
