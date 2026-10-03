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
# 10-03-2026

"""
Appearance re-identification of street-level object candidates.

When a survey drives the same street twice, one parked car is discovered as
two candidates a few metres apart, each backed by sightings from a different
capture sequence. Position alone cannot tell such a pair from two cars parked
nose to tail, but their image crops can: this module embeds the best crop of
each candidate with a self-supervised vision backbone (DINOv2 by default) and
merges nearby candidates whose embeddings are close in cosine similarity.

Two rules keep the merge conservative. Candidates that share a frame were
seen at the same instant and are therefore different objects, so they are
never merged, directly or through a chain. By default candidates from the
same sequence are not compared either: duplicates within one pass are handled
upstream by the localisation clustering, and two objects visible to one camera
at once are distinct.

The heavy work is split so that a caller can download only the imagery it
needs: :func:`pairs_to_compare` lists the candidate pairs worth comparing,
:func:`crop_observation` cuts a crop from a thumbnail, and
:func:`merge_by_appearance` embeds the crops and returns the merge groups.

Example:
    >>> from rapidtools.processing.reid import (
    ...     AppearanceEmbedder, ReidCandidate, merge_by_appearance,
    ... )
    >>> a = ReidCandidate('a', -117.41, 47.66, frozenset({'s1'}), ('img-1',))
    >>> b = ReidCandidate('b', -117.41001, 47.66, frozenset({'s2'}), ('img-9',))
    >>> embedder = AppearanceEmbedder()  # loads facebook/dinov2-small lazily
    >>> merge_by_appearance(  # doctest: +SKIP
    ...     [a, b], {'a': crop_a, 'b': crop_b}, embedder
    ... )
    [['a', 'b']]
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from rapidtools.core import Observation

from .street_localization import haversine_m, local_projection

logger = logging.getLogger(__name__)

DEFAULT_REID_MODEL = 'facebook/dinov2-small'


# ------------------------------------------------------------- embedding
def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    """
    Scale every row of a 2-D array to unit L2 norm.

    All-zero rows are left as zeros instead of producing NaNs.

    Args:
        vectors: Array of shape ``(n, d)``.

    Returns:
        np.ndarray: ``float32`` array of the same shape with unit-norm rows.

    Example:
        >>> import numpy as np
        >>> normalise_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))  # doctest: +SKIP
        array([[0.6, 0.8],
               [0. , 0. ]], dtype=float32)
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f'Expected a 2-D array of embeddings, got shape {array.shape}.'
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


class AppearanceEmbedder:
    """
    L2-normalised image embeddings from a Hugging Face vision backbone.

    The default checkpoint is DINOv2-small, whose CLS token is a strong
    generic descriptor for "is this the same object" comparisons without
    any fine-tuning. Weights are loaded on first use, so constructing the
    embedder is free and ``torch`` is only imported when an image is embedded.

    Args:
        model_id: Hugging Face model identifier of a vision backbone that
            :class:`transformers.AutoModel` can load.
        device: ``'auto'`` picks CUDA when available and falls back to the
            CPU; any other value is passed to ``torch`` unchanged.
        batch_size: Number of images per forward pass.
        backend: Optional replacement for the model: a callable mapping a
            sequence of PIL images to an ``(n, d)`` array. Used in tests and
            for plugging in a different feature extractor.

    Example:
        >>> from PIL import Image
        >>> from rapidtools.processing.reid import AppearanceEmbedder
        >>> embedder = AppearanceEmbedder('facebook/dinov2-small')
        >>> vectors = embedder.embed(  # doctest: +SKIP
        ...     [Image.open('car_a.jpg'), Image.open('car_b.jpg')]
        ... )
        >>> vectors.shape, float(vectors[0] @ vectors[1])  # doctest: +SKIP
        ((2, 384), 0.91)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_REID_MODEL,
        device: str = 'auto',
        batch_size: int = 16,
        backend: Callable[[Sequence[Image.Image]], np.ndarray] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._backend = backend
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._resolved_device: str | None = None

    @property
    def is_loaded(self) -> bool:
        """``True`` once the backbone weights are in memory."""
        return self._model is not None

    def _load(self) -> None:
        """Import torch/transformers and load the backbone once."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        from rapidtools.auth import ensure_huggingface_login

        ensure_huggingface_login()
        device = self.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Loading re-identification backbone {self.model_id} on {device}.')
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(device).eval()
        self._torch = torch
        self._resolved_device = device

    def _embed_with_model(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Run the backbone over ``images`` in batches and pool each output."""
        if self._model is None:
            self._load()
        torch = self._torch
        rows: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            batch = [img.convert('RGB') for img in chunk]
            inputs = self._processor(images=batch, return_tensors='pt')
            inputs = inputs.to(self._resolved_device)
            with torch.inference_mode():
                outputs = self._model(**inputs)
            pooled = getattr(outputs, 'pooler_output', None)
            if pooled is None:
                pooled = outputs.last_hidden_state.mean(dim=1)
            rows.append(pooled.float().cpu().numpy())
        return np.concatenate(rows, axis=0)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """
        Embed images into unit-norm feature vectors.

        Args:
            images: PIL images; any mode, converted to RGB internally.

        Returns:
            np.ndarray: ``float32`` array of shape ``(n, d)`` whose rows have
            unit L2 norm, so a dot product is a cosine similarity. An empty
            input yields an array of shape ``(0, 0)``.

        Example:
            >>> import numpy as np
            >>> from PIL import Image
            >>> fake = lambda imgs: np.ones((len(imgs), 4))
            >>> AppearanceEmbedder(backend=fake).embed([Image.new('RGB', (8, 8))])
            array([[0.5, 0.5, 0.5, 0.5]], dtype=float32)
        """
        if len(images) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if self._backend is not None:
            vectors = np.asarray(self._backend(images), dtype=np.float32)
        else:
            vectors = self._embed_with_model(images)
        if vectors.shape[0] != len(images):
            raise ValueError(
                f'Embedding backend returned {vectors.shape[0]} rows for '
                f'{len(images)} images.'
            )
        return normalise_rows(vectors)


# ------------------------------------------------------------ candidates
@dataclass(frozen=True)
class ReidCandidate:
    """
    What the re-identification step needs to know about one candidate.

    Attributes:
        key (str):
            Caller's identifier for the candidate (for example the asset id).
        lon (float):
            Longitude of the candidate's estimated position in WGS84 degrees.
        lat (float):
            Latitude of the candidate's estimated position in WGS84 degrees.
        sequence_ids (frozenset[str]):
            Capture sequences whose images saw the candidate.
        image_ids (tuple[str, ...]):
            Identifiers of the frames that saw the candidate, closest view
            first when the caller knows the ordering.
        frame_ids (frozenset[str]):
            ``image_ids`` as a set; derived automatically.

    Example:
        >>> c = ReidCandidate('a', -117.4, 47.66, frozenset({'s1'}), ('i1', 'i2'))
        >>> sorted(c.frame_ids)
        ['i1', 'i2']
    """

    key: str
    lon: float
    lat: float
    sequence_ids: frozenset[str]
    image_ids: tuple[str, ...]
    frame_ids: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Coerce the collections and derive ``frame_ids``."""
        object.__setattr__(self, 'sequence_ids', frozenset(self.sequence_ids))
        object.__setattr__(self, 'image_ids', tuple(self.image_ids))
        object.__setattr__(self, 'frame_ids', frozenset(self.image_ids))

    @classmethod
    def from_observations(
        cls, key: str, lon: float, lat: float, observations: Sequence[Observation]
    ) -> ReidCandidate:
        """
        Build a candidate from the observations backing an asset.

        Args:
            key: Caller's identifier for the candidate.
            lon: Longitude of the candidate in WGS84 degrees.
            lat: Latitude of the candidate in WGS84 degrees.
            observations: Sightings of the candidate; their order is kept for
                ``image_ids`` and sightings without a sequence id contribute
                no sequence.

        Returns:
            ReidCandidate: The candidate record.

        Example:
            >>> from rapidtools.core import Observation
            >>> obs = Observation('i1', 'car', [(0, 0), (1, 0), (1, 1)], 0.0, 0.0,
            ...                   0.0, sequence_id='s1')
            >>> ReidCandidate.from_observations('a', 0.0, 0.0, [obs]).sequence_ids
            frozenset({'s1'})
        """
        sequences = frozenset(
            o.sequence_id for o in observations if o.sequence_id is not None
        )
        image_ids = tuple(dict.fromkeys(o.image_id for o in observations))
        return cls(key, lon, lat, sequences, image_ids)


# -------------------------------------------------------------- cropping
def crop_observation(
    image: Image.Image,
    polygon: Sequence[tuple[float, float]],
    buffer: float = 0.15,
) -> Image.Image:
    """
    Cut the axis-aligned crop of a detection polygon out of its source image.

    The polygon's bounding box is expanded by ``buffer`` times its width and
    height on each side and clamped to the image. In an equirectangular
    panorama (width twice the height) a polygon that wraps around the seam
    is handled by rolling the image half a turn first, so the crop stays
    contiguous.

    Args:
        image: Source image (thumbnail or full frame).
        polygon: Detection outline in normalised image coordinates, ``x``
            right and ``y`` down in ``[0, 1]``.
        buffer: Context margin as a fraction of the box size.

    Returns:
        PIL.Image.Image: The crop; at least one pixel in each direction.

    Example:
        >>> from PIL import Image
        >>> image = Image.new('RGB', (200, 100))
        >>> box = [(0.4, 0.5), (0.5, 0.5), (0.5, 0.7), (0.4, 0.7)]
        >>> crop_observation(image, box, buffer=0.0).size
        (20, 20)
    """
    width, height = image.size
    points = [(float(x), float(y)) for x, y in polygon]
    if not points:
        raise ValueError('A detection polygon needs at least one vertex.')
    xs = [x for x, _ in points]
    if width == 2 * height and (max(xs) - min(xs)) > 0.5:
        # The object straddles the panorama seam: make it contiguous.
        if image.mode not in ('RGB', 'RGBA', 'L'):
            image = image.convert('RGB')
        image = Image.fromarray(np.roll(np.asarray(image), width // 2, axis=1))
        points = [((x + 0.5) % 1.0, y) for x, y in points]
    # Round away float noise from the seam shift before snapping to pixels:
    px = [round(x * width, 6) for x, _ in points]
    py = [round(y * height, 6) for _, y in points]
    left, right = min(px), max(px)
    top, bottom = min(py), max(py)
    margin_x = buffer * (right - left)
    margin_y = buffer * (bottom - top)
    x0 = max(0, int(math.floor(left - margin_x)))
    x1 = min(width, int(math.ceil(right + margin_x)))
    y0 = max(0, int(math.floor(top - margin_y)))
    y1 = min(height, int(math.ceil(bottom + margin_y)))
    if x1 <= x0:
        x0, x1 = (x0, x0 + 1) if x0 < width else (width - 1, width)
    if y1 <= y0:
        y0, y1 = (y0, y0 + 1) if y0 < height else (height - 1, height)
    return image.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------- pairing
def _comparable(
    a: ReidCandidate, b: ReidCandidate, require_different_sequence: bool
) -> bool:
    """Whether two candidates may be the same object as far as metadata goes."""
    if a.frame_ids & b.frame_ids:
        return False
    if require_different_sequence and (a.sequence_ids & b.sequence_ids):
        return False
    return True


def _check_unique_keys(candidates: Sequence[ReidCandidate]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.key in seen:
            raise ValueError(f'Duplicate candidate key {candidate.key!r}.')
        seen.add(candidate.key)


def pairs_to_compare(
    candidates: Sequence[ReidCandidate],
    max_distance_m: float = 6.0,
    require_different_sequence: bool = True,
) -> list[tuple[str, str]]:
    """
    List the candidate pairs whose appearance is worth comparing.

    A pair qualifies when the two positions are within ``max_distance_m``,
    the candidates share no frame, and (by default) they come from different
    capture sequences. A grid index with cells of ``max_distance_m`` keeps
    this linear in the number of candidates for a city-scale survey.

    Args:
        candidates: Candidate records with unique keys.
        max_distance_m: Largest great-circle separation to consider.
        require_different_sequence: Skip pairs that share a sequence id.

    Returns:
        list[tuple[str, str]]: Key pairs in input order, each listed once.

    Example:
        >>> a = ReidCandidate('a', 0.0, 0.0, frozenset({'s1'}), ('i1',))
        >>> b = ReidCandidate('b', 0.00003, 0.0, frozenset({'s2'}), ('i2',))
        >>> c = ReidCandidate('c', 0.001, 0.0, frozenset({'s2'}), ('i3',))
        >>> pairs_to_compare([a, b, c], max_distance_m=6.0)
        [('a', 'b')]
    """
    _check_unique_keys(candidates)
    if len(candidates) < 2 or max_distance_m <= 0:
        return []
    project, _ = local_projection(candidates[0].lon, candidates[0].lat)
    cell = float(max_distance_m)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cells: list[tuple[int, int]] = []
    for index, candidate in enumerate(candidates):
        x, y = project(candidate.lon, candidate.lat)
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        grid[key].append(index)
        cells.append(key)
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(candidates):
        gx, gy = cells[i]
        neighbours: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbours.extend(j for j in grid.get((gx + dx, gy + dy), ()) if j > i)
        for j in sorted(neighbours):
            b = candidates[j]
            if not _comparable(a, b, require_different_sequence):
                continue
            if haversine_m(a.lon, a.lat, b.lon, b.lat) > max_distance_m:
                continue
            pairs.append((a.key, b.key))
    return pairs


# ---------------------------------------------------------------- merging
class _UnionFind:
    """Disjoint sets over string keys with path compression."""

    def __init__(self, keys: Sequence[str]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, key: str) -> str:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra
        return ra

    def groups(self, order: Sequence[str]) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for key in order:
            grouped[self.find(key)].append(key)
        return list(grouped.values())


def merge_by_appearance(
    candidates: Sequence[ReidCandidate],
    crops: Mapping[str, Image.Image],
    embedder: AppearanceEmbedder,
    max_distance_m: float = 6.0,
    min_similarity: float = 0.80,
    require_different_sequence: bool = True,
) -> list[list[str]]:
    """
    Group candidates that are close together and look alike.

    Candidate pairs from :func:`pairs_to_compare` whose crops reach a cosine
    similarity of ``min_similarity`` are joined with union-find, most similar
    pairs first. A join that would put two candidates sharing a frame into
    one group is refused, so the shared-frame rule holds through chains as
    well. Each crop is embedded once; candidates without a crop stay alone.

    Args:
        candidates: Candidate records with unique keys.
        crops: Best crop per candidate key; keys may be missing.
        embedder: Produces unit-norm embeddings of the crops.
        max_distance_m: Largest separation of two candidates to compare.
        min_similarity: Cosine similarity at or above which a pair merges.
        require_different_sequence: Only compare candidates from different
            capture sequences.

    Returns:
        list[list[str]]: Groups of candidate keys covering every candidate,
        singletons included, in first-appearance order.

    Example:
        >>> groups = merge_by_appearance(  # doctest: +SKIP
        ...     candidates, crops, AppearanceEmbedder()
        ... )
        >>> [g for g in groups if len(g) > 1]  # doctest: +SKIP
        [['car-12', 'car-57']]
    """
    all_pairs = pairs_to_compare(candidates, max_distance_m, require_different_sequence)
    pairs = [(a, b) for a, b in all_pairs if a in crops and b in crops]
    order = [c.key for c in candidates]
    union = _UnionFind(order)
    if not pairs:
        return union.groups(order)

    keys = list(dict.fromkeys(k for pair in pairs for k in pair))
    vectors = embedder.embed([crops[k] for k in keys])
    column = {k: i for i, k in enumerate(keys)}
    scored: list[tuple[float, int, str, str]] = []
    for rank, (a, b) in enumerate(pairs):
        similarity = float(vectors[column[a]] @ vectors[column[b]])
        if similarity >= min_similarity:
            scored.append((similarity, rank, a, b))
    scored.sort(key=lambda item: (-item[0], item[1]))

    frames = {c.key: set(c.frame_ids) for c in candidates}
    for similarity, _, a, b in scored:
        ra, rb = union.find(a), union.find(b)
        if ra == rb:
            continue
        if frames[ra] & frames[rb]:
            logger.debug(
                f'Not merging {a} and {b} (similarity {similarity:.2f}): '
                'their groups share a frame.'
            )
            continue
        root = union.union(ra, rb)
        frames[root] = frames[ra] | frames[rb]
    groups = union.groups(order)
    merged = sum(1 for g in groups if len(g) > 1)
    logger.debug(
        f'Appearance re-identification compared {len(pairs)} pairs and '
        f'merged {merged} groups.'
    )
    return groups


def merged_position(
    group_positions: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """
    Mean position of a merged group, averaged in a local metre frame.

    Args:
        group_positions: ``(lon, lat)`` pairs of the group members.

    Returns:
        tuple[float, float]: ``(lon, lat)`` of the centroid.

    Example:
        >>> lon, lat = merged_position([(-117.4, 47.66), (-117.4, 47.6601)])
        >>> round(lat, 5)
        47.66005
    """
    if not group_positions:
        raise ValueError('merged_position needs at least one position.')
    lon0, lat0 = group_positions[0]
    project, unproject = local_projection(lon0, lat0)
    xy = np.array([project(lon, lat) for lon, lat in group_positions], dtype=float)
    x, y = xy.mean(axis=0)
    return unproject(float(x), float(y))
