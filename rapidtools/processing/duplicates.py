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
# 10-06-2026

"""
Fold two reports of one vehicle into one by asking a model to look.

Where one parked vehicle was reported twice, a long vehicle seen as a cab
and a box, a track split by a corner, or two passes that geometry could not
join, the two objects sit a few metres apart with good crops each. Geometry
cannot settle such pairs without also joining cars parked side by side, but
a vision-language model can: shown the crops of both with the detection
outlined, and told that the viewpoints differ, it says whether they show the
same physical vehicle. :class:`DuplicateResolver` is a pipeline step (stage
``VERIFY``, meant to run after :class:`~rapidtools.processing.DetectionVerifier`)
that asks that question of every pair of objects within ``max_distance_m``
and merges the pairs the model is confident about.

Each crop is zoomed to its detection outline, which is drawn thick, so a
neighbour that the cropper's margin let into the frame is not what gets
judged, and the model is made to name each vehicle's type and colour before
it answers: lighting changes brightness, never colour. On a labelled block
of the Spokane survey the step removed every remaining duplicate and merged
one pair of real vehicles by mistake.

Example:
    >>> from rapidtools import DuplicateResolver, Pipeline
    >>> from rapidtools.models import load
    >>> step = DuplicateResolver(load('gemini', api_key='AIza...'))  # doctest: +SKIP
    >>> vehicles = step(vehicles)  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.models.base import BaseInferenceModel, GenerationConfig

from .step import Stage

logger = logging.getLogger(__name__)

#: The question put to the model. Row A and row B are the two objects; the
#: viewpoints differ by construction, so the model is told to judge the
#: vehicle rather than the scene.
PAIR_PROMPT = (
    'Row A (top) and row B (bottom) each show crops of one parked vehicle '
    'detected along a street-view survey; the detection is outlined in '
    'magenta in each crop. The two detections lie within a few metres of '
    'each other and were made from DIFFERENT camera positions, often on '
    'different passes, so the angle, the background and the apparent '
    'position differ even when it is the same vehicle. A different '
    'viewpoint or background is not evidence of a different vehicle. '
    'Lighting changes brightness, never colour: a white vehicle is never '
    'the same as a red one, and a car is never the same as a pickup. First '
    'describe the outlined vehicle in each row, then decide. Return JSON '
    'with keys "type_a", "colour_a", "type_b", "colour_b" (short strings), '
    '"same" (true only if type, body style and colour all agree and no '
    'visible marking, roof rack, canopy or damage differs), "confidence" '
    '(0 to 1) and "reason" (one sentence naming the deciding property).'
)

Judge = Callable[[Sequence[Path], Sequence[Path]], tuple[bool | None, float | None]]


def _lon_lat(asset: PhysicalAsset) -> tuple[float, float] | None:
    geometry = asset.geometry
    if geometry is None or geometry.is_empty:
        return None
    centroid = geometry.centroid
    return float(centroid.x), float(centroid.y)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = (math.radians(v) for v in a)
    lon2, lat2 = (math.radians(v) for v in b)
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def _range_key(image: ImageAsset) -> float:
    value = image.properties.get('range_m')
    if value is None or isinstance(value, bool):
        return math.inf
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return math.inf
    return distance if distance == distance else math.inf


def parse_pair_verdict(text: str | None) -> tuple[bool | None, float | None, str]:
    """
    Read ``same``, ``confidence`` and ``reason`` out of a model reply.

    Args:
        text: The reply, JSON possibly wrapped in a code fence or a list.

    Returns:
        tuple[bool | None, float | None, str]: The verdict (``None`` when
        unreadable), the confidence (``None`` when absent) and the reason.

    Example:
        >>> parse_pair_verdict('{"same": true, "confidence": 0.95, "reason": "x"}')
        (True, 0.95, 'x')
        >>> parse_pair_verdict('not json')
        (None, None, '')
    """
    if not text:
        return None, None, ''
    body = text.strip()
    if body.startswith('```'):
        body = body.strip('`')
        body = body.removeprefix('json').strip()
    try:
        data: Any = json.loads(body)
    except ValueError:
        return None, None, ''
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), {})
    if not isinstance(data, dict):
        return None, None, ''
    same = data.get('same')
    if isinstance(same, str):
        same = same.strip().lower()
        same = (
            True
            if same in ('true', 'yes')
            else False
            if same in ('false', 'no')
            else None
        )
    elif same is not None:
        same = bool(same)
    confidence = data.get('confidence')
    try:
        confidence = (
            None if confidence is None else max(0.0, min(1.0, float(confidence)))
        )
    except (TypeError, ValueError):
        confidence = None
    return same, confidence, str(data.get('reason', '') or '')


class DuplicateResolver:
    """
    Merge pairs of nearby objects that a model says are one vehicle.

    Every pair of assets within ``max_distance_m`` of each other that both
    have downloaded crops is shown to the model as two rows of crops, row A
    and row B, with :data:`PAIR_PROMPT`. Pairs answered ``same`` with at
    least ``min_confidence`` are joined (chains too, through union-find).
    In each group the asset seen in the most frames is kept; it gains the
    others' sightings and crops and the attribute ``<prefix>_merged`` with
    their ids, and the others are removed (or kept, flagged
    ``<prefix>_of``, with ``keep_duplicates=True``).

    Args:
        model: Any wrapper from :func:`rapidtools.models.load` that accepts
            images. Exactly one of ``model`` and ``judge`` must be given.
        judge: A callable ``judge(paths_a, paths_b) -> (same, confidence)``
            standing in for the model, for tests or a custom comparator.
        max_distance_m: Farthest apart two objects may be and still be
            compared. Defaults to 8.
        min_confidence: Confidence at or above which a ``same`` verdict is
            acted on. Defaults to 0.9; on the labelled block every true
            duplicate scored 0.95 or more and the one false match that
            was not flagged at 0.5 came from crops without outlines.
        max_images_per_asset: Crops of each object shown, closest first.
            Defaults to 2.
        prompt: Replaces :data:`PAIR_PROMPT`.
        generation: Generation settings; defaults to JSON mode at
            temperature 0.
        max_workers: Parallel model calls. Defaults to one at a time.
        keep_duplicates: Keep the merged-away assets, flagged, instead of
            removing them.
        attribute_prefix: Prefix of the attributes written. Defaults to
            ``'duplicate'``.
        save_directory: Where the pair composites are written; a temporary
            directory by default.
        cancel_event: Set to stop early.

    Example:
        >>> resolver = DuplicateResolver(judge=lambda a, b: (True, 1.0))
        >>> resolver.stage
        <Stage.VERIFY: 45>
    """

    stage = Stage.VERIFY

    def __init__(
        self,
        model: BaseInferenceModel | None = None,
        *,
        judge: Judge | None = None,
        max_distance_m: float = 8.0,
        min_confidence: float = 0.9,
        max_images_per_asset: int = 2,
        prompt: str | None = None,
        generation: GenerationConfig | None = None,
        max_workers: int | None = None,
        keep_duplicates: bool = False,
        attribute_prefix: str = 'duplicate',
        save_directory: str | Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if (model is None) == (judge is None):
            raise ValueError('Give exactly one of model and judge.')
        if judge is not None and not callable(judge):
            raise TypeError('judge must be callable.')
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f'min_confidence must be between 0 and 1, got {min_confidence!r}.'
            )
        if max_images_per_asset < 1:
            raise ValueError('max_images_per_asset must be at least 1.')
        self.model = model
        self.judge = judge
        self.max_distance_m = float(max_distance_m)
        self.min_confidence = float(min_confidence)
        self.max_images_per_asset = int(max_images_per_asset)
        self.prompt = prompt or PAIR_PROMPT
        self.generation = generation or GenerationConfig(
            json_mode=True, temperature=0.0
        )
        self.max_workers = max_workers
        self.keep_duplicates = keep_duplicates
        self.attribute_prefix = attribute_prefix
        self.save_directory = Path(save_directory) if save_directory else None
        self.cancel_event = cancel_event

    @property
    def model_name(self) -> str:
        """Identifier of the model or ``'judge'``."""
        if self.model is None:
            return 'judge'
        return str(getattr(self.model, 'model_id', None) or type(self.model).__name__)

    # ------------------------------------------------------------- pairs
    def _crops(self, asset: PhysicalAsset) -> list[ImageAsset]:
        images = [img for img in asset.image_assets if img.is_downloaded]
        images.sort(key=_range_key)
        return images[: self.max_images_per_asset]

    def _crop_paths(self, asset: PhysicalAsset) -> list[Path]:
        return [Path(img.path) for img in self._crops(asset)]

    def candidate_pairs(
        self, collection: PhysicalAssetCollection
    ) -> list[tuple[PhysicalAsset, PhysicalAsset, float]]:
        """
        Pairs of assets within ``max_distance_m`` that both have crops.

        Returns:
            list[tuple[PhysicalAsset, PhysicalAsset, float]]: Each pair
            once, with its separation in metres.

        Example:
            >>> resolver.candidate_pairs(vehicles)  # doctest: +SKIP
            [(<PhysicalAsset veh_12>, <PhysicalAsset veh_57>, 3.1)]
        """
        located: list[tuple[PhysicalAsset, tuple[float, float]]] = []
        for asset in collection:
            position = _lon_lat(asset)
            if position is not None and self._crop_paths(asset):
                located.append((asset, position))
        # A degree of latitude is ~111 km; bucket by max_distance_m to stay linear.
        cell = max(self.max_distance_m, 1.0) / 111000.0
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        keys = []
        for i, (_, (lon, lat)) in enumerate(located):
            key = (int(math.floor(lon / cell)), int(math.floor(lat / cell)))
            grid[key].append(i)
            keys.append(key)
        pairs: list[tuple[PhysicalAsset, PhysicalAsset, float]] = []
        for i, (a, lla) in enumerate(located):
            gx, gy = keys[i]
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    for j in grid.get((gx + dx, gy + dy), ()):
                        if j <= i:
                            continue
                        b, llb = located[j]
                        d = _haversine_m(lla, llb)
                        if d <= self.max_distance_m:
                            pairs.append((a, b, d))
        pairs.sort(key=lambda t: (t[2], t[0].id, t[1].id))
        return pairs

    # ---------------------------------------------------------- composite
    @staticmethod
    def _tile(
        image: ImageAsset, size: tuple[int, int], margin: float = 0.2
    ) -> Image.Image | None:
        """
        One crop, zoomed to its outline with ``margin`` around it and the
        outline drawn thick, so the model cannot mistake a neighbour in the
        frame for the object. Without an outline record the crop is shown
        as it is.
        """
        try:
            with Image.open(image.path) as source:
                tile = source.convert('RGB')
        except OSError:
            return None
        outline = image.properties.get('outline') or []
        points = [(float(x), float(y)) for x, y in outline] if len(outline) >= 3 else []
        if points:
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            dx, dy = margin * (max(xs) - min(xs)), margin * (max(ys) - min(ys))
            box = (
                max(0, int(min(xs) - dx)),
                max(0, int(min(ys) - dy)),
                min(tile.width, int(max(xs) + dx) + 1),
                min(tile.height, int(max(ys) + dy) + 1),
            )
            if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                tile = tile.crop(box)
                points = [(x - box[0], y - box[1]) for x, y in points]
            stroke = max(3, int(round(min(tile.size) / 60)))
            ImageDraw.Draw(tile).line(
                points + [points[0]], fill=(255, 0, 255), width=stroke
            )
        if tile.height < size[1] // 2:
            # A far view is a small crop; the model cannot judge an 80-pixel
            # tile, so enlarge it (blurred is better than invisible).
            scale = (size[1] // 2) / tile.height
            tile = tile.resize(
                (max(1, int(tile.width * scale)), size[1] // 2),
                Image.Resampling.LANCZOS,
            )
        tile.thumbnail(size)
        return tile

    def compose(
        self, crops_a: Sequence[ImageAsset], crops_b: Sequence[ImageAsset], out: Path
    ) -> Path:
        """
        Lay two objects' crops out as row A over row B and save the image.

        Each crop is zoomed to its outline, which is drawn thick, so that a
        neighbour the cropper's margin let into the frame is not what the
        model judges.

        Args:
            crops_a: Crops of the first object, as attached by the cropper.
            crops_b: Crops of the second.
            out: Where to write the composite.

        Returns:
            Path: ``out``.
        """
        tile_w, tile_h, label_w = 580, 400, 90
        width = label_w + tile_w * max(len(crops_a), len(crops_b), 1) + 20
        canvas = Image.new('RGB', (width, 2 * (tile_h + 20)), 'black')
        draw = ImageDraw.Draw(canvas)
        for row, (tag, crops) in enumerate((('A', crops_a), ('B', crops_b))):
            y0 = row * (tile_h + 20) + 10
            draw.text((12, y0 + 4), f'row {tag}', fill='yellow')
            x = label_w
            for crop in crops:
                tile = self._tile(crop, (tile_w, tile_h))
                if tile is None:
                    continue
                canvas.paste(tile, (x, y0))
                x += tile_w + 10
        canvas.save(out, quality=90)
        return out

    def _ask(
        self, a: PhysicalAsset, b: PhysicalAsset, out: Path
    ) -> tuple[bool | None, float | None, str]:
        if self.judge is not None:
            same, confidence = self.judge(self._crop_paths(a), self._crop_paths(b))
            return same, confidence, ''
        assert self.model is not None
        composite = self.compose(self._crops(a), self._crops(b), out)
        try:
            result = self.model.run_inference(
                image_inputs=[composite], prompt=self.prompt, config=self.generation
            )
        except Exception as exc:  # a failed call is an unreadable verdict
            logger.warning(f'DuplicateResolver: model call failed ({exc}).')
            return None, None, ''
        text = getattr(result, 'text', None) if result is not None else None
        return parse_pair_verdict(text)

    # --------------------------------------------------------------- run
    def __call__(self, collection: PhysicalAssetCollection) -> PhysicalAssetCollection:
        """
        Compare nearby objects and merge the ones the model calls one vehicle.

        Args:
            collection: Assets with crops attached (after the cropper and,
                ideally, the verifier).

        Returns:
            PhysicalAssetCollection: The same collection, duplicates merged.
        """
        pairs = self.candidate_pairs(collection)
        if not pairs:
            logger.info(
                'DuplicateResolver: no pairs of objects within range to compare.'
            )
            return collection
        work_dir = self.save_directory or Path(
            tempfile.mkdtemp(prefix='rapidtools_pairs_')
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        def compare(index: int) -> tuple[int, bool | None, float | None, str]:
            if self.cancel_event is not None and self.cancel_event.is_set():
                return index, None, None, ''
            a, b, _ = pairs[index]
            out = work_dir / f'pair_{a.id}_{b.id}.jpg'
            same, confidence, reason = self._ask(a, b, out)
            return index, same, confidence, reason

        if self.max_workers and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                verdicts = list(pool.map(compare, range(len(pairs))))
        else:
            verdicts = [compare(i) for i in range(len(pairs))]

        parent: dict[str, str] = {a.id: a.id for a in collection}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        joined = unreadable = 0
        prefix = self.attribute_prefix
        for index, same, confidence, reason in verdicts:
            a, b, d = pairs[index]
            if same is None:
                unreadable += 1
                continue
            score = 1.0 if confidence is None else confidence
            if same and score >= self.min_confidence:
                parent[find(a.id)] = find(b.id)
                joined += 1
                logger.debug(
                    f'DuplicateResolver: {a.id} and {b.id} ({d:.1f} m) are one vehicle '
                    f'({score:.2f}): {reason}'
                )
        groups: dict[str, list[PhysicalAsset]] = defaultdict(list)
        for asset in collection:
            groups[find(asset.id)].append(asset)
        removed: list[str] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(
                key=lambda a: (
                    -int(a.attributes.get('n_images') or len(a.image_assets)),
                    a.id,
                )
            )
            keeper, others = members[0], members[1:]
            records = list(keeper.attributes.get('observations') or [])
            for other in others:
                records.extend(other.attributes.get('observations') or [])
                for image in other.image_assets:
                    keeper.add_image_assets(image)
                other.attributes[f'{prefix}_of'] = keeper.id
                removed.append(other.id)
            keeper.attributes['observations'] = records
            keeper.attributes['n_observations'] = len(records)
            keeper.attributes['n_images'] = len(
                {r.get('image_id') for r in records}
            ) or keeper.attributes.get('n_images')
            keeper.attributes[f'{prefix}_merged'] = sorted(
                keeper.attributes.get(f'{prefix}_merged', []) + [o.id for o in others]
            )
        logger.info(
            f'DuplicateResolver ({self.model_name}): {len(pairs)} pairs compared, '
            f'{joined} called one vehicle, {unreadable} without a verdict; '
            f'{len(removed)} duplicate objects merged.'
        )
        if removed and not self.keep_duplicates:
            collection.remove(removed)
        return collection
