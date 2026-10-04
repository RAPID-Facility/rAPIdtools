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
# 09-22-2026

"""
Pipeline step that confirms detections with a model and drops false positives.

Street-level detectors such as :class:`~rapidtools.processing.MapillaryFeatureExtractor`
return every object Mapillary labelled as, say, a vehicle, and some of those
are not vehicles at all. :class:`DetectionVerifier` sends each object's
nearest crops to a model of your choice, asks whether the image really shows
what the asset claims to be, and removes the objects that fail. It sits in the
:attr:`~rapidtools.processing.Stage.VERIFY` stage, after the imagery
extractors and before :class:`~rapidtools.processing.AssetAnalyzer`, so
rejected objects are never sent to the (usually paid) analysis model.

Example:
    >>> from rapidtools.models import load
    >>> from rapidtools.processing import DetectionVerifier
    >>> verifier = DetectionVerifier(load('gemini', api_key='AIza...'))
    >>> collection = verifier(collection)  # doctest: +SKIP
    >>> collection.get('veh_01').attributes['verify_accepted']  # doctest: +SKIP
    True
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rapidtools.core import (
    ImageAsset,
    ImageCollection,
    PhysicalAsset,
    PhysicalAssetCollection,
    raise_if_cancelled,
)
from rapidtools.models.base import BaseInferenceModel, GenerationConfig

from .image_analyzers import AssetAnalyzer, RateLimitPolicy
from .step import Stage

logger = logging.getLogger(__name__)

#: Signature of a user-supplied classifier: ``(image_paths, description)`` ->
#: one ``(match, confidence, label)`` tuple per image.
Classifier = Callable[[list[Path], str], list[tuple[bool, float, str]]]

#: What common street-level asset types must look like to be accepted.
_VEHICLE = (
    'a motor vehicle: a car, SUV, van, pickup, truck, bus or motorcycle. Not a '
    'trailer, camper or caravan, boat, jet ski, bicycle, lawn mower, piece of '
    'equipment or a part of a vehicle on its own (a wheel, a bumper)'
)
DEFAULT_DESCRIPTIONS: dict[str, str] = {
    'vehicle': _VEHICLE,
    'vehicles': _VEHICLE,
    'motor_vehicle': _VEHICLE,
    'motor_vehicles': _VEHICLE,
    'car': _VEHICLE,
    'cars': _VEHICLE,
    'trailer': 'a trailer, camper or caravan (towed, no engine of its own)',
    'trailers': 'a trailer, camper or caravan (towed, no engine of its own)',
    'boat': 'a boat (on a trailer, in a driveway or on the water)',
    'boats': 'a boat (on a trailer, in a driveway or on the water)',
    'utility_pole': 'a utility pole (a wooden, concrete or steel pole carrying '
    'power or telecommunication lines)',
    'utility_poles': 'a utility pole (a wooden, concrete or steel pole carrying '
    'power or telecommunication lines)',
    'fire_hydrant': 'a fire hydrant',
    'fire_hydrants': 'a fire hydrant',
    'traffic_sign': 'a traffic sign (a regulatory, warning or guide sign on a post)',
    'traffic_signs': 'a traffic sign (a regulatory, warning or guide sign on a post)',
    'street_light': 'a street light (a lamp post or light fixture lighting the road)',
    'street_lights': 'a street light (a lamp post or light fixture lighting the road)',
    'debris_pile': 'a debris pile (a heap of rubble, vegetation, waste or '
    'wreckage left outdoors)',
    'debris_piles': 'a debris pile (a heap of rubble, vegetation, waste or '
    'wreckage left outdoors)',
}

#: Description used when an asset declares no type at all.
_UNTYPED_DESCRIPTION = 'the object it was detected as'

DEFAULT_PROMPT = (
    'You are checking the output of an automatic object detector. The image '
    'is a crop from a street-level photograph; the thin outline marks the '
    'detection to judge. Judge only the object inside the outline, not the '
    'rest of the crop.\n\n'
    'Question: is the outlined object {description}?\n\n'
    'Be strict. Answer true only if you can clearly recognise the outlined '
    'object as {description}. Answer false if the outline contains something '
    'else, only a small part of such an object, or an object so hidden, '
    'distant or blurred that you cannot tell what it is. Also answer false '
    'when the outline is so small or loose that it takes in several objects, '
    'or when you cannot say which single object it marks.\n\n'
    'Reply with a single JSON object and nothing else, with these keys:\n'
    '  "match": true or false as defined above;\n'
    '  "confidence": a number from 0 to 1 giving how sure you are of "match";\n'
    '  "visible_fraction": a number from 0 to 1 estimating how much of the '
    'outlined object is actually visible (1 = fully visible, 0.3 = mostly '
    'hidden by other things or cut off);\n'
    '  "label": a short noun phrase naming what the outline actually contains '
    '(for example "parked sedan", "utility trailer", "recycling bin", '
    '"tree trunk").'
)

_TRUE_WORDS = frozenset({'true', 'yes', 'y', '1', 'match', 'correct'})
_FALSE_WORDS = frozenset({'false', 'no', 'n', '0', 'mismatch', 'incorrect'})


def normalize_match(value: Any) -> bool | None:
    """
    Coerce a model's ``match`` value to a boolean.

    Accepts booleans, numbers and the usual spellings (``'true'``, ``'yes'``,
    ``'no'`` ...). Returns ``None`` when the value cannot be interpreted.

    Example:
        >>> normalize_match('Yes'), normalize_match('false'), normalize_match(1)
        (True, False, True)
        >>> normalize_match('maybe') is None
        True
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower().rstrip('.')
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
    return None


def normalize_confidence(value: Any) -> float | None:
    """
    Coerce a model's ``confidence`` value to a float in ``[0, 1]``.

    Percentages (``'85%'`` or any number above 1) are scaled down; anything
    that is not a number yields ``None``.

    Example:
        >>> normalize_confidence('0.9'), normalize_confidence('85%')
        (0.9, 0.85)
        >>> normalize_confidence('high') is None
        True
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        value = value.strip().rstrip('%')
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if number > 1.0:
        number = number / 100.0
    return min(1.0, max(0.0, number))


def describe_asset_type(asset_type: str | None) -> str:
    """
    Return the default description of what an asset type must look like.

    Known street-level types come from :data:`DEFAULT_DESCRIPTIONS`; other
    types are spelled out from their name.

    Example:
        >>> describe_asset_type('utility_pole')  # doctest: +ELLIPSIS
        'a utility pole (a wooden, concrete or steel pole carrying power or...'
        >>> describe_asset_type('mail_box')
        'a mail box'
        >>> describe_asset_type('awning')
        'an awning'
    """
    if not asset_type:
        return _UNTYPED_DESCRIPTION
    key = asset_type.strip().lower()
    if key in DEFAULT_DESCRIPTIONS:
        return DEFAULT_DESCRIPTIONS[key]
    words = key.replace('_', ' ').replace('-', ' ').strip() or _UNTYPED_DESCRIPTION
    article = 'an' if words[0] in 'aeiou' else 'a'
    return f'{article} {words}'


def _range_key(image: ImageAsset) -> tuple[int, float]:
    """Sort key placing the nearest images first and unknown ranges last."""
    value = image.properties.get('range_m')
    if value is None or isinstance(value, bool):
        return (1, 0.0)
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return (1, 0.0)
    if distance != distance:  # NaN
        return (1, 0.0)
    return (0, distance)


class _VerificationAnalyzer(AssetAnalyzer):
    """
    :class:`AssetAnalyzer` that only records the verification verdict.

    The parent writes every key of the model's JSON plus ``ai_model_used``;
    here only ``match``, ``confidence`` and ``label`` are kept (normalised),
    so the verifier never clobbers attributes written by a later analysis.
    """

    def _apply_result(self, asset: PhysicalAsset, img_paths: list, result: Any) -> bool:
        if result is None or not result.text:
            return False
        data = self._parse_structured_results(result.text)
        prefix = self.attribute_prefix
        match = normalize_match(data.get('match'))
        if match is None:
            # The reply did not contain a usable verdict; keep the raw text
            # so the user can see what came back.
            asset.attributes[f'{prefix}_raw'] = result.text
            return False
        asset.attributes[f'{prefix}_match'] = match
        asset.attributes[f'{prefix}_confidence'] = normalize_confidence(
            data.get('confidence')
        )
        asset.attributes[f'{prefix}_visible_fraction'] = normalize_confidence(
            data.get('visible_fraction')
        )
        label = data.get('label')
        asset.attributes[f'{prefix}_label'] = (
            str(label).strip() if label is not None else None
        )
        return True


class DetectionVerifier:
    """
    Confirm that each detected object is what its asset type claims.

    For every asset the verifier picks the nearest downloaded images (by the
    ``range_m`` image property written by
    :class:`~rapidtools.processing.MapillaryObjectImageExtractor`), asks a
    model whether they show the expected kind of object, and removes the
    assets that do not. Any wrapper from :func:`rapidtools.models.load` can be
    used, or a plain Python callable for a custom classifier.

    Each verified asset receives the attributes ``<prefix>_match`` (bool),
    ``<prefix>_confidence`` (float or ``None``), ``<prefix>_label`` (what the
    model saw), ``<prefix>_accepted`` (bool) and ``<prefix>_model``. Assets
    without downloaded images, and assets whose request failed, are kept with
    ``<prefix>_accepted = None``.

    Args:
        model (BaseInferenceModel | None):
            Any rapidtools model wrapper (see :func:`rapidtools.models.load`).
            Exactly one of ``model`` and ``classifier`` must be given.
        classifier (Callable | None):
            Custom alternative to ``model``: called once per asset as
            ``classifier(image_paths, description)`` and must return one
            ``(match, confidence, label)`` tuple per image. The asset is
            accepted when any image matches with at least ``min_confidence``.
        descriptions (Mapping[str, str] | None):
            ``asset_type -> what it must be``, e.g. ``{'vehicles': 'a vehicle
            (car, truck, bus, van, pickup, trailer or motorcycle)'}``. Types
            not listed fall back to :func:`describe_asset_type`.
        prompt (str | None):
            Replaces :data:`DEFAULT_PROMPT`; ``{description}`` is substituted
            with the asset type's description. The model must still reply with
            a JSON object holding ``match``, ``confidence`` and ``label``.
        max_images_per_asset (int):
            Nearest images sent per asset. Defaults to 2.
        min_visible_fraction (float):
            Objects the model reports as less visible than this fraction
            (hidden behind other things, cut off, too distant to tell) are
            rejected even when it says they match. Defaults to 0.5.
        min_confidence (float):
            Minimum ``confidence`` for a positive ``match`` to count. A
            missing confidence counts as 1.0. Defaults to 0.5.
        keep_rejected (bool):
            Keep rejected assets (flagged ``<prefix>_accepted = False``)
            instead of removing them. Defaults to ``False``.
        attribute_prefix (str):
            Prefix of the attributes written. Defaults to ``'verify'``.
        generation (GenerationConfig | None):
            Generation options for ``model``; defaults to JSON mode at
            temperature 0.
        max_workers (int | None):
            Concurrent requests for hosted models (see
            :class:`~rapidtools.processing.AssetAnalyzer`).
        rate_limit (RateLimitPolicy | None):
            Back-off and retry policy for hosted models.
        cancel_event (threading.Event | None):
            Cooperative cancellation flag; raises
            :class:`~rapidtools.core.OperationCancelled` when set.

    Raises:
        ValueError: If neither or both of ``model`` and ``classifier`` are
            given, or ``min_confidence`` is outside ``[0, 1]``.

    Example:
        Detect vehicles from Mapillary, attach their crops, discard the
        detections a model does not recognise as vehicles, and only then pay
        for the detailed analysis:

        >>> import rapidtools as rt
        >>> from rapidtools.models import load
        >>> from rapidtools.processing import (
        ...     AssetAnalyzer,
        ...     DetectionVerifier,
        ...     MapillaryFeatureExtractor,
        ...     MapillaryObjectImageExtractor,
        ... )
        >>> pipeline = rt.Pipeline([  # doctest: +SKIP
        ...     MapillaryFeatureExtractor(token, labels=['vehicles']),
        ...     MapillaryObjectImageExtractor(token, save_directory='crops'),
        ...     DetectionVerifier(load('gemini')),
        ...     AssetAnalyzer(load('gemini'), prompt='Describe the damage as JSON.'),
        ... ])
        >>> collection = pipeline.run(collection)  # doctest: +SKIP

        A custom classifier needs no model wrapper at all:

        >>> def my_classifier(paths, description):
        ...     return [(True, 0.9, 'car') for _ in paths]
        >>> verifier = DetectionVerifier(classifier=my_classifier, min_confidence=0.8)
        >>> verifier.stage
        <Stage.VERIFY: 45>
    """

    stage = Stage.VERIFY

    def __init__(
        self,
        model: BaseInferenceModel | None = None,
        *,
        classifier: Classifier | None = None,
        descriptions: Mapping[str, str] | None = None,
        prompt: str | None = None,
        max_images_per_asset: int = 2,
        min_confidence: float = 0.5,
        min_visible_fraction: float = 0.5,
        keep_rejected: bool = False,
        attribute_prefix: str = 'verify',
        generation: GenerationConfig | None = None,
        max_workers: int | None = None,
        rate_limit: RateLimitPolicy | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Configure the verifier; see the class docstring for the arguments."""
        if (model is None) == (classifier is None):
            raise ValueError(
                'DetectionVerifier needs exactly one of model= or classifier=.'
            )
        if classifier is not None and not callable(classifier):
            raise TypeError('classifier must be callable.')
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f'min_confidence must be between 0 and 1, got {min_confidence!r}.'
            )
        if max_images_per_asset < 1:
            raise ValueError('max_images_per_asset must be at least 1.')
        if not attribute_prefix:
            raise ValueError('attribute_prefix must not be empty.')

        self.model = model
        self.classifier = classifier
        self.descriptions = {
            str(key).strip().lower(): value
            for key, value in (descriptions or {}).items()
        }
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.max_images_per_asset = int(max_images_per_asset)
        self.min_confidence = float(min_confidence)
        if not 0.0 <= min_visible_fraction <= 1.0:
            raise ValueError(
                'min_visible_fraction must be between 0 and 1, got '
                f'{min_visible_fraction!r}.'
            )
        self.min_visible_fraction = float(min_visible_fraction)
        self.keep_rejected = keep_rejected
        self.attribute_prefix = attribute_prefix
        self.generation = (
            generation
            if generation is not None
            else GenerationConfig(json_mode=True, temperature=0.0)
        )
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.cancel_event = cancel_event

    # --------------------------------------------------------------- helpers
    @property
    def model_name(self) -> str:
        """Identifier written to ``<prefix>_model`` on every verified asset."""
        if self.model is None:
            return 'custom'
        model_id = getattr(self.model, 'model_id', None)
        return str(model_id) if model_id else type(self.model).__name__

    def description_for(self, asset_type: str | None) -> str:
        """
        Return what assets of ``asset_type`` must look like to be accepted.

        Example:
            >>> v = DetectionVerifier(classifier=lambda p, d: [])
            >>> v.description_for('vehicles')
            'a vehicle (car, truck, bus, van, pickup, trailer or motorcycle)'
        """
        key = (asset_type or '').strip().lower()
        if key in self.descriptions:
            return self.descriptions[key]
        return describe_asset_type(asset_type)

    def prompt_for(self, asset_type: str | None) -> str:
        """Return the prompt sent for assets of ``asset_type``."""
        return self.prompt.replace('{description}', self.description_for(asset_type))

    def _sort_images_by_range(self, asset: PhysicalAsset) -> None:
        """Reorder an asset's images nearest first so the analyzer picks them."""
        images = list(asset.image_assets)
        if len(images) < 2:
            return
        ordered = sorted(images, key=_range_key)
        if ordered != images:
            asset.image_assets = ImageCollection(ordered)

    def _selected_paths(self, asset: PhysicalAsset) -> list[Path]:
        """Paths of the nearest downloaded images of ``asset``."""
        downloaded = [img for img in asset.image_assets if img.is_downloaded]
        downloaded.sort(key=_range_key)
        return [img.path for img in downloaded[: self.max_images_per_asset]]

    def _clear_verdict(self, asset: PhysicalAsset) -> None:
        """Drop any verdict left by a previous run."""
        for suffix in (
            'match',
            'confidence',
            'visible_fraction',
            'label',
            'accepted',
            'model',
            'raw',
        ):
            asset.attributes.pop(f'{self.attribute_prefix}_{suffix}', None)

    def _decide(self, asset: PhysicalAsset) -> bool | None:
        """Turn the recorded ``match``/``confidence`` into ``accepted``."""
        prefix = self.attribute_prefix
        match = normalize_match(asset.attributes.get(f'{prefix}_match'))
        if match is None:
            return None
        if not match:
            return False
        confidence = normalize_confidence(asset.attributes.get(f'{prefix}_confidence'))
        if confidence is None:
            confidence = 1.0
        visible = normalize_confidence(
            asset.attributes.get(f'{prefix}_visible_fraction')
        )
        if visible is not None and visible < self.min_visible_fraction:
            return False  # too hidden to trust, as asked
        return confidence >= self.min_confidence

    # ---------------------------------------------------------- model route
    def _verify_with_model(self, groups: dict[str | None, list[PhysicalAsset]]) -> None:
        """Run one analyzer per asset type so each group gets its own prompt."""
        assert self.model is not None
        for asset_type, assets in groups.items():
            raise_if_cancelled(self.cancel_event, 'detection verification')
            for asset in assets:
                self._sort_images_by_range(asset)
            analyzer = _VerificationAnalyzer(
                self.model,
                prompt=self.prompt_for(asset_type),
                attribute_prefix=self.attribute_prefix,
                max_images_per_asset=self.max_images_per_asset,
                generation=self.generation,
                max_workers=self.max_workers,
                rate_limit=self.rate_limit,
                cancel_event=self.cancel_event,
            )
            analyzer.PROVIDER_NAME = f'Verification ({asset_type or "untyped"})'
            logger.info(
                f'Verifying {len(assets)} {asset_type or "untyped"} assets as '
                f'{self.description_for(asset_type)!r} with {self.model_name}.'
            )
            analyzer(PhysicalAssetCollection(assets))

    # ----------------------------------------------------- classifier route
    def _verify_with_classifier(
        self, groups: dict[str | None, list[PhysicalAsset]]
    ) -> None:
        """Call the user's classifier once per asset."""
        assert self.classifier is not None
        prefix = self.attribute_prefix
        for asset_type, assets in groups.items():
            description = self.description_for(asset_type)
            for asset in assets:
                raise_if_cancelled(self.cancel_event, 'detection verification')
                paths = self._selected_paths(asset)
                if not paths:
                    continue
                try:
                    verdicts = list(self.classifier(paths, description))
                except Exception as exc:  # noqa: BLE001 - reported per asset
                    logger.error(f'Classifier failed on asset {asset.id}: {exc}')
                    continue
                parsed = [
                    (
                        bool(normalize_match(match)),
                        normalize_confidence(confidence),
                        label,
                    )
                    for match, confidence, label in verdicts
                ]
                if not parsed:
                    continue
                positives = [v for v in parsed if v[0]]
                pool = positives or parsed
                best = max(pool, key=lambda v: v[1] if v[1] is not None else 1.0)
                asset.attributes[f'{prefix}_match'] = bool(positives)
                asset.attributes[f'{prefix}_confidence'] = best[1]
                asset.attributes[f'{prefix}_label'] = (
                    str(best[2]).strip() if best[2] is not None else None
                )

    # -------------------------------------------------------------- __call__
    def __call__(self, collection: PhysicalAssetCollection) -> PhysicalAssetCollection:
        """
        Verify every asset and remove the ones the model rejects.

        Args:
            collection: The detections to verify (mutated in place).

        Returns:
            PhysicalAssetCollection: The same collection without the rejected
            assets (or with them flagged when ``keep_rejected`` is set).

        Raises:
            OperationCancelled: If ``cancel_event`` is set during the run.
        """
        raise_if_cancelled(self.cancel_event, 'detection verification')
        prefix = self.attribute_prefix
        assets = list(collection)
        if not assets:
            logger.warning('DetectionVerifier: the collection is empty. Skipping.')
            return collection

        groups: dict[str | None, list[PhysicalAsset]] = {}
        unverifiable: list[PhysicalAsset] = []
        for asset in assets:
            self._clear_verdict(asset)
            if self._selected_paths(asset):
                groups.setdefault(asset.asset_type, []).append(asset)
            else:
                unverifiable.append(asset)
        if unverifiable:
            logger.info(
                f'DetectionVerifier: {len(unverifiable)} assets have no downloaded '
                'images and cannot be verified; they are kept.'
            )

        if self.classifier is not None:
            self._verify_with_classifier(groups)
        else:
            self._verify_with_model(groups)
        raise_if_cancelled(self.cancel_event, 'detection verification')

        accepted = rejected = failed = 0
        rejected_ids: list[str] = []
        for asset in unverifiable:
            asset.attributes[f'{prefix}_accepted'] = None
            asset.attributes[f'{prefix}_model'] = self.model_name
        for group in groups.values():
            for asset in group:
                verdict = self._decide(asset)
                asset.attributes[f'{prefix}_accepted'] = verdict
                asset.attributes[f'{prefix}_model'] = self.model_name
                if verdict is None:
                    failed += 1
                elif verdict:
                    accepted += 1
                else:
                    rejected += 1
                    rejected_ids.append(asset.id)

        verified = accepted + rejected
        logger.info(
            f'DetectionVerifier ({self.model_name}): {verified} assets verified, '
            f'{accepted} accepted, {rejected} rejected, '
            f'{failed + len(unverifiable)} unverifiable '
            f'({len(unverifiable)} without images, {failed} failed requests).'
        )
        if rejected_ids and not self.keep_rejected:
            collection.remove(rejected_ids)
            logger.info(
                f'DetectionVerifier: removed {len(rejected_ids)} rejected assets; '
                f'{len(collection)} remain.'
            )
        return collection
