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
Universal contracts shared by every rapidtools inference model.

:class:`ModelOutput` is the single return type of all models (cloud APIs and
local checkpoints alike) and :class:`BaseInferenceModel` defines the
``run_inference`` / ``run_batch`` interface that pipeline components rely on.

Example:
    >>> from rapidtools.models.base import ModelOutput
    >>> out = ModelOutput(text='CHS Level: 3')
    >>> out.has_text, out.has_masks, out.has_bounding_boxes
    (True, False, False)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Text-generation options shared by every vision-language wrapper.

    Attributes:
        temperature: Sampling temperature; ``None`` keeps the model default.
        max_tokens: Maximum number of generated tokens; ``None`` keeps the
            model default.
        json_mode: Ask for a JSON-only answer (native flag on APIs that
            support it, prompt instruction otherwise).
        system_instruction: System prompt override for this call.

    Example:
        >>> from rapidtools.models import GenerationConfig
        >>> GenerationConfig(temperature=0.0, json_mode=True).json_mode
        True
    """

    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool = False
    system_instruction: str | None = None

    def merged(self, **overrides: Any) -> GenerationConfig:
        """
        Return a copy with the non-``None`` ``overrides`` applied.

        Example:
            >>> GenerationConfig(temperature=0.4).merged(max_tokens=5).max_tokens
            5
        """
        values = {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'json_mode': self.json_mode,
            'system_instruction': self.system_instruction,
        }
        for key, value in overrides.items():
            if key not in values:
                raise TypeError(f'Unknown generation option {key!r}.')
            if value is not None:
                values[key] = value
        return GenerationConfig(**values)


@dataclass
class SegmentationConfig:
    """
    Options for promptable segmentation models such as SAM 3.

    Attributes:
        threshold: Detection confidence threshold.
        mask_threshold: Threshold for binarising mask logits.

    Example:
        >>> from rapidtools.models.base import SegmentationConfig
        >>> SegmentationConfig(threshold=0.3).mask_threshold
        0.5
    """

    threshold: float = 0.5
    mask_threshold: float = 0.5


@dataclass
class ModelOutput:
    """
    A universal container for multimodal AI model outputs.

    Attributes:
        text: Generated text (vision-language models).
        masks: Segmentation masks, e.g. a list of NumPy arrays for SAM 3.
        bounding_boxes: Detected boxes as ``[x0, y0, x1, y1]`` lists.
        raw_response: The provider's raw JSON or the raw tensors/metadata of
            a local model, kept for debugging.
        error: Why the request produced no output (``None`` on success).
            Transient failures (rate limits, timeouts) are still reported as
            a plain ``None`` return from ``run_inference``; this field marks
            responses the provider actively rejected.
        retryable: Whether the same request could succeed later. Provider
            rejections (refusals, safety blocks, 4xx errors) set it to
            ``False`` so callers do not pause and retry them.

    Example:
        >>> from rapidtools.models.base import ModelOutput
        >>> out = ModelOutput(text='  ', bounding_boxes=[[0, 0, 4, 4]])
        >>> out.has_text
        False
        >>> out.has_bounding_boxes
        True
        >>> ModelOutput.failure('refused').retryable
        False
    """

    text: str | None = None
    masks: Any | None = None
    bounding_boxes: list | None = None
    raw_response: Any | None = None
    #: Why the request produced no usable output; ``None`` on success.
    error: str | None = None
    #: Whether sending the same request again could succeed. ``False`` for
    #: provider rejections (refusals, safety blocks, 4xx errors).
    retryable: bool = True

    @classmethod
    def failure(
        cls, error: str, *, retryable: bool = False, raw_response: Any | None = None
    ) -> ModelOutput:
        """Build an output that records why the request failed."""
        return cls(error=error, retryable=retryable, raw_response=raw_response)

    @property
    def failed(self) -> bool:
        """``True`` when the request failed (see :attr:`error`)."""
        return self.error is not None

    @property
    def has_text(self) -> bool:
        """``True`` when ``text`` contains non-whitespace characters."""
        return self.text is not None and len(self.text.strip()) > 0

    @property
    def has_masks(self) -> bool:
        """``True`` when ``masks`` is set."""
        return self.masks is not None

    @property
    def has_bounding_boxes(self) -> bool:
        """``True`` when at least one bounding box is present."""
        return self.bounding_boxes is not None and len(self.bounding_boxes) > 0


_PROMPT_FILE_SUFFIXES = ('.txt', '.md', '.prompt', '.text')


def looks_like_prompt_file(prompt: str) -> bool:
    """
    Return ``True`` when a prompt string is almost certainly meant as a file path.

    A single line ending in a text-file suffix (``.txt``, ``.md``, ...) with no
    spaces is treated as a path, so a typo in the path raises instead of being
    sent to the model as the prompt itself.

    Example:
        >>> looks_like_prompt_file('prompts/aerial_CHS_prompts.txt')
        True
        >>> looks_like_prompt_file('Rate the damage from 0 to 5.')
        False
    """
    stripped = prompt.strip()
    return (
        0 < len(stripped) < 260
        and '\n' not in stripped
        and ' ' not in stripped
        and stripped.lower().endswith(_PROMPT_FILE_SUFFIXES)
    )


def resolve_prompt_text(prompt: str | Path) -> str:
    """
    Return the prompt text, reading it from disk when ``prompt`` is a file.

    Args:
        prompt: Prompt text, or a path (``str`` or :class:`pathlib.Path`) to a
            UTF-8 text file containing it.

    Returns:
        str: The prompt text (file contents are stripped of surrounding
        whitespace).

    Raises:
        FileNotFoundError: If ``prompt`` is a :class:`pathlib.Path`, or a
            string that looks like a path to a text file, and the file does
            not exist. Relative paths are resolved against the current
            working directory, which is included in the message.

    Example:
        >>> resolve_prompt_text('Describe the roof.')
        'Describe the roof.'
        >>> resolve_prompt_text(Path('aerial_CHS_prompts.txt'))[:4]
        'TASK'
    """
    if isinstance(prompt, Path):
        if not prompt.is_file():
            raise FileNotFoundError(
                f'Prompt file not found: {prompt} (resolved against {Path.cwd()}).'
            )
        return prompt.read_text(encoding='utf-8').strip()

    prompt_str = str(prompt)
    try:
        prompt_path = Path(prompt_str)
        if prompt_path.is_file():
            return prompt_path.read_text(encoding='utf-8').strip()
    except OSError:
        return prompt_str
    if looks_like_prompt_file(prompt_str):
        raise FileNotFoundError(
            f'Prompt file not found: {prompt_str} (resolved against {Path.cwd()}). '
            'Pass the prompt text itself, or an existing path.'
        )
    return prompt_str


class BaseInferenceModel(ABC):
    """
    The top-level contract for all AI models (API or local).

    Subclasses implement :meth:`run_inference`; :meth:`run_batch` provides a
    threaded default suitable for network-bound API models (local models
    override it to run sequentially).

    Example:
        A minimal fake model useful in tests and pipelines:

        >>> from rapidtools.models.base import BaseInferenceModel, ModelOutput
        >>> class EchoModel(BaseInferenceModel):
        ...     model_id = 'echo'
        ...     def run_inference(self, image_inputs, prompt, **kwargs):
        ...         return ModelOutput(text=f'{len(image_inputs)} image(s): {prompt}')
        >>> EchoModel().run_inference(['a.jpg'], 'hi').text
        '1 image(s): hi'
    """

    @staticmethod
    def _resolve_prompt(prompt: str | Path) -> str:
        """
        Resolve a prompt given as text or as a path to a text file.

        Args:
            prompt: The prompt itself, or a path to a UTF-8 file containing it.

        Returns:
            str: The prompt text (file contents are stripped of whitespace).

        Example:
            >>> from rapidtools.models.base import BaseInferenceModel
            >>> BaseInferenceModel._resolve_prompt('Describe the image.')
            'Describe the image.'
        """
        try:
            return resolve_prompt_text(prompt)
        except FileNotFoundError as exc:
            # Models stay lenient (the analyzer already validated its prompt),
            # but a filename sent as the prompt is almost always a mistake.
            logger.warning(f'{exc} Sending the string itself as the prompt.')
            return str(prompt)

    def _resolve_generation(
        self, config: GenerationConfig | None = None, **overrides: Any
    ) -> GenerationConfig:
        """
        Combine instance defaults, a ``GenerationConfig`` and call overrides.

        Precedence (lowest to highest): the wrapper's constructor defaults
        (``temperature``, ``max_tokens``, ``system_instruction``), ``config``,
        then explicit keyword overrides that are not ``None``.

        Example:
            >>> from rapidtools.models import GenerationConfig
            >>> model.temperature, model.max_tokens = 0.4, 2048
            >>> cfg = model._resolve_generation(
            ...     GenerationConfig(max_tokens=64), temperature=0.0
            ... )
            >>> cfg.temperature, cfg.max_tokens
            (0.0, 64)
        """
        resolved = GenerationConfig(
            temperature=getattr(self, 'temperature', None),
            max_tokens=getattr(self, 'max_tokens', None),
            json_mode=False,
            system_instruction=getattr(self, 'system_instruction', None),
        )
        if config is not None:
            resolved = resolved.merged(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                json_mode=config.json_mode or None,
                system_instruction=config.system_instruction,
            )
        if 'json_mode' in overrides and overrides['json_mode'] is False:
            overrides['json_mode'] = None  # explicit False never disables a config flag
        return resolved.merged(**overrides)

    @abstractmethod
    def run_inference(
        self, image_inputs: Any, prompt: str | Path, **kwargs
    ) -> ModelOutput | None:
        """
        Execute a single inference pass.

        Args:
            image_inputs: One image path/URL or a list of them.
            prompt: Prompt text or path to a prompt file.
            **kwargs: Provider-specific options (``temperature``,
                ``max_tokens``, ``json_mode``, ...).

        Returns:
            ModelOutput | None: The standardized output, or ``None`` on failure.
        """

    def run_batch(
        self,
        asset_inputs: Iterable[tuple[str, Any]],
        prompt: str | Path,
        **kwargs,
    ) -> Generator[tuple[str, str, ModelOutput | str], None, None]:
        """
        Run :meth:`run_inference` over many assets with a thread pool.

        Uses ``min(self.max_workers, 10)`` threads (default 10 when the model
        defines no ``max_workers``).

        Args:
            asset_inputs: ``(asset_id, image_inputs)`` pairs.
            prompt: Prompt text or file shared by every asset.
            **kwargs: Forwarded to :meth:`run_inference`.

        Yields:
            tuple[str, str, ModelOutput | str]: ``(asset_id, status, payload)``
            where ``status`` is ``'ok'`` (payload is a :class:`ModelOutput`)
            or ``'failed'`` (payload is an error message).

        Example:
            >>> from rapidtools.models import GeminiInference
            >>> model = GeminiInference(api_key='AIza...')
            >>> results = dict(
            ...     (a, out) for a, status, out in model.run_batch(
            ...         [('b1', ['b1.jpg']), ('b2', ['b2.jpg'])], 'Rate damage 0-5.'
            ...     ) if status == 'ok'
            ... )
            >>> sorted(results)
            ['b1', 'b2']
        """
        prompt_str = self._resolve_prompt(prompt)

        def _process(
            asset_id: str, img_inputs: Any
        ) -> tuple[str, str, ModelOutput | str]:
            result = self.run_inference(img_inputs, prompt_str, **kwargs)
            if result is not None and not result.failed:
                return asset_id, 'ok', result
            message = (result.error if result is not None else None) or (
                'Inference failed or was blocked.'
            )
            return asset_id, 'failed', message

        asset_list = list(asset_inputs)
        if not asset_list:
            logger.warning('No assets provided to run_batch.')
            return

        effective_workers = max(1, min(getattr(self, 'max_workers', 10), 10))
        logger.info(
            f'Starting batch of {len(asset_list)} assets using '
            f'{effective_workers} threads.'
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_asset = {
                executor.submit(_process, a_id, imgs): a_id for a_id, imgs in asset_list
            }
            with tqdm(total=len(asset_list), unit='asset', desc='Processing') as pbar:
                for future in as_completed(future_to_asset):
                    asset_id = future_to_asset[future]
                    try:
                        yield future.result()
                    except Exception as e:  # noqa: BLE001 - reported per asset
                        yield asset_id, 'failed', f'Thread Exception: {str(e)}'
                    finally:
                        pbar.update(1)
