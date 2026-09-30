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
Graphical user interface for rapidtools.

Provides a local web application that loads or downloads imagery, detects
assets with SAM 3 (or discovers objects along a Mapillary street survey),
gathers aerial or street-level crops of each asset and runs vision-language-
model inference on them with a custom prompt, without writing any code. A
guided prompt builder with an optional language-model assistant helps write
that prompt. The server is built on the Python standard library, so no extra
dependencies are required.

Example:
    >>> from rapidtools.gui import launch_asset_analysis_app
    >>> launch_asset_analysis_app()  # opens http://localhost:8765/ in a browser
"""

from .notify import JobSummary, NotificationConfig, Notifier
from .prompt_builder import (
    OutputField,
    PromptAssistant,
    PromptSpec,
    RubricEntry,
    assemble_prompt,
)
from .workflow import (
    IMAGERY_SOURCES,
    MODEL_BACKENDS,
    AssetAnalysisWorkflow,
    AssistSettings,
    DetectionResult,
    DetectionSettings,
    InferenceResult,
    InferenceSettings,
    RegionImagerySettings,
    StreetDetectionResult,
    StreetDetectionSettings,
    WorkflowCancelled,
    list_available_models,
    parse_asset_list,
)


def launch_asset_analysis_app(**kwargs) -> None:
    """
    Start the GUI server and open it in the default browser (blocks).

    This is a thin convenience wrapper that imports the server lazily (so
    ``import rapidtools.gui`` stays cheap) and forwards every keyword argument
    to :func:`rapidtools.gui.server.launch_asset_analysis_app`.

    Args:
        **kwargs:
            ``host`` (str), ``port`` (int), ``output_dir`` (str | Path),
            ``open_browser`` (bool), ``token`` (str | None) and ``data_root``
            (str | Path | None). See the server function for details.

    Raises:
        SystemExit: If the requested port cannot be bound.

    Example:
        >>> from rapidtools.gui import launch_asset_analysis_app
        >>>
        >>> # Serve on a custom port without opening a browser tab:
        >>> launch_asset_analysis_app(port=9000, open_browser=False)
    """
    from .server import launch_asset_analysis_app as _launch

    _launch(**kwargs)


__all__ = [
    'IMAGERY_SOURCES',
    'MODEL_BACKENDS',
    'AssetAnalysisWorkflow',
    'AssistSettings',
    'DetectionResult',
    'DetectionSettings',
    'InferenceResult',
    'InferenceSettings',
    'JobSummary',
    'NotificationConfig',
    'Notifier',
    'OutputField',
    'PromptAssistant',
    'PromptSpec',
    'RegionImagerySettings',
    'RubricEntry',
    'StreetDetectionResult',
    'StreetDetectionSettings',
    'WorkflowCancelled',
    'assemble_prompt',
    'launch_asset_analysis_app',
    'list_available_models',
    'parse_asset_list',
]
