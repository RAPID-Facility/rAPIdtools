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
# 02-22-2026

import base64
import logging
import mimetypes
from abc import abstractmethod
from pathlib import Path

from rapidtools.config import REQUESTS_TIMEOUT_VAL, get_configured_session

from .base import BaseInferenceModel


class BaseAPIInferenceModel(BaseInferenceModel):
    """
    Intermediate base class for cloud API models (Gemini, Claude, OpenAI).
    Handles network sessions and image Base64 encoding.
    """

    def __init__(self, max_retries: int = 3, max_workers: int = 10):
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.session = get_configured_session(retries=self.max_retries)

    @staticmethod
    @abstractmethod
    def list_available_models(api_key: str | None = None) -> list[str]:
        """Every API wrapper must implement a way to list valid models."""
        pass

    def _get_mime_type(self, image_path: Path) -> str:
        """Guesses MIME type, prioritizing the child class's MIME_MAP."""
        ext = image_path.suffix.lower()
        child_mime_map = getattr(self, 'MIME_MAP', {})
        if ext in child_mime_map:
            return child_mime_map[ext]
        guess, _ = mimetypes.guess_type(image_path)
        return guess or 'image/jpeg'

    def _fetch_and_encode_image(self, image_input: str | Path) -> dict[str, str] | None:
        """Universal Base64 encoder for ALL API models."""
        input_str = str(image_input)
        image_bytes = None
        mime_type = 'image/jpeg'

        try:
            if input_str.startswith(('http://', 'https://')):
                logging.info(f'Downloading image from URL: {input_str}')
                resp = self.session.get(input_str, timeout=REQUESTS_TIMEOUT_VAL)
                resp.raise_for_status()
                image_bytes = resp.content
                mime_type = resp.headers.get('Content-Type') or self._get_mime_type(Path(input_str))
            else:
                path_obj = Path(image_input)
                if not path_obj.exists():
                    logging.error(f'File not found: {path_obj}')
                    return None
                mime_type = self._get_mime_type(path_obj)
                image_bytes = path_obj.read_bytes()
        except Exception as e:
            logging.error(f'Error preparing image {input_str}: {e}')
            return None

        if not image_bytes:
            return None

        return {
            'mime_type': mime_type,
            'data': base64.b64encode(image_bytes).decode('utf-8')
        }
