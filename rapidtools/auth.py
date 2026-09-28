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
Opt-in authentication helpers.

Earlier releases logged in to the Hugging Face Hub as a side effect of
``import rapidtools``. Authentication is now explicit: call :func:`login`
yourself, or let the local model wrappers call
:func:`ensure_huggingface_login` right before they download gated weights.

Example:
    >>> import rapidtools
    >>> rapidtools.login()            # uses the cached token or the registry token
    True
    >>> rapidtools.login(token='hf_...')   # explicit token
    True
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import get_token
from huggingface_hub import login as hf_login

from .datasets import download_dataset

logger = logging.getLogger(__name__)


def login(token: str | None = None, dataset_id: str = 'hf_token') -> bool:
    """
    Authenticate this machine with the Hugging Face Hub.

    Resolution order: an explicit ``token``, a token already cached by
    ``huggingface-cli login``, then the token file published in the
    rapidtools dataset registry (downloaded, used, and deleted).

    Args:
        token: Hugging Face access token. When given it is cached for future
            sessions.
        dataset_id: Registry entry holding the fallback token file.

    Returns:
        bool: ``True`` when a token is available afterwards, ``False`` when
        every source failed (the failure is logged, never raised).

    Example:
        >>> from rapidtools.auth import login
        >>> login(token='hf_example')
        True
    """
    if token:
        try:
            hf_login(token=token.strip(), add_to_git_credential=False)
            return True
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            logger.error(f'Hugging Face login failed: {exc}')
            return False

    if get_token() is not None:
        return True

    try:
        [token_path] = download_dataset([dataset_id])
        token_file = Path(token_path)
        hf_token = token_file.read_text().strip()
        hf_login(token=hf_token, add_to_git_credential=False)
        token_file.unlink()
        logger.info('Hugging Face auto-authentication successful.')
        return True
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        logger.error(f'Failed to auto-authenticate with Hugging Face: {exc}')
        return False


def ensure_huggingface_login() -> bool:
    """
    Log in only when no Hugging Face token is cached yet.

    Local model wrappers call this before loading weights so that gated
    repositories (Gemma, Llama, ...) download without manual setup.

    Returns:
        bool: ``True`` when a token is available.

    Example:
        >>> from rapidtools.auth import ensure_huggingface_login
        >>> ensure_huggingface_login()
        True
    """
    if get_token() is not None:
        return True
    return login()
