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
Cooperative cancellation for long-running rapidtools components.

Pipeline components accept an optional ``cancel_event`` (a
``threading.Event``). They call :func:`raise_if_cancelled` between units of
work (tiles, batches, assets) so that a caller running them on a background
thread can stop them promptly by setting the event.

Example:
    >>> import threading
    >>> from rapidtools import Gemma4AssetAnalyzer
    >>> stop = threading.Event()
    >>> analyzer = Gemma4AssetAnalyzer(prompt='...', cancel_event=stop)
    >>> # from another thread: stop.set()  -> analyzer raises OperationCancelled
"""

from __future__ import annotations

import threading


class OperationCancelled(RuntimeError):
    """
    Raised by a component when its ``cancel_event`` has been set.

    The exception derives from :class:`RuntimeError` so that callers who do
    not care about cancellation specifically can still catch it with a generic
    handler, while callers that do can distinguish a user-requested stop from
    a genuine failure.

    Example:
        >>> import threading
        >>> from rapidtools.core import OperationCancelled, raise_if_cancelled
        >>>
        >>> stop = threading.Event()
        >>> stop.set()
        >>> try:
        ...     raise_if_cancelled(stop, context='tiling')
        ... except OperationCancelled as exc:
        ...     print(exc)
        Operation cancelled during tiling.
    """


def raise_if_cancelled(cancel_event: threading.Event | None, context: str = '') -> None:
    """
    Raise :class:`OperationCancelled` if ``cancel_event`` has been set.

    Components call this between units of work so that a caller running them
    on a background thread can stop them promptly. Passing ``None`` disables
    cancellation entirely, which keeps the call sites free of ``if`` guards.

    Args:
        cancel_event (threading.Event | None):
            The event to inspect. ``None`` means cancellation is not enabled.
        context (str, optional):
            A short description of the work in progress (e.g. ``'tile 3/10'``)
            that is appended to the exception message. Defaults to ``''``.

    Raises:
        OperationCancelled: If ``cancel_event`` is not ``None`` and is set.

    Example:
        >>> import threading
        >>> from rapidtools.core import raise_if_cancelled
        >>>
        >>> stop = threading.Event()
        >>> raise_if_cancelled(stop)  # not set: returns silently
        >>> raise_if_cancelled(None)  # cancellation disabled: returns silently
        >>> stop.set()
        >>> raise_if_cancelled(stop, context='inference')
        Traceback (most recent call last):
            ...
        OperationCancelled: Operation cancelled during inference.
    """
    if cancel_event is not None and cancel_event.is_set():
        suffix = f' during {context}' if context else ''
        raise OperationCancelled(f'Operation cancelled{suffix}.')


def is_cancelled(cancel_event: threading.Event | None) -> bool:
    """
    Return ``True`` if ``cancel_event`` exists and has been set.

    This is the non-raising counterpart of :func:`raise_if_cancelled`, useful
    in loops that want to exit gracefully and keep partial results.

    Args:
        cancel_event (threading.Event | None):
            The event to inspect. ``None`` means cancellation is not enabled.

    Returns:
        bool: ``True`` only when an event was supplied and it is set.

    Example:
        >>> import threading
        >>> from rapidtools.core import is_cancelled
        >>>
        >>> stop = threading.Event()
        >>> is_cancelled(None), is_cancelled(stop)
        (False, False)
        >>> stop.set()
        >>> is_cancelled(stop)
        True
    """
    return cancel_event is not None and cancel_event.is_set()
