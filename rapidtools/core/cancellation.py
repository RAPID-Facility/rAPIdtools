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
``threading.Event``). They call :func:`raise_if_canceled` between units of
work (tiles, batches, assets) so that a caller running them on a background
thread can stop them promptly by setting the event.

Example:
    >>> import threading
    >>> from rapidtools import Gemma4AssetAnalyzer
    >>> stop = threading.Event()
    >>> analyzer = Gemma4AssetAnalyzer(prompt='...', cancel_event=stop)
    >>> # from another thread: stop.set()  -> analyzer raises OperationCanceled
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Mapping
from typing import Any

#: Spellings used before 0.3.0, mapped to the names that replaced them.
RENAMED = {
    'OperationCancelled': 'OperationCanceled',
    'raise_if_cancelled': 'raise_if_canceled',
    'is_cancelled': 'is_canceled',
}


class OperationCanceled(RuntimeError):
    """
    Raised by a component when its ``cancel_event`` has been set.

    The exception derives from :class:`RuntimeError` so that callers who do
    not care about cancellation specifically can still catch it with a generic
    handler, while callers that do can distinguish a user-requested stop from
    a genuine failure.

    Before 0.3.0 this class was spelled ``OperationCancelled``, and the
    helpers ``raise_if_cancelled`` and ``is_cancelled``. Those names still
    work and emit a :class:`DeprecationWarning`.

    Example:
        >>> import threading
        >>> from rapidtools.core import OperationCanceled, raise_if_canceled
        >>>
        >>> stop = threading.Event()
        >>> stop.set()
        >>> try:
        ...     raise_if_canceled(stop, context='tiling')
        ... except OperationCanceled as exc:
        ...     print(exc)
        Operation canceled during tiling.
    """


def raise_if_canceled(cancel_event: threading.Event | None, context: str = '') -> None:
    """
    Raise :class:`OperationCanceled` if ``cancel_event`` has been set.

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
        OperationCanceled: If ``cancel_event`` is not ``None`` and is set.

    Example:
        >>> import threading
        >>> from rapidtools.core import raise_if_canceled
        >>>
        >>> stop = threading.Event()
        >>> raise_if_canceled(stop)  # not set: returns silently
        >>> raise_if_canceled(None)  # cancellation disabled: returns silently
        >>> stop.set()
        >>> raise_if_canceled(stop, context='inference')
        Traceback (most recent call last):
            ...
        OperationCanceled: Operation canceled during inference.
    """
    if cancel_event is not None and cancel_event.is_set():
        suffix = f' during {context}' if context else ''
        raise OperationCanceled(f'Operation canceled{suffix}.')


def is_canceled(cancel_event: threading.Event | None) -> bool:
    """
    Return ``True`` if ``cancel_event`` exists and has been set.

    This is the non-raising counterpart of :func:`raise_if_canceled`, useful
    in loops that want to exit gracefully and keep partial results.

    Args:
        cancel_event (threading.Event | None):
            The event to inspect. ``None`` means cancellation is not enabled.

    Returns:
        bool: ``True`` only when an event was supplied and it is set.

    Example:
        >>> import threading
        >>> from rapidtools.core import is_canceled
        >>>
        >>> stop = threading.Event()
        >>> is_canceled(None), is_canceled(stop)
        (False, False)
        >>> stop.set()
        >>> is_canceled(stop)
        True
    """
    return cancel_event is not None and cancel_event.is_set()


def renamed_attribute(
    module: str, attr: str, renamed: Mapping[str, str], namespace: Mapping[str, Any]
) -> Any:
    """
    Resolve a pre-0.3.0 spelling of a name, with a :class:`DeprecationWarning`.

    Modules that used to export ``OperationCancelled`` and its relatives call
    this from their module-level ``__getattr__``, so code written against
    0.2.x keeps working: the old name returns the very object the new name
    refers to, and ``except OperationCancelled`` still catches the exception.

    Args:
        module (str): Name of the module the attribute was requested from.
        attr (str): The requested attribute.
        renamed (Mapping[str, str]): Old spellings mapped to current names.
        namespace (Mapping[str, Any]): The module's globals.

    Returns:
        Any: The object the current name refers to.

    Raises:
        AttributeError: If ``attr`` is not a renamed name.

    Example:
        >>> import warnings
        >>> import rapidtools.core as core
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter('ignore', DeprecationWarning)
        ...     core.OperationCancelled is core.OperationCanceled
        True
    """
    new_name = renamed.get(attr)
    if new_name is None:
        raise AttributeError(f'module {module!r} has no attribute {attr!r}')
    warnings.warn(
        f'{module}.{attr} is deprecated and will be removed in a future '
        f'release; use {new_name} instead.',
        DeprecationWarning,
        stacklevel=3,
    )
    return namespace[new_name]


def __getattr__(attr: str) -> Any:
    """Keep the pre-0.3.0 spellings importable from this module."""
    return renamed_attribute(__name__, attr, RENAMED, globals())
