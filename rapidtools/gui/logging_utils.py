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
Utilities for streaming ``logging`` records and ``tqdm`` progress bars from a
background job into the rapidtools GUI.

Three small pieces cooperate: :class:`CallbackLogHandler` forwards formatted
log records to a callback, :class:`ProgressStream` turns the carriage-return
redraws that ``tqdm`` writes to ``sys.stderr`` into discrete progress events,
and :class:`LogBuffer` stores everything with monotonically increasing sequence
numbers so the browser can poll for "entries newer than N".

Example:
    >>> import logging
    >>> from rapidtools.gui.logging_utils import CallbackLogHandler, LogBuffer
    >>>
    >>> buffer = LogBuffer()
    >>> handler = CallbackLogHandler(buffer.add_line)
    >>> logging.getLogger('demo').addHandler(handler)
    >>> logging.getLogger('demo').warning('careful')
    >>> [e['text'].endswith('WARNING: careful') for e in buffer.since(0)]
    [True]
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

LOG_FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
DATE_FORMAT = '%H:%M:%S'


class CallbackLogHandler(logging.Handler):
    """
    Forward formatted log records to ``callback(text)``.

    Records are formatted with :data:`LOG_FORMAT` (time, level, message) so
    that the GUI log pane mirrors what a terminal user would see.

    Example:
        >>> import logging
        >>> from rapidtools.gui.logging_utils import CallbackLogHandler
        >>>
        >>> lines = []
        >>> handler = CallbackLogHandler(lines.append)
        >>> log = logging.getLogger('demo')
        >>> log.addHandler(handler)
        >>> log.error('boom')
        >>> lines[0].endswith('ERROR: boom')
        True
    """

    def __init__(self, callback: Callable[[str], None]) -> None:
        """
        Initialize the handler.

        Args:
            callback (Callable[[str], None]):
                Function invoked with each formatted log line.
        """
        super().__init__()
        self._callback = callback
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        """
        Format ``record`` and pass it to the callback.

        Any exception raised by the callback is routed to
        :meth:`logging.Handler.handleError` so that logging can never crash a
        running job.

        Args:
            record (logging.LogRecord): The record to forward.
        """
        try:
            self._callback(self.format(record))
        except Exception:  # pragma: no cover - never let logging crash a job
            self.handleError(record)


class ProgressStream:
    """
    Minimal file-like object that captures ``tqdm`` progress bars.

    ``tqdm`` redraws its bar with carriage returns, so text following a ``\\r``
    is reported through ``on_progress`` as a transient message that replaces
    the previous one. The newline that ends a finished bar is reported through
    ``on_progress_end`` so the final state stays in the log. Plain lines are
    reported through ``on_line``.

    Example:
        >>> from rapidtools.gui.logging_utils import ProgressStream
        >>>
        >>> events = []
        >>> stream = ProgressStream(
        ...     on_line=lambda t: events.append(('line', t)),
        ...     on_progress=lambda t: events.append(('progress', t)),
        ...     on_progress_end=lambda: events.append(('end', '')),
        ... )
        >>> _ = stream.write('10%\\r50%\\r100%\\n')
        >>> _ = stream.write('done\\n')
        >>> events[:3]
        [('progress', '10%'), ('progress', '50%'), ('progress', '100%')]
        >>> events[3:]
        [('end', ''), ('line', 'done')]
    """

    def __init__(
        self,
        on_line: Callable[[str], None],
        on_progress: Callable[[str], None],
        on_progress_end: Callable[[], None],
        fallback: Any = None,
    ) -> None:
        """
        Initialize the stream.

        Args:
            on_line (Callable[[str], None]):
                Called with each complete, non-progress line.
            on_progress (Callable[[str], None]):
                Called with the latest state of a live progress bar.
            on_progress_end (Callable[[], None]):
                Called when a progress bar is finished (newline received).
            fallback (Any, optional):
                A real stream (e.g. the original ``sys.stderr``) that also
                receives every write, so the terminal still shows output.
                Errors writing to it are ignored. Defaults to ``None``.
        """
        self._on_line = on_line
        self._on_progress = on_progress
        self._on_progress_end = on_progress_end
        self._fallback = fallback
        self._buffer = ''
        self._in_progress = False

    def write(self, text: str) -> int:
        """
        Consume ``text``, emitting line and progress events as they complete.

        Args:
            text (str): Raw text written by ``tqdm`` or ``print``.

        Returns:
            int: The number of characters consumed (``len(text)``).
        """
        if self._fallback is not None:
            try:
                self._fallback.write(text)
            except Exception:
                pass
        self._buffer += text
        while True:
            cr = self._buffer.find('\r')
            nl = self._buffer.find('\n')
            if cr == -1 and nl == -1:
                break
            if nl != -1 and (cr == -1 or nl < cr):
                line, self._buffer = self._buffer[:nl], self._buffer[nl + 1 :]
                if self._in_progress:
                    if line.strip():
                        self._on_progress(line.rstrip())
                    self._on_progress_end()
                    self._in_progress = False
                elif line.strip():
                    self._on_line(line.rstrip())
            else:
                line, self._buffer = self._buffer[:cr], self._buffer[cr + 1 :]
                if line.strip():
                    self._on_progress(line.rstrip())
                self._in_progress = True
        # Text after the last CR is the live state of the bar; show it now.
        if self._in_progress and self._buffer.strip():
            self._on_progress(self._buffer.rstrip())
            self._buffer = ''
        return len(text)

    def flush(self) -> None:
        """Flush the fallback stream, if any (errors are ignored)."""
        if self._fallback is not None:
            try:
                self._fallback.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        """
        Report that this is not a terminal.

        Returns:
            bool: Always ``False`` so ``tqdm`` uses its plain-text mode.
        """
        return False


class LogBuffer:
    """
    Thread-safe, sequence-numbered log store polled by the web front end.

    Each entry is ``{'seq': int, 'text': str, 'kind': 'log' | 'progress'}``.
    A live progress line is replaced in place until it is finalized. The
    buffer keeps at most ``max_entries`` entries, dropping the oldest first.

    Example:
        >>> from rapidtools.gui.logging_utils import LogBuffer
        >>>
        >>> buf = LogBuffer()
        >>> buf.add_line('start')
        >>> buf.set_progress('10%')
        >>> buf.set_progress('20%')
        >>> [(e['text'], e['kind']) for e in buf.since(0)]
        [('start', 'log'), ('20%', 'progress')]
        >>> buf.end_progress()
        >>> buf.since(0)[-1]['kind']
        'log'
    """

    def __init__(self, max_entries: int = 2000) -> None:
        """
        Initialize an empty buffer.

        Args:
            max_entries (int):
                Maximum number of entries to retain. Defaults to ``2000``.
        """
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._seq = 0
        self._progress_index: int | None = None
        self._max_entries = max_entries

    def _next_seq(self) -> int:
        """Advance and return the sequence counter (caller holds the lock)."""
        self._seq += 1
        return self._seq

    def _trim(self) -> None:
        """Drop the oldest entries beyond ``max_entries`` (caller holds lock)."""
        overflow = len(self._entries) - self._max_entries
        if overflow > 0:
            del self._entries[:overflow]
            if self._progress_index is not None:
                self._progress_index = max(self._progress_index - overflow, 0)

    def add_line(self, text: str, kind: str = 'log') -> None:
        """
        Append a finished line, finalizing any live progress line first.

        Args:
            text (str): The line to store.
            kind (str): Entry kind, normally ``'log'``. Defaults to ``'log'``.
        """
        with self._lock:
            self._finalize_progress_locked()
            self._entries.append({'seq': self._next_seq(), 'text': text, 'kind': kind})
            self._trim()

    def set_progress(self, text: str) -> None:
        """
        Set (or replace) the live progress line.

        Args:
            text (str): The current rendering of the progress bar.
        """
        with self._lock:
            entry = {'seq': self._next_seq(), 'text': text, 'kind': 'progress'}
            if self._progress_index is None:
                self._entries.append(entry)
                self._progress_index = len(self._entries) - 1
            else:
                self._entries[self._progress_index] = entry
            self._trim()

    def end_progress(self) -> None:
        """Convert the live progress line, if any, into a permanent log line."""
        with self._lock:
            self._finalize_progress_locked()

    def _finalize_progress_locked(self) -> None:
        """Mark the live progress entry as a plain log (caller holds lock)."""
        if self._progress_index is not None:
            self._entries[self._progress_index]['kind'] = 'log'
            self._progress_index = None

    @property
    def seq(self) -> int:
        """
        Return the sequence number of the most recent entry.

        Returns:
            int: ``0`` for an empty buffer, otherwise the last assigned seq.
        """
        with self._lock:
            return self._seq

    def since(self, seq: int) -> list[dict[str, Any]]:
        """
        Return entries newer than ``seq`` (a live progress line included).

        Args:
            seq (int): The last sequence number the caller has already seen.

        Returns:
            list[dict[str, Any]]: Copies of the matching entries, oldest first.

        Example:
            >>> from rapidtools.gui.logging_utils import LogBuffer
            >>>
            >>> buf = LogBuffer()
            >>> buf.add_line('a')
            >>> buf.add_line('b')
            >>> [e['text'] for e in buf.since(1)]
            ['b']
        """
        with self._lock:
            return [dict(e) for e in self._entries if e['seq'] > seq]

    def clear(self) -> None:
        """Remove all entries (the sequence counter keeps increasing)."""
        with self._lock:
            self._entries.clear()
            self._progress_index = None
