"""Tests for log / progress capture used by the rapidtools GUI."""

import logging

from rapidtools.gui.logging_utils import CallbackLogHandler, LogBuffer, ProgressStream


def test_progress_stream_splits_progress_and_lines():
    events = []
    stream = ProgressStream(
        on_line=lambda t: events.append(('log', t)),
        on_progress=lambda t: events.append(('progress', t)),
        on_progress_end=lambda: events.append(('end', '')),
    )
    stream.write('Scanning: 0%\rScanning: 50%')
    stream.write('\rScanning: 100%\n')
    stream.write('plain line\n')
    stream.write('partial')
    stream.write(' line\n')
    assert events == [
        ('progress', 'Scanning: 0%'),
        ('progress', 'Scanning: 50%'),
        ('progress', 'Scanning: 100%'),
        ('end', ''),
        ('log', 'plain line'),
        ('log', 'partial line'),
    ]
    assert not stream.isatty()
    stream.flush()


def test_log_buffer_replaces_live_progress_line():
    buf = LogBuffer()
    buf.add_line('start')
    buf.set_progress('10%')
    buf.set_progress('20%')
    assert [e['text'] for e in buf.since(0)] == ['start', '20%']
    assert buf.since(0)[-1]['kind'] == 'progress'

    buf.add_line('next')  # finalizes the progress line
    entries = buf.since(0)
    assert [e['text'] for e in entries] == ['start', '20%', 'next']
    assert all(e['kind'] == 'log' for e in entries)

    # Polling with the last seen seq returns only newer entries:
    seq = entries[1]['seq']
    assert [e['text'] for e in buf.since(seq)] == ['next']
    buf.clear()
    assert buf.since(0) == []


def test_log_buffer_trims_and_keeps_progress_index():
    buf = LogBuffer(max_entries=3)
    for i in range(5):
        buf.add_line(f'line {i}')
    buf.set_progress('p')
    assert [e['text'] for e in buf.since(0)] == ['line 3', 'line 4', 'p']
    buf.set_progress('q')
    assert [e['text'] for e in buf.since(0)] == ['line 3', 'line 4', 'q']


def test_callback_log_handler_formats_records():
    received = []
    handler = CallbackLogHandler(received.append)
    log = logging.getLogger('rapidtools.gui.test')
    log.addHandler(handler)
    try:
        log.warning('careful')
    finally:
        log.removeHandler(handler)
    assert len(received) == 1
    assert received[0].endswith('WARNING: careful')


# --------------------------------------------------------------- fallback
class _RecordingStream:
    """Minimal stream that records writes and flushes."""

    def __init__(self):
        self.written = []
        self.flushed = 0

    def write(self, text):
        self.written.append(text)

    def flush(self):
        self.flushed += 1


class _BrokenStream:
    """Stream whose write and flush always fail."""

    def write(self, text):
        raise OSError('closed')

    def flush(self):
        raise OSError('closed')


def test_progress_stream_mirrors_to_fallback():
    """Every write and flush is forwarded to the fallback stream."""
    fallback = _RecordingStream()
    events = []
    stream = ProgressStream(
        on_line=events.append,
        on_progress=events.append,
        on_progress_end=lambda: events.append('END'),
        fallback=fallback,
    )
    assert stream.write('hello\n') == 6
    stream.flush()
    assert fallback.written == ['hello\n']
    assert fallback.flushed == 1
    assert events == ['hello']


def test_progress_stream_ignores_broken_fallback():
    """Errors raised by the fallback stream never propagate to the writer."""
    events = []
    stream = ProgressStream(
        on_line=events.append,
        on_progress=events.append,
        on_progress_end=lambda: events.append('END'),
        fallback=_BrokenStream(),
    )
    assert stream.write('50%\r') == 4
    stream.flush()  # must not raise
    assert events == ['50%']


def test_progress_stream_blank_segments_are_dropped():
    """Whitespace-only lines and carriage-return segments emit nothing."""
    events = []
    stream = ProgressStream(
        on_line=events.append,
        on_progress=events.append,
        on_progress_end=lambda: events.append('END'),
    )
    stream.write('   \n\r\r  \n')
    assert events == ['END']


def test_log_buffer_seq_and_end_progress_without_progress():
    """seq tracks the latest entry and end_progress is a no-op with no bar."""
    buf = LogBuffer()
    assert buf.seq == 0
    buf.end_progress()
    buf.add_line('a', kind='log')
    assert buf.seq == 1
    buf.set_progress('p')
    assert buf.seq == 2
    buf.end_progress()
    assert [e['kind'] for e in buf.since(0)] == ['log', 'log']
    buf.clear()
    assert buf.since(0) == [] and buf.seq == 2
    buf.set_progress('fresh')  # a new bar can start after clearing
    assert [e['text'] for e in buf.since(0)] == ['fresh']
