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

import threading

import pytest

from rapidtools.core import OperationCancelled, is_cancelled, raise_if_cancelled

# ==========================================
# 1. OperationCancelled
# ==========================================


def test_operation_cancelled_is_runtime_error():
    """OperationCancelled can be caught as a RuntimeError."""
    assert issubclass(OperationCancelled, RuntimeError)
    with pytest.raises(RuntimeError):
        raise OperationCancelled('stop')


# ==========================================
# 2. raise_if_cancelled
# ==========================================


def test_raise_if_cancelled_noop_when_unset_or_none():
    """Nothing happens without an event or with an unset event."""
    raise_if_cancelled(None)
    raise_if_cancelled(threading.Event(), context='tiling')


def test_raise_if_cancelled_raises_with_context():
    """A set event raises and the context is appended to the message."""
    event = threading.Event()
    event.set()
    with pytest.raises(OperationCancelled, match=r'^Operation cancelled\.$'):
        raise_if_cancelled(event)
    with pytest.raises(OperationCancelled, match='Operation cancelled during batch 3.'):
        raise_if_cancelled(event, context='batch 3')


# ==========================================
# 3. is_cancelled
# ==========================================


def test_is_cancelled():
    """is_cancelled mirrors the event state and is False for None."""
    event = threading.Event()
    assert is_cancelled(None) is False
    assert is_cancelled(event) is False
    event.set()
    assert is_cancelled(event) is True
