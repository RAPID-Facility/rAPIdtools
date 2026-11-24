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

import importlib
import threading

import pytest

from rapidtools.core import OperationCanceled, is_canceled, raise_if_canceled

# ==========================================
# 1. OperationCanceled
# ==========================================


def test_operation_canceled_is_runtime_error():
    """OperationCanceled can be caught as a RuntimeError."""
    assert issubclass(OperationCanceled, RuntimeError)
    with pytest.raises(RuntimeError):
        raise OperationCanceled('stop')


# ==========================================
# 2. raise_if_canceled
# ==========================================


def test_raise_if_canceled_noop_when_unset_or_none():
    """Nothing happens without an event or with an unset event."""
    raise_if_canceled(None)
    raise_if_canceled(threading.Event(), context='tiling')


def test_raise_if_canceled_raises_with_context():
    """A set event raises and the context is appended to the message."""
    event = threading.Event()
    event.set()
    with pytest.raises(OperationCanceled, match=r'^Operation canceled\.$'):
        raise_if_canceled(event)
    with pytest.raises(OperationCanceled, match='Operation canceled during batch 3.'):
        raise_if_canceled(event, context='batch 3')


# ==========================================
# 3. is_canceled
# ==========================================


def test_is_canceled():
    """is_canceled mirrors the event state and is False for None."""
    event = threading.Event()
    assert is_canceled(None) is False
    assert is_canceled(event) is False
    event.set()
    assert is_canceled(event) is True


# ==========================================
# 4. Spellings used before 0.3.0
# ==========================================


@pytest.mark.parametrize(
    ('module_name', 'old', 'new'),
    [
        ('rapidtools', 'OperationCancelled', 'OperationCanceled'),
        ('rapidtools.core', 'OperationCancelled', 'OperationCanceled'),
        ('rapidtools.core', 'raise_if_cancelled', 'raise_if_canceled'),
        ('rapidtools.core', 'is_cancelled', 'is_canceled'),
        ('rapidtools.core.cancellation', 'OperationCancelled', 'OperationCanceled'),
        ('rapidtools.core.cancellation', 'raise_if_cancelled', 'raise_if_canceled'),
        ('rapidtools.core.cancellation', 'is_cancelled', 'is_canceled'),
        ('rapidtools.gui', 'WorkflowCancelled', 'WorkflowCanceled'),
        ('rapidtools.gui.workflow', 'WorkflowCancelled', 'WorkflowCanceled'),
    ],
)
def test_old_spellings_still_resolve_and_warn(module_name, old, new):
    """Names from 0.2.x return the renamed object and say what to use."""
    module = importlib.import_module(module_name)
    with pytest.warns(DeprecationWarning, match=f'{old} is deprecated.*use {new}'):
        assert getattr(module, old) is getattr(module, new)
    assert old not in dir(module)


def test_old_exception_name_catches_the_new_exception():
    """Code written as ``except OperationCancelled`` keeps catching a stop."""
    event = threading.Event()
    event.set()
    with pytest.warns(DeprecationWarning):
        from rapidtools.core import OperationCancelled, raise_if_cancelled
    with pytest.raises(OperationCancelled, match='Operation canceled'):
        raise_if_cancelled(event)


@pytest.mark.parametrize(
    'module_name',
    ['rapidtools.core', 'rapidtools.core.cancellation', 'rapidtools.gui.workflow'],
)
def test_unknown_names_still_raise_attribute_error(module_name):
    """The alias hook does not swallow lookups of names that never existed."""
    module = importlib.import_module(module_name)
    with pytest.raises(AttributeError, match='no_such_name'):
        module.no_such_name  # noqa: B018
