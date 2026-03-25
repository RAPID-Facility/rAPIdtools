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
# 03-24-2026

import math

from rapidtools.constants import (
    EARTH_RADIUS_KM,
    LATITUDE_SPACING_KM,
    METERS_CONVERSION_FACTORS,
    UNIT_ALIASES,
)


def test_earth_radius_constant():
    """
    Verify EARTH_RADIUS_KM constant.

    Checks:
    1. Value matches standard integer approximation (6371 km).
    2. Data type is integer.
    """
    # Value check
    assert EARTH_RADIUS_KM == 6371

    # Type check
    assert isinstance(EARTH_RADIUS_KM, int)


def test_latitude_spacing_constant():
    """
    Verify LATITUDE_SPACING_KM constant.

    Checks:
    1. Value matches the standard approximation (111.320 km per degree).
    2. Data type is float.
    """
    # Value check
    assert LATITUDE_SPACING_KM == 111.320

    # Type check (Updated to float)
    assert isinstance(LATITUDE_SPACING_KM, float)


def test_unit_aliases_mapping():
    """
    Verify UNIT_ALIASES dictionary correctly standardizes input strings.
    """
    assert isinstance(UNIT_ALIASES, dict)

    # Spot-check specific conversions
    assert UNIT_ALIASES['px'] == 'pixels'
    assert UNIT_ALIASES['metre'] == 'meters'
    assert UNIT_ALIASES['ft'] == 'feet'
    assert UNIT_ALIASES['mi'] == 'miles'

    # Ensure every single value maps to an expected canonical unit
    expected_canonical_units = {
        'pixels',
        'meters',
        'kilometers',
        'feet',
        'miles',
        'yards',
    }
    for mapped_value in UNIT_ALIASES.values():
        assert mapped_value in expected_canonical_units


def test_meters_conversion_factors():
    """
    Verify METERS_CONVERSION_FACTORS provides mathematically accurate
    multipliers for distance conversions.
    """
    assert isinstance(METERS_CONVERSION_FACTORS, dict)

    # Ensure all distance-based canonical units have a factor
    # (Excluding 'pixels' as it is not a physical real-world distance)
    expected_keys = {'meters', 'kilometers', 'feet', 'miles', 'yards'}
    for key in expected_keys:
        assert key in METERS_CONVERSION_FACTORS

    # Use math.isclose to prevent floating-point precision assertion errors
    assert math.isclose(METERS_CONVERSION_FACTORS['meters'], 1.0)
    assert math.isclose(METERS_CONVERSION_FACTORS['kilometers'], 0.001)
    assert math.isclose(METERS_CONVERSION_FACTORS['feet'], 1.0 / 0.3048)
    assert math.isclose(METERS_CONVERSION_FACTORS['yards'], 1.0 / 0.9144)
    assert math.isclose(METERS_CONVERSION_FACTORS['miles'], 1.0 / 1609.344)
