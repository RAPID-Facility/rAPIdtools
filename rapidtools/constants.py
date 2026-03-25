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

"""
Definitions for global constants used throughout the rapidtools package.

This module centralizes static reference data, mathematical constants,
and shared mapping dictionaries for consistency across all modules.
"""

EARTH_RADIUS_KM = 6371  # Mean radius of the Earth in km
LATITUDE_SPACING_KM = 111.320  # Approx distance of 1 degree latitude in km

# Dictionary that maps variations, abbreviations, and plurals to a standard key:
UNIT_ALIASES = {
    'px': 'pixels',
    'pixel': 'pixels',
    'pixels': 'pixels',
    'm': 'meters',
    'meter': 'meters',
    'meters': 'meters',
    'metre': 'meters',
    'km': 'kilometers',
    'kilometer': 'kilometers',
    'kilometers': 'kilometers',
    'ft': 'feet',
    'foot': 'feet',
    'feet': 'feet',
    'mi': 'miles',
    'mile': 'miles',
    'miles': 'miles',
    'yd': 'yards',
    'yard': 'yards',
    'yards': 'yards',
    'yds': 'yards',
}

# Conversion factors from meters to the canonical distance units:
METERS_CONVERSION_FACTORS = {
    'meters': 1.0,
    'kilometers': 0.001,
    'feet': 1.0 / 0.3048,
    'yards': 1.0 / 0.9144,
    'miles': 1.0 / 1609.344,
}
