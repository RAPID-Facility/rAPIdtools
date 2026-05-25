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
# 05-25-2026

class MapillaryLabels:
    """Auto-generated Mapillary Segmentation Labels"""

    CONSTRUCTION_BARRIER_ACOUSTIC = 'construction--barrier--acoustic'
    CONSTRUCTION_BARRIER_AMBIGUOUS = 'construction--barrier--ambiguous'
    CONSTRUCTION_BARRIER_CONCRETE_BLOCK = 'construction--barrier--concrete-block'
    CONSTRUCTION_BARRIER_CURB = 'construction--barrier--curb'
    CONSTRUCTION_BARRIER_FENCE = 'construction--barrier--fence'
    CONSTRUCTION_BARRIER_GUARD_RAIL = 'construction--barrier--guard-rail'
    CONSTRUCTION_BARRIER_OTHER_BARRIER = 'construction--barrier--other-barrier'
    CONSTRUCTION_BARRIER_ROAD_MEDIAN = 'construction--barrier--road-median'
    CONSTRUCTION_BARRIER_ROAD_SIDE = 'construction--barrier--road-side'
    CONSTRUCTION_BARRIER_SEPARATOR = 'construction--barrier--separator'
    CONSTRUCTION_BARRIER_TEMPORARY = 'construction--barrier--temporary'
    CONSTRUCTION_BARRIER_WALL = 'construction--barrier--wall'
    CONSTRUCTION_FLAT_BIKE_LANE = 'construction--flat--bike-lane'
    CONSTRUCTION_FLAT_CROSSWALK_PLAIN = 'construction--flat--crosswalk-plain'
    CONSTRUCTION_FLAT_CURB_CUT = 'construction--flat--curb-cut'
    CONSTRUCTION_FLAT_DRIVEWAY = 'construction--flat--driveway'
    CONSTRUCTION_FLAT_PARKING = 'construction--flat--parking'
    CONSTRUCTION_FLAT_PARKING_AISLE = 'construction--flat--parking-aisle'
    CONSTRUCTION_FLAT_PEDESTRIAN_AREA = 'construction--flat--pedestrian-area'
    CONSTRUCTION_FLAT_RAIL_TRACK = 'construction--flat--rail-track'
    CONSTRUCTION_FLAT_ROAD = 'construction--flat--road'
    CONSTRUCTION_FLAT_ROAD_SHOULDER = 'construction--flat--road-shoulder'
    CONSTRUCTION_FLAT_SERVICE_LANE = 'construction--flat--service-lane'
    CONSTRUCTION_FLAT_SIDEWALK = 'construction--flat--sidewalk'
    CONSTRUCTION_FLAT_TRAFFIC_ISLAND = 'construction--flat--traffic-island'
    CONSTRUCTION_STRUCTURE_BRIDGE = 'construction--structure--bridge'
    CONSTRUCTION_STRUCTURE_BUILDING = 'construction--structure--building'
    CONSTRUCTION_STRUCTURE_GARAGE = 'construction--structure--garage'
    CONSTRUCTION_STRUCTURE_TUNNEL = 'construction--structure--tunnel'
    HUMAN_PERSON_INDIVIDUAL = 'human--person--individual'
    HUMAN_PERSON_PERSON_GROUP = 'human--person--person-group'
    HUMAN_RIDER_BICYCLIST = 'human--rider--bicyclist'
    HUMAN_RIDER_MOTORCYCLIST = 'human--rider--motorcyclist'
    HUMAN_RIDER_OTHER_RIDER = 'human--rider--other-rider'
    MARKING_CONTINUOUS_DASHED = 'marking--continuous--dashed'
    MARKING_CONTINUOUS_SOLID = 'marking--continuous--solid'
    MARKING_CONTINUOUS_ZIGZAG = 'marking--continuous--zigzag'
    MARKING_DISCRETE_AMBIGUOUS = 'marking--discrete--ambiguous'
    MARKING_DISCRETE_ARROW_AMBIGUOUS = 'marking--discrete--arrow--ambiguous'
    MARKING_DISCRETE_ARROW_LEFT = 'marking--discrete--arrow--left'
    MARKING_DISCRETE_ARROW_OTHER = 'marking--discrete--arrow--other'
    MARKING_DISCRETE_ARROW_RIGHT = 'marking--discrete--arrow--right'
    MARKING_DISCRETE_ARROW_SPLIT_LEFT_OR_RIGHT = 'marking--discrete--arrow--split-left-or-right'
    MARKING_DISCRETE_ARROW_SPLIT_LEFT_OR_RIGHT_OR_STRAIGHT = 'marking--discrete--arrow--split-left-or-right-or-straight'
    MARKING_DISCRETE_ARROW_SPLIT_LEFT_OR_STRAIGHT = 'marking--discrete--arrow--split-left-or-straight'
    MARKING_DISCRETE_ARROW_SPLIT_RIGHT_OR_STRAIGHT = 'marking--discrete--arrow--split-right-or-straight'
    MARKING_DISCRETE_ARROW_STRAIGHT = 'marking--discrete--arrow--straight'
    MARKING_DISCRETE_ARROW_U_TURN = 'marking--discrete--arrow--u-turn'
    MARKING_DISCRETE_CROSSWALK_ZEBRA = 'marking--discrete--crosswalk-zebra'
    MARKING_DISCRETE_GIVE_WAY_ROW = 'marking--discrete--give-way-row'
    MARKING_DISCRETE_GIVE_WAY_SINGLE = 'marking--discrete--give-way-single'
    MARKING_DISCRETE_HATCHED_CHEVRON = 'marking--discrete--hatched--chevron'
    MARKING_DISCRETE_HATCHED_DIAGONAL = 'marking--discrete--hatched--diagonal'
    MARKING_DISCRETE_OTHER_MARKING = 'marking--discrete--other-marking'
    MARKING_DISCRETE_STOP_LINE = 'marking--discrete--stop-line'
    MARKING_DISCRETE_SYMBOL_AMBIGUOUS = 'marking--discrete--symbol--ambiguous'
    MARKING_DISCRETE_SYMBOL_BICYCLE = 'marking--discrete--symbol--bicycle'
    MARKING_DISCRETE_SYMBOL_OTHER = 'marking--discrete--symbol--other'
    MARKING_DISCRETE_SYMBOL_PEDESTRIAN = 'marking--discrete--symbol--pedestrian'
    MARKING_DISCRETE_SYMBOL_WHEELCHAIR = 'marking--discrete--symbol--wheelchair'
    MARKING_DISCRETE_TEXT_ = 'marking--discrete--text--'
    MARKING_DISCRETE_TEXT_AMBIGUOUS = 'marking--discrete--text--ambiguous'
    MARKING_DISCRETE_TEXT_BUS = 'marking--discrete--text--bus'
    MARKING_DISCRETE_TEXT_OTHER = 'marking--discrete--text--other'
    MARKING_DISCRETE_TEXT_SCHOOL = 'marking--discrete--text--school'
    MARKING_DISCRETE_TEXT_SLOW = 'marking--discrete--text--slow'
    MARKING_DISCRETE_TEXT_STOP = 'marking--discrete--text--stop'
    MARKING_DISCRETE_TEXT_TAXI = 'marking--discrete--text--taxi'
    NATURE_BEACH = 'nature--beach'
    NATURE_MOUNTAIN = 'nature--mountain'
    NATURE_SAND = 'nature--sand'
    NATURE_SKY = 'nature--sky'
    NATURE_SNOW = 'nature--snow'
    NATURE_TERRAIN = 'nature--terrain'
    NATURE_VEGETATION = 'nature--vegetation'
    NATURE_WATER = 'nature--water'
    OBJECT_BANNER = 'object--banner'
    OBJECT_BENCH = 'object--bench'
    OBJECT_BIKE_RACK = 'object--bike-rack'
    OBJECT_CATCH_BASIN = 'object--catch-basin'
    OBJECT_CCTV_CAMERA = 'object--cctv-camera'
    OBJECT_FIRE_HYDRANT = 'object--fire-hydrant'
    OBJECT_JUNCTION_BOX = 'object--junction-box'
    OBJECT_MAILBOX = 'object--mailbox'
    OBJECT_MANHOLE = 'object--manhole'
    OBJECT_PARKING_METER = 'object--parking-meter'
    OBJECT_PHONE_BOOTH = 'object--phone-booth'
    OBJECT_POTHOLE = 'object--pothole'
    OBJECT_RAMP = 'object--ramp'
    OBJECT_SIGN_ADVERTISEMENT = 'object--sign--advertisement'
    OBJECT_SIGN_AMBIGUOUS = 'object--sign--ambiguous'
    OBJECT_SIGN_BACK = 'object--sign--back'
    OBJECT_SIGN_INFORMATION = 'object--sign--information'
    OBJECT_SIGN_OTHER = 'object--sign--other'
    OBJECT_SIGN_STORE = 'object--sign--store'
    OBJECT_STREET_LIGHT = 'object--street-light'
    OBJECT_SUPPORT_POLE = 'object--support--pole'
    OBJECT_SUPPORT_POLE_GROUP = 'object--support--pole-group'
    OBJECT_SUPPORT_TRAFFIC_SIGN_FRAME = 'object--support--traffic-sign-frame'
    OBJECT_SUPPORT_UTILITY_POLE = 'object--support--utility-pole'
    OBJECT_TRAFFIC_CONE = 'object--traffic-cone'
    OBJECT_TRAFFIC_LIGHT_AMBIGUOUS = 'object--traffic-light--ambiguous'
    OBJECT_TRAFFIC_LIGHT_CYCLISTS_BACK = 'object--traffic-light--cyclists-back'
    OBJECT_TRAFFIC_LIGHT_CYCLISTS_FRONT = 'object--traffic-light--cyclists-front'
    OBJECT_TRAFFIC_LIGHT_CYCLISTS_SIDE = 'object--traffic-light--cyclists-side'
    OBJECT_TRAFFIC_LIGHT_GENERAL_HORIZONTAL_BACK = 'object--traffic-light--general-horizontal-back'
    OBJECT_TRAFFIC_LIGHT_GENERAL_HORIZONTAL_FRONT = 'object--traffic-light--general-horizontal-front'
    OBJECT_TRAFFIC_LIGHT_GENERAL_HORIZONTAL_SIDE = 'object--traffic-light--general-horizontal-side'
    OBJECT_TRAFFIC_LIGHT_GENERAL_SINGLE_BACK = 'object--traffic-light--general-single-back'
    OBJECT_TRAFFIC_LIGHT_GENERAL_SINGLE_FRONT = 'object--traffic-light--general-single-front'
    OBJECT_TRAFFIC_LIGHT_GENERAL_SINGLE_SIDE = 'object--traffic-light--general-single-side'
    OBJECT_TRAFFIC_LIGHT_GENERAL_UPRIGHT_BACK = 'object--traffic-light--general-upright-back'
    OBJECT_TRAFFIC_LIGHT_GENERAL_UPRIGHT_FRONT = 'object--traffic-light--general-upright-front'
    OBJECT_TRAFFIC_LIGHT_GENERAL_UPRIGHT_SIDE = 'object--traffic-light--general-upright-side'
    OBJECT_TRAFFIC_LIGHT_OTHER = 'object--traffic-light--other'
    OBJECT_TRAFFIC_LIGHT_PEDESTRIANS_BACK = 'object--traffic-light--pedestrians-back'
    OBJECT_TRAFFIC_LIGHT_PEDESTRIANS_FRONT = 'object--traffic-light--pedestrians-front'
    OBJECT_TRAFFIC_LIGHT_PEDESTRIANS_SIDE = 'object--traffic-light--pedestrians-side'
    OBJECT_TRAFFIC_LIGHT_WARNING = 'object--traffic-light--warning'
    OBJECT_TRAFFIC_SIGN_AMBIGUOUS = 'object--traffic-sign--ambiguous'
    OBJECT_TRAFFIC_SIGN_BACK = 'object--traffic-sign--back'
    OBJECT_TRAFFIC_SIGN_DIRECTION_BACK = 'object--traffic-sign--direction-back'
    OBJECT_TRAFFIC_SIGN_DIRECTION_FRONT = 'object--traffic-sign--direction-front'
    OBJECT_TRAFFIC_SIGN_FRONT = 'object--traffic-sign--front'
    OBJECT_TRAFFIC_SIGN_INFORMATION_PARKING = 'object--traffic-sign--information-parking'
    OBJECT_TRAFFIC_SIGN_TEMPORARY_BACK = 'object--traffic-sign--temporary-back'
    OBJECT_TRAFFIC_SIGN_TEMPORARY_FRONT = 'object--traffic-sign--temporary-front'
    OBJECT_TRASH_CAN = 'object--trash-can'
    OBJECT_VEHICLE_BICYCLE = 'object--vehicle--bicycle'
    OBJECT_VEHICLE_BOAT = 'object--vehicle--boat'
    OBJECT_VEHICLE_BUS = 'object--vehicle--bus'
    OBJECT_VEHICLE_CAR = 'object--vehicle--car'
    OBJECT_VEHICLE_CARAVAN = 'object--vehicle--caravan'
    OBJECT_VEHICLE_MOTORCYCLE = 'object--vehicle--motorcycle'
    OBJECT_VEHICLE_ON_RAILS = 'object--vehicle--on-rails'
    OBJECT_VEHICLE_OTHER_VEHICLE = 'object--vehicle--other-vehicle'
    OBJECT_VEHICLE_TRAILER = 'object--vehicle--trailer'
    OBJECT_VEHICLE_TRUCK = 'object--vehicle--truck'
    OBJECT_VEHICLE_VEHICLE_GROUP = 'object--vehicle--vehicle-group'
    OBJECT_VEHICLE_WHEELED_SLOW = 'object--vehicle--wheeled-slow'
    OBJECT_WATER_VALVE = 'object--water-valve'
    OBJECT_WIRE_GROUP = 'object--wire-group'
    VOID_CAR_MOUNT = 'void--car-mount'
    VOID_DYNAMIC = 'void--dynamic'
    VOID_EGO_VEHICLE = 'void--ego-vehicle'
    VOID_GROUND = 'void--ground'
    VOID_STATIC = 'void--static'
    VOID_UNLABELED = 'void--unlabeled'
