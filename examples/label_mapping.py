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
# 05-26-2026

"""
This script provides an example of the rapidtools dynamic label mapping by:
1. Defining a list of target objects in plain, natural English.
2. Initializing a local Gemma-4 vision-language model.
3. Configuring the MapillaryLabelMapper using the local model.
4. Translating the plain English inputs into exact Mapillary API semantic labels.
"""

import json

from rapidtools import (
    Gemma4Inference,
    MapillaryLabelMapper,
)

# Define the target objects in plain English:
print('\n--- Step 1: Defining Target Objects ---')
user_input = [
    'cars', 
    'people crossing the street', 
    'utility poles',
    'buildings and houses'
]
print('User Input:\n' + '\n'.join(f' - {item}' for item in user_input))

# Initialize the local Gemma model. We use the efficient 2B parameter version
# loaded onto the GPU for rapid processing:
print('\n--- Step 2: Initializing Local LLM ---')
gemma_model = Gemma4Inference(model_id='google/gemma-4-E2B-it')

# Wrap the loaded inference model in the mapper utility:
print('\n--- Step 3: Configuring the Label Mapper ---')
mapper = MapillaryLabelMapper(llm_model=gemma_model)

# Execute semantic translation:
print('\n--- Step 4: Translating to Mapillary syntax ---')

print('Querying Mapillary label schema...')
# The LLM dynamically searches the Mapillary label schema and finds exact matches:
exact_labels = mapper.map_classes(user_input)

# Display the translated list of labels that can now be passed directly 
# into the MapillaryImageExtractor:
print('\nFinal Mapillary API Labels:')
print(json.dumps(exact_labels, indent=4))