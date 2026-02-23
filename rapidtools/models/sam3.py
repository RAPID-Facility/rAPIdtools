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
# 02-22-2026

import logging
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForVision2Seq, AutoProcessor

from .base import ModelOutput
from .local_base import BaseLocalInferenceModel


class SAM3Inference(BaseLocalInferenceModel):
    """
    Universal implementation for SAM 3 (Segment Anything 3).
    Capable of handling models that output pure segmentation masks,
    as well as multimodal variants ("Omni") that also return text.
    """

    def __init__(
        self,
        model_id: str = 'facebook/sam3-base',
        device: str = 'auto',
        load_in_4bit: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 1024
    ):
        super().__init__(device=device, temperature=temperature, max_tokens=max_tokens)
        self.model_id = model_id

        logging.info(f'Loading processor and weights for {self.model_id}...')

        # 1. Load the Processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # 2. Configure VRAM Management
        model_kwargs = {'device_map': self.device}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            model_kwargs['torch_dtype'] = torch.float16

        # 3. Load the Model
        # We use AutoModelForVision2Seq as the default for multimodal variants,
        # but fallback to standard AutoModel if it is a pure vision SAM.
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
        except ValueError:
            logging.info(f'{self.model_id} is a pure vision model. Falling back to AutoModel.')
            self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)

        self.model.eval()
        logging.info(f'SAM 3 model {self.model_id} loaded successfully.')

    @staticmethod
    def list_available_models(auth_key: str | None = None) -> list[str]:
        """List of supported Hugging Face repository IDs for SAM 3."""
        return sorted([
            'facebook/sam3-base',
            'facebook/sam3-large',
            'facebook/sam3-omni-base',
            'facebook/sam3-omni-large'
        ])

    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str | Path,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> ModelOutput | None:
        """
        Executes a forward pass.
        Extracts segmentation masks and optionally text (if supported by the model).
        """
        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # 1. Load images using the local base class helper
        loaded_images = []
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                loaded_images.append(pil_img)

        if not loaded_images:
            logging.error(f'{log_ctx} No valid images could be loaded.')
            return None

        # 2. Format inputs for the Processor
        try:
            # Note: SAM 3 / Omni models typically expect images and a text prompt
            # to ground the segmentation (e.g., "Segment the rusted areas").
            processor_args = {
                'images': loaded_images,
                'return_tensors': 'pt'
            }
            # Only pass text if a prompt was actually provided
            if prompt_str:
                processor_args['text'] = [prompt_str] * len(loaded_images)

            inputs = self.processor(**processor_args)

            # Move to target device
            inputs = inputs.to(self.model.device)

        except Exception as e:
            logging.error(f'{log_ctx} Failed to process inputs for SAM 3: {e}')
            return None

        final_temp = temperature if temperature is not None else self.temperature
        final_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # 3. Run Generation (Masks + optional Text)
        try:
            with torch.no_grad():
                # If generating text, we need to use `generate`. Pure SAM models might just need a forward pass.
                if hasattr(self.model, 'generate'):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=final_tokens,
                        temperature=final_temp,
                        do_sample=(final_temp > 0.0),
                        return_dict_in_generate=True,
                        output_hidden_states=False
                    )
                else:
                    # Pure vision fallback
                    outputs = self.model(**inputs)

            # 4. Safely Extract Text (Only if the model actually generated sequences)
            extracted_text = None
            if hasattr(outputs, 'sequences') and outputs.sequences is not None:
                generated_ids = outputs.sequences

                # If the model echoes the prompt, slice it off
                if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                    input_length = inputs.input_ids.shape[1]
                    generated_ids = generated_ids[:, input_length:]

                extracted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # 5. Extract Masks
            # Omni-models usually attach the predicted masks to the generation output
            extracted_masks = None
            if hasattr(outputs, 'pred_masks'):
                # Shape is usually (batch_size, num_masks, height, width)
                extracted_masks = outputs.pred_masks.cpu().numpy()
            elif hasattr(outputs, 'masks'):
                extracted_masks = outputs.masks.cpu().numpy()

            # If the model also outputs bounding boxes, extract those too!
            extracted_boxes = None
            if hasattr(outputs, 'pred_boxes'):
                extracted_boxes = outputs.pred_boxes.cpu().numpy().tolist()

            # 6. Return the Unified Data Structure!
            return ModelOutput(
                text=extracted_text,
                masks=extracted_masks,
                bounding_boxes=extracted_boxes,
                raw_response={'status': 'success', 'device': str(self.device)}
            )

        except torch.OutOfMemoryError:
            logging.error(f'{log_ctx} GPU OUT OF MEMORY ERROR. Try smaller images or set load_in_4bit=True.')
            return None
        except Exception as e:
            logging.error(f'{log_ctx} Unexpected SAM 3 generation error: {e}')
            return None
