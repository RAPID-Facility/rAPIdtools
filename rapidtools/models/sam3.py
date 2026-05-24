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
# 03-05-2026

import logging
from pathlib import Path

import torch
from transformers import Sam3Processor, Sam3Model

from .base import ModelOutput
from .local_base import BaseLocalInferenceModel


class SAM3Inference(BaseLocalInferenceModel):
    """
    Universal implementation for SAM 3 (Segment Anything 3).
    Instantiate this class ONCE, then call `run_inference` multiple times 
    to avoid the high overhead of loading the model into memory.
    """

    def __init__(
        self,
        model_id: str = 'facebook/sam3',
        device: str = 'auto',
        load_in_4bit: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 1024
    ):
        super().__init__(
            device=device, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        # Silence noisy loggers:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        
        self.model_id = model_id

        logging.info(f'Loading processor and weights for {self.model_id} (this may take a minute)...')

        # 1. Load the Processor
        self.processor = Sam3Processor.from_pretrained(self.model_id)

        # 2. Configure VRAM Management
        model_kwargs = {
            'torch_dtype': torch.float16
        }
        
        # Ensure device map applies cleanly
        if self.device == 'auto':
            model_kwargs['device_map'] = 'auto'
        else:
            model_kwargs['device_map'] = self.device

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        # 3. Load the correct SAM 3 Model class
        self.model = Sam3Model.from_pretrained(
            self.model_id, 
            **model_kwargs
        )

        self.model.eval()
        logging.info(f'SAM 3 model {self.model_id} loaded successfully.')

    @staticmethod
    def list_available_models(auth_key: str | None = None) -> list[str]:
        """List of supported Hugging Face repository IDs for SAM 3."""
        return sorted([
            'facebook/sam3'
        ])

    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str | Path | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        **kwargs
    ) -> ModelOutput | None:
        """
        Executes a forward pass. Handles single images or batches natively.
        Extracts segmentation masks and bounding boxes.
        """
        prompt_str = self._resolve_prompt(prompt) if prompt is not None else ""
        log_ctx = f"[Prompt: '{prompt_str[:30]}...']" if prompt_str else "[No prompt]"

        # Ensure image_inputs is always a list for batch consistency
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # 1. Load images using the local base class helper
        loaded_images =[]
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                loaded_images.append(pil_img)

        if not loaded_images:
            logging.error(f'{log_ctx} No valid images could be loaded.')
            return None

        # 2. Format inputs for the Processor
        try:
            processor_args = {
                'images': loaded_images,
                'return_tensors': 'pt'
            }
            
            # If a text prompt is passed, apply it to all images in the batch
            if prompt_str:
                processor_args['text'] =[prompt_str] * len(loaded_images)

            inputs = self.processor(**processor_args)

            # Safely move tensors to the correct device
            target_device = self.model.device
            inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }

        except Exception as e:
            logging.error(f'{log_ctx} Failed to process inputs for SAM 3: {e}')
            return None

        # 3. Run Forward Pass & Post-Process
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Map raw logits back to original image dimensions via post_processing
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )

            # 4. Extract Masks and Data for the whole batch
            extracted_masks = []
            extracted_boxes = []
            extracted_scores =[]

            for res in results:
                # Convert PyTorch tensors to numpy arrays/lists
                extracted_masks.append(res['masks'].cpu().numpy())
                
                # Check for bounding boxes or confidence scores
                if 'boxes' in res:
                    extracted_boxes.append(res['boxes'].cpu().numpy().tolist())
                if 'scores' in res:
                    extracted_scores.append(res['scores'].cpu().numpy().tolist())

            # If only a single image was given, unwrap the outer list for a cleaner return format
            if len(loaded_images) == 1:
                extracted_masks = extracted_masks[0]
                extracted_boxes = extracted_boxes[0] if extracted_boxes else None
                extracted_scores = extracted_scores[0] if extracted_scores else None

            # 5. Return the Unified Data Structure
            return ModelOutput(
                text=None,  # Pure SAM models output segments, not conversational text
                masks=extracted_masks,
                bounding_boxes=extracted_boxes,
                raw_response={
                    'status': 'success', 
                    'device': str(self.model.device),
                    'scores': extracted_scores
                }
            )

        except torch.OutOfMemoryError:
            logging.error(
                f'{log_ctx} GPU OUT OF MEMORY ERROR. Try smaller images, smaller batch sizes, '
                'or set load_in_4bit=True.'
            )
            return None
        except Exception as e:
            logging.error(f'{log_ctx} Unexpected SAM 3 generation error: {e}')
            return None