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
# Author:
# Barbaros Cetiner
#
# Last updated:
# 05-25-2026

import logging
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers import logging as hf_logging

from .base import ModelOutput
from .local_base import BaseLocalInferenceModel


class Gemma4Inference(BaseLocalInferenceModel):
    """
    Local inference model for Google's Gemma-4 vision-language models.

    This class handles the initialization and inference execution for the Gemma-4
    family of multimodal models (31B, 26B, E4B, and E2B variants). It extends
    `BaseLocalInferenceModel` to leverage robust local image loading and safe
    sequential batch processing to protect against Out-Of-Memory (OOM) errors.
    """

    VALID_MODELS = [
        'google/gemma-4-31B-it',
        'google/gemma-4-26B-A4B-it',
        'google/gemma-4-E4B-it',
        'google/gemma-4-E2B-it',
    ]

    def __init__(
        self,
        model_id: str = 'google/gemma-4-E2B-it',
        device: str = 'auto',
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> None:
        """
        Initialize the processor and multimodal model.

        Args:
            model_id: 
                The Hugging Face model identifier to load. Must be one of
                the officially supported Gemma-4 variants. Defaults to
                'google/gemma-4-E2B-it'.
            device: 
                Device to load the model on (e.g., 'auto', 'cuda', 'cpu').
                Defaults to 'auto'.
            temperature: 
                Sampling temperature for text generation. Defaults to 0.4.
            max_tokens: 
                Maximum number of new tokens to generate. Defaults to 2048.
        """
        # Suppress HuggingFace HTTP and warning messages:
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        hf_logging.set_verbosity_error()

        super().__init__(
            device=device,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        self.model_id = model_id
        
        if self.model_id not in self.VALID_MODELS:
            logging.warning(
                f'Model ID {self.model_id} not in officially supported Gemma-4 list.'
            )

        # Display custom logs:
        logging.info(f'Loading processor for {self.model_id}...')
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        logging.info(f'Loading model {self.model_id} (this may take a while)...')
        
        # Safely handle the device placement to force it onto the GPU if requested
        if self.device == 'auto':
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id,
                torch_dtype='auto',
                device_map='auto',
            )
        else:
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id,
                torch_dtype='auto',
            ).to(self.device)
            
        self.model.eval()

    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str,
        **kwargs,
    ) -> ModelOutput | None:
        """
        Run multimodal inference using Gemma-4 on a single prompt and image(s).

        This method loads the provided images using the base class's robust PIL
        loader, applies the Gemma-4 chat template, and generates a response.

        Args:
            image_inputs: 
                A single path/URL or a list of paths/URLs pointing to
                the target images.
            prompt: 
                The text prompt or instruction to accompany the images.
            **kwargs: 
                Optional generation parameters to override class defaults
                (e.g., `temperature`, `max_tokens`).

        Returns:
            ModelOutput | None: 
                A standardized output object containing the generated text and
                raw response dictionary, or None if inference fails.
        """
        # Handle None or empty inputs gracefully
        if image_inputs is None:
            image_inputs = []
        elif not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        content_list = []

        # Load images if any were provided:
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                content_list.append({'type': 'image', 'image': pil_img})
            else:
                logging.warning(f'Skipping failed image: {img_input}')

        # ONLY abort if the user *tried* to pass images but they all failed to load.
        # If they passed an empty list on purpose (text-only prompt), let it proceed!
        if len(image_inputs) > 0 and not content_list:
            logging.error('No valid images loaded. Cannot proceed with inference.')
            return None

        # Append the text prompt:
        content_list.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': content_list}]

        try:
            # Process inputs via chat template:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt',
            ).to(self.model.device)

            input_len = inputs['input_ids'].shape[-1]

            # Prepare generation parameters:
            gen_kwargs = {
                'max_new_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'do_sample': kwargs.get('temperature', self.temperature) > 0.0,
            }

            # Generate output:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode generated tokens:
            response_text = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

            # Safely parse response
            if hasattr(self.processor, 'parse_response'):
                parsed = self.processor.parse_response(response_text, prefix="")
            else:
                parsed = response_text

            return ModelOutput(
                text=response_text.strip(),
                raw_response={
                    'generated_text': response_text,
                    'parsed_response': parsed,
                },
            )

        except Exception as e:
            logging.error(f'Inference failed for {self.model_id}: {e}')
            return None
        
    def run_inference_batch(
        self,
        batch_image_inputs: list[list[str | Path]],
        batch_prompts: list[str],
        **kwargs,
    ) -> list[ModelOutput | None]:
        """
        Run batched multimodal inference using Gemma-4 on multiple prompts and image sets.

        This method processes a batch of assets simultaneously, significantly improving 
        throughput on GPUs with high VRAM. It robustly handles image loading failures 
        by returning `None` for invalid entries while successfully processing the rest 
        of the batch. The returned list aligns perfectly with the input indices.

        Args:
            batch_image_inputs: 
                A list of batches, where each element is a list of image paths or URLs 
                associated with a single asset (e.g., `[[img1, img2], [img3], ...]`).
            batch_prompts: 
                A list of text prompts corresponding to each asset in the batch. Must 
                be the same length as `batch_image_inputs`.
            **kwargs: 
                Optional generation parameters to override class defaults for this batch 
                (e.g., `temperature`, `max_tokens`).

        Returns:
            list[ModelOutput | None]: 
                A list of standardized output objects containing the generated text 
                and raw response dictionary. The length and order of the list match 
                the input batches. If an individual asset fails to process (e.g., due 
                to invalid images), its corresponding index will contain `None`.
        """
        batch_messages = []
        
        for img_inputs, prompt in zip(batch_image_inputs, batch_prompts):
            content_list = []
            for img_input in img_inputs:
                pil_img = self._load_image_as_pil(img_input)
                if pil_img:
                    content_list.append({'type': 'image', 'image': pil_img})
            
            if not content_list:
                # If no valid images were loaded for this asset, append None 
                # so we can maintain index alignment for the final output.
                batch_messages.append(None)
                continue
                
            content_list.append({'type': 'text', 'text': prompt})
            batch_messages.append([{'role': 'user', 'content': content_list}])

        # Filter out invalid entries to avoid processor crashes, but keep track
        # of their original indices to reconstruct the full list later.
        valid_indices = [i for i, msg in enumerate(batch_messages) if msg is not None]
        valid_messages = [batch_messages[i] for i in valid_indices]

        # If all items in the batch failed to load images, return a list of Nones
        if not valid_messages:
            return [None] * len(batch_messages)

        try:
            inputs = self.processor.apply_chat_template(
                valid_messages,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
                padding=True,
                add_generation_prompt=True,
            ).to(self.model.device)

            input_len = inputs['input_ids'].shape[-1]
            
            # Prepare generation parameters (overrides default via kwargs):
            gen_kwargs = {
                'max_new_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'do_sample': kwargs.get('temperature', self.temperature) > 0.0,
            }

            # Generate output for the entire valid batch simultaneously:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Safely decode the valid outputs one by one to avoid 2D tensor slicing errors:
            decoded_responses = []
            for single_output in outputs:
                # single_output is 1D, so we can safely slice it using [input_len:]
                response_text = self.processor.decode(
                    single_output[input_len:], skip_special_tokens=True
                )
                decoded_responses.append(response_text)

            # Reconstruct the results list, placing None where assets failed
            results = [None] * len(batch_messages)
            for valid_idx, response_text in zip(valid_indices, decoded_responses):
                parsed = getattr(self.processor, 'parse_response', lambda x: x)(response_text)
                results[valid_idx] = ModelOutput(
                    text=response_text.strip(),
                    raw_response={'generated_text': response_text, 'parsed_response': parsed},
                )
            return results

        except Exception as e:
            logging.error(f'Batched inference failed: {e}')
            return [None] * len(batch_messages)
