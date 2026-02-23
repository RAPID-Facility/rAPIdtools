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

# Important: These require `pip install torch transformers accelerate pillow`
# For load_in_4bit=True, you also need `pip install bitsandbytes`
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

from .base import ModelOutput
from .local_base import BaseLocalInferenceModel


class LlamaVisionInference(BaseLocalInferenceModel):
    """
    Implementation for Meta's Llama 3.2 Vision models using Hugging Face Transformers.
    Specifically uses the Mllama architecture designed for cross-attention vision tasks.
    """

    def __init__(
        self,
        model_id: str = 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        device: str = 'auto',
        load_in_4bit: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048
    ):
        """
        Initializes the Llama Vision model and loads it into GPU/CPU memory.

        Args:
            model_id: The Hugging Face repo ID.
            device: e.g., "cuda", "cpu", or "auto" (distributes across available GPUs).
            load_in_4bit: If True, uses BitsAndBytes to aggressively compress the model
                          so large models (like 11B) can fit on consumer GPUs (e.g., RTX 3090/4090).
        """
        super().__init__(device=device, temperature=temperature, max_tokens=max_tokens)
        self.model_id = model_id

        logging.info(f'Loading processor and weights for {self.model_id}. This may take a moment...')

        # 1. Load the Processor (Handles tokenization and image resizing/normalization)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # 2. Configure Model Loading & VRAM Management
        model_kwargs = {'device_map': self.device}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            # Default to half-precision for massive speed and memory savings over float32
            model_kwargs['torch_dtype'] = torch.float16

        # 3. Load the specific Mllama Architecture
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        # Ensure model is strictly in inference mode (disables dropout layers, etc.)
        self.model.eval()
        logging.info(f'Llama model {self.model_id} loaded successfully.')

    @staticmethod
    def list_available_models(auth_key: str | None = None) -> list[str]:
        """
        Returns a list of supported Hugging Face repository IDs for Llama Vision models.
        (Note: Downloading these from HF requires accepting Meta's license agreement first).
        """
        return sorted([
            # The 11-Billion parameter models (Great for 24GB VRAM GPUs)
            'meta-llama/Llama-3.2-11B-Vision-Instruct',
            'meta-llama/Llama-3.2-11B-Vision',

            # The massive 90-Billion parameter models (Requires multiple GPUs or heavy quantization)
            'meta-llama/Llama-3.2-90B-Vision-Instruct',
            'meta-llama/Llama-3.2-90B-Vision',
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
        Executes a forward pass of the Llama Vision model and returns the universal ModelOutput.
        """

        # 1. Resolve prompt (using universal helper from BaseInferenceModel)
        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # 2. Load all images as PIL Objects (using helper from BaseLocalInferenceModel)
        loaded_images = []
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                loaded_images.append(pil_img)

        if not loaded_images:
            logging.error(f'{log_ctx} No valid images could be loaded into PIL.')
            return None

        # Hack for open-source models: Since they don't have a strict API 'json_mode' flag,
        # we enforce it via the prompt instruction if requested.
        if json_mode:
            prompt_str += '\n\nYou must respond with only valid JSON and no other conversational text.'

        # 3. Format the payload for Llama's specific chat template
        content_block = [{'type': 'image'} for _ in loaded_images]
        if prompt_str:
            content_block.append({'type': 'text', 'text': prompt_str})

        messages = [
            {'role': 'user', 'content': content_block}
        ]

        # 4. Process into PyTorch Tensors
        try:
            # Apply Llama's specific <|image|> and <|start_header_id|> tags
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=text_prompt,
                images=loaded_images,
                return_tensors='pt'
            )

            # Move tensors to the exact device the model weights are currently on
            inputs = inputs.to(self.model.device)

        except Exception as e:
            logging.error(f'{log_ctx} Failed to process Llama tensors: {e}')
            return None

        # 5. Neural Network Generation
        final_temp = temperature if temperature is not None else self.temperature
        final_tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            # Context manager disables gradients to save massive amounts of VRAM
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=final_tokens,
                    temperature=final_temp,
                    do_sample=(final_temp > 0.0) # Greedy decoding if temp == 0
                )

            # 6. Decode the raw ID numbers back into human text
            # Slice [inputs.input_ids.shape[1]:] to only include the *newly generated* tokens,
            # effectively stripping the prompt out of the final string.
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            extracted_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

            # Return the unified ModelOutput!
            return ModelOutput(
                text=extracted_text,
                # Store the raw tensor outputs in case advanced debugging is needed later
                raw_response={'output_ids': output_ids.cpu().tolist()}
            )

        except torch.OutOfMemoryError:
            logging.error(f'{log_ctx} GPU OUT OF MEMORY ERROR. Try smaller images or setting load_in_4bit=True.')
            return None
        except Exception as e:
            logging.error(f'{log_ctx} Unexpected tensor generation error: {e}')
            return None
