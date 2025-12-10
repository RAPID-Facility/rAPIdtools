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
# 12-10-2025

import base64
import os
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Custom Exceptions for Better Error Handling ---
class ClaudeClientError(Exception):
    """Base exception for the Claude client."""
    pass

class ImageEncodingError(ClaudeClientError):
    """Raised when an image cannot be encoded."""
    pass

class APIRequestError(ClaudeClientError):
    """Raised when an API request fails."""
    pass

# --- The Claude Client Class ---

class ClaudeClient:
    """Client for interacting with Anthropic's Claude 3 Vision API.

    This class provides convenience methods for:
        * Encoding local image files as base64.
        * Sending single-image vision requests to the Claude API.
        * Running concurrent, batched description tasks over a directory of 
          images with automatic retries and progress reporting.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    
    # Using the latest models with vision capabilities
    SUPPORTED_MODELS = {
        "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet (Newest)",
        "claude-3-opus-20240229": "Claude 3 Opus (Most powerful)",
        "claude-3-sonnet-20240229": "Claude 3 Sonnet (Balanced)",
        "claude-3-haiku-20240307": "Claude 3 Haiku (Fastest)",
    }

    def __init__(
        self,
        api_key: str,
        model_id: str = "claude-3-5-sonnet-20240620",
        max_workers: int = 10,
        max_retries: int = 3,
    ):
        """
        Initialize a ClaudeClient instance.

        Args:
            api_key: 
                Anthropic API key used to authenticate requests.
            model_id: 
                Identifier of the Claude model to use by default when
                generating descriptions. Should typically be one of
                :attr:`SUPPORTED_MODELS`. Defaults to
                ``"claude-3-5-sonnet-20240620"``.
            max_workers: 
                Maximum number of parallel worker threads to use when
                processing images in batch. Defaults to ``10``.
            max_retries: 
                Maximum number of attempts per image in the batch
                workflow (initial attempt plus retry attempts). Defaults
                to ``3``.

        Raises:
            ValueError: If ``api_key`` is empty or missing.
        """
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        if model_id not in self.SUPPORTED_MODELS:
            logging.warning(
                f"Model '{model_id}' is not an official Claude vision model. "
                "It may not work."
            )
            
        self.api_key = api_key
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # Use a single session for connection pooling and to set headers once
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",  # Required by the Anthropic API
            "Content-Type": "application/json",
        })

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        Read an image from disk and returns its base64-encoded string.

        Args:
            image_path: Filesystem path to the image to be encoded.

        Returns:
            str: Base64-encoded representation of the image bytes.

        Raises:
            ImageEncodingError: If the image file cannot be opened or read.
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise ImageEncodingError(
                f"Failed to encode image at {image_path}: {e}"
            ) from e

    def describe_image(
        self, 
        image_path: str, 
        prompt: str,
        max_tokens: int = 1024,
        model_id: str | None = None
    ) -> str:
        """
        Generate a text description for a single image using Claude Vision.

        This method:
            * Encodes the specified image as base64.
            * Builds a Claude API request that includes both the image and the
              provided text prompt.
            * Sends the request and returns the resulting text description.

        Args:
            image_path: 
                Path to the input image file to be described.
            prompt: 
                Natural-language prompt that guides how the model should
                describe the image (e.g., instructions, detail level, focus).
            max_tokens: 
                Maximum number of tokens to generate in the model's
                response. Defaults to ``1024``.
            model_id: 
                Optional override for the model identifier to use. If
                ``None``, the client's default :attr:`model_id` is used.

        Returns:
            str: The model-generated text description for the image.

        Raises:
            ImageEncodingError: 
                If the image cannot be read or base64-encoded.
            APIRequestError: 
                If the HTTP request fails (network/HTTP error) or the API 
                returns an unexpected response format.
        """
        target_model_id = model_id or self.model_id
        base64_image = self._encode_image(image_path)
        
        payload = {
            "model": target_model_id,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        try:
            response = self.session.post(self.API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            error_details = e.response.json().get("error", {}).get(
                "message", "No details provided."
            )
            raise APIRequestError(
                f"API request failed: {e}. Details: {error_details}"
            ) from e
        except (KeyError, IndexError) as e:
            raise APIRequestError(
                f"Unexpected API response format: {data}"
            ) from e

    def batch_describe_directory(
        self,
        image_dir: str,
        prompt: str,
        results_dir: str,
    ) -> dict[str, str]:
        """
        Describe all images in a directory concurrently, with retries.

        This method:
            * Scans the specified input directory for image files.
            * Spawns a thread pool with up to :attr:`max_workers` workers.
            * For each image, calls :meth:`_process_and_save` to generate a
              description and write it to a corresponding ``.txt`` file in the
              results directory.
            * Retries failed images up to :attr:`max_retries` times.
            * Returns a mapping from image filename to a status string.

        Args:
            image_dir: 
                Path to the directory containing images (``.png``,
                ``.jpg``, or ``.jpeg``) to be processed.
            prompt: 
                Text prompt to pass to :meth:`describe_image` for each
                image.
            results_dir: 
                Directory where per-image description text files
                will be written. The directory is created if it does not
                already exist.

        Returns:
            dict[str, str]: 
                A dictionary mapping each image filename to a
                status string. A status of ``"ok"`` indicates success; other
                values (e.g., ``"error: ..."``) describe failure reasons.
        """
        os.makedirs(results_dir, exist_ok=True)
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        final_results: dict[str, str] = {}
        images_to_process = image_files[:]

        for attempt in range(1, self.max_retries + 1):
            if not images_to_process:
                break
            
            logging.info(
                f"--- Starting attempt {attempt}/{self.max_retries} for "
                f"{len(images_to_process)} images ---"
            )
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                with tqdm(
                        total=len(images_to_process),
                        desc=f"Attempt {attempt}"
                    ) as pbar:
                    futures = {
                        executor.submit(
                            self._process_and_save,
                            img,
                            image_dir,
                            prompt,
                            results_dir,
                        ): img
                        for img in images_to_process
                    }
                    for future in as_completed(futures):
                        img_filename = futures[future]
                        try:
                            status = future.result()
                        except Exception as e:
                            status = f"error: {e}"
                        final_results[img_filename] = status
                        pbar.update(1)

            images_to_process = [
                img for img, status in final_results.items() if status != "ok"
            ]
        
        return final_results

    def _process_and_save(
        self,
        image_filename: str,
        image_dir: str,
        prompt: str,
        results_dir: str,
    ) -> str:
        """
        Process a single image file and saves its description to disk.

        This helper is used by :meth:`batch_describe_directory` as a worker
        function within a thread pool. It:
            * Builds the full path to the image file.
            * Calls :meth:`describe_image` to obtain a textual description.
            * Writes the description to a ``.txt`` file in the specified
              results directory.

        Args:
            image_filename: 
                Name of the image file (relative to ``image_dir``) to process.
            image_dir: 
                Directory containing the image file.
            prompt: 
                Prompt passed through to :meth:`describe_image`.
            results_dir: 
                Directory where the generated description file will
                be written.

        Returns:
            str: `
            `"ok"`` if the description was generated and written
            successfully. Any exceptions thrown by :meth:`describe_image`
            or file I/O are propagated to the caller (and handled there).
        """
        image_path = os.path.join(image_dir, image_filename)
        description = self.describe_image(image_path=image_path, prompt=prompt)
        
        output_filename = os.path.splitext(image_filename)[0] + "_description.txt"
        output_path = os.path.join(results_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(description)
        return "ok"