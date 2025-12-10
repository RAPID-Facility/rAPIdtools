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
import time
import requests
import logging
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# API URL Information:
API_BASE_URL = "https://generativelanguage.googleapis.com/v1/models/"
API_SUFFIX = ":generateContent?key="

# Supported Gemini Models:
SUPPORTED_MODELS = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite-preview-06-17": "Gemini 2.5 Flash‑Lite (preview)",
    "gemini-2.5-flash-preview-native-audio-dialog": "Gemini 2.5 Flash Native Audio (dialog)",
    "gemini-2.5-flash-exp-native-audio-thinking-dialog": "Gemini 2.5 Flash Native Audio (thinking dialog)",
    "gemini-2.5-flash-preview-tts": "Gemini 2.5 Flash Preview TTS",
    "gemini-2.5-pro-preview-tts": "Gemini 2.5 Pro Preview TTS",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-preview-image-generation": "Gemini 2.0 Flash Preview Image Generation",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash‑Lite",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini-1.5-flash-8b": "Gemini 1.5 Flash‑8B",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini-embedding-001": "Gemini Embedding",
    "imagen-4.0-generate-preview-06-06": "Imagen 4 (standard)",
    "imagen-4.0-ultra-generate-preview-06-06": "Imagen 4 Ultra",
    "imagen-3.0-generate-002": "Imagen 3",
    "veo-2.0-generate-001": "Veo 2",
    "gemini-live-2.5-flash-preview": "Gemini 2.5 Flash Live",
    "gemini-2.0-flash-live-001": "Gemini 2.0 Flash Live",
    "models/text-embedding-004": "Text Embedding (legacy)",
    "models/aqa": "AQA",
}

# Maximum threads and retries allowed for API queries:
MAX_WORKERS = 10
MAX_RETRIES = 3

class GeminiClient:
    """Handles batch processing of images using the Google Gemini API."""

    def __init__(
        self,
        api_key: str,
        input_dir: str,
        output_dir: str,
        model_id: str = "gemini-2.5-pro",
        max_workers: int = 10,
        max_retries: int = 3,
    ):
        """Initializes the GeminiClient.

        Args:
            api_key: 
                Google Gemini API key used for authentication.
            input_dir: 
                Path to the directory containing source image files.
            output_dir: 
                Path to the directory where text descriptions will be
                saved. The directory is created if it does not exist.
            model_id: 
                Identifier of the Gemini model to use for image
                description. Defaults to ``"gemini-2.5-pro"``.
            max_workers: 
                Maximum number of parallel worker threads to use when
                processing images in batch. Defaults to ``10``.
            max_retries: 
                Maximum number of retry attempts for failed image
                requests in batch processing. Defaults to ``3``.
        """
        self.api_key = api_key
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_retries = max_retries

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def show_supported_models() -> None:
        """
        Print all supported Gemini models.

        This method iterates over the internally defined ``SUPPORTED_MODELS``
        mapping and prints each model ID alongside its human-readable name.
        """
        print("Supported Gemini Models:\n")
        for model_id, model_name in sorted(SUPPORTED_MODELS.items()):
            print(f"- {model_id}: {model_name}")

    def _encode_image(self, image_path: str) -> str:
        """
        Read an image file and encodes it as a base64 string.

        Args:
            image_path: Path to the image file to be encoded.

        Returns:
            str: Base64-encoded string representation of the image content.

        Raises:
            RuntimeError: If the image file cannot be opened or read.
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")

    def _build_url(self) -> str:
        """
        Construct the full Gemini API URL for the configured model.

        Returns:
            str: The fully qualified API endpoint URL including model ID
            and API key query parameter.
        """
        return f"{self.API_BASE_URL}{self.model_id}{self.API_SUFFIX}{self.api_key}"

    def describe_image(self, image_path: str, prompt: str) -> Optional[str]:
        """
        Generate a text description for a single image via the Gemini API.

        This method:
            * Encodes the image as base64.
            * Constructs the appropriate API request payload using the
              provided prompt and the encoded image.
            * Sends the request to the Gemini API.
            * Parses and returns the first candidate text description.

        Args:
            image_path: Path to the image file to be described.
            prompt: Text prompt that guides the description the model should
                generate (e.g., task instructions or desired style).

        Returns:
            Optional[str]: The generated text description if the request
            succeeds and the response can be parsed; otherwise ``None``.

        Logs:
            - Errors for network issues, unexpected response formats, or any
              other runtime exceptions, tagged with the image path.
        """
        try:
            encoded_image = self._encode_image(image_path)
            api_url = self._build_url()

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": encoded_image,
                                }
                            },
                        ],
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {image_path}: {e}")
        except KeyError:
            logging.error(
                f"Unexpected response format for {image_path}: {response.json()}"
            )
        except Exception as e:
            logging.error(f"General error for {image_path}: {e}")

        return None

    def process_single_file(
        self, 
        filename: str, 
        prompt: str
    ) -> Tuple[str, str]:
        """
        Process a single image file and writes its description to disk.

        This worker method:
            * Builds full input and output paths based on the configured
              ``input_dir`` and ``output_dir``.
            * Calls :meth:`describe_image` to obtain a description.
            * Writes the description to a ``.txt`` file next to the output
              directory.

        Args:
            filename: 
                Name of the image file (relative to ``input_dir``) to
                be processed.
            prompt: 
                Text prompt to use when requesting the image description
                from the Gemini API.

        Returns:
            Tuple[str, str]: A tuple of ``(filename, status_message)`` where
                ``status_message`` is one of:

                ``ok``:
                    The description was successfully generated and saved.
                ``api_error``:
                    The API call failed or produced no text.
                ``write_error: <details>``:
                    The description could not be written to disk.
        """
        image_path = os.path.join(self.input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + "_description.txt"
        output_path = os.path.join(self.output_dir, output_filename)

        description = self.describe_image(image_path, prompt)

        if description:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(description)
                return filename, "ok"
            except IOError as e:
                return filename, f"write_error: {e}"
        else:
            return filename, "api_error"

    def _run_batch(
            self, 
            files: List[str], 
            prompt: str, 
            attempt: int
        ) -> Dict[str, str]:
        """
        Execute a batch of image description tasks using a thread pool.

        Each file in ``files`` is processed in parallel via
        :meth:`process_single_file`. Progress is tracked using a ``tqdm``
        progress bar.

        Args:
            files: 
                List of image filenames (relative to ``input_dir``) to be
                processed in this batch.
            prompt: 
                Text prompt to pass to :meth:`describe_image` for each image.
            attempt: 
                Current attempt number, used only for display in the progress
                bar and logging (e.g., for retries).

        Returns:
            Dict[str, str]: 
                A mapping from filename to status message, where
                the status values follow the same convention as
                :meth:`process_single_file` (e.g., ``"ok"``, ``"api_error"``,
                ``"write_error: ..."`` or ``"crash: ..."`` for uncaught 
                errors).
        """
        results: Dict[str, str] = {}

        # Adjust workers if we have fewer files than max_workers
        current_workers = min(self.max_workers, max(1, len(files)))

        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            with tqdm(total=len(files), desc=f"Processing (Attempt {attempt})") as pbar:

                # Submit tasks
                future_to_file = {
                    executor.submit(self.process_single_file, f, prompt): f
                    for f in files
                }

                for future in as_completed(future_to_file):
                    file_name = future_to_file[future]
                    try:
                        _, status = future.result()
                    except Exception as e:
                        status = f"crash: {e}"

                    results[file_name] = status
                    pbar.update(1)

        return results

    def run(self, prompt: str) -> None:
        """
        Run batch processing over all images in the input directory.

        This is the main entry point for using the client. It:
            1. Scans the configured ``input_dir`` for supported image files.
            2. Runs an initial batch processing pass for all images.
            3. Retries failed images up to ``max_retries - 1`` additional times.
            4. Logs a final summary including success count and elapsed time.

        Args:
            prompt: Text prompt to provide to the Gemini model for every image
                in the input directory.

        Logs:
            - An error if the input directory does not exist.
            - A warning if no image files are found.
            - Information about the total number of images, retries, and
              final success statistics including elapsed time.
        """
        # 1. Gather files
        try:
            all_files = [
                f
                for f in os.listdir(self.input_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        except FileNotFoundError:
            logging.error(f"Input directory not found: {self.input_dir}")
            return

        if not all_files:
            logging.warning("No image files found in input directory.")
            return

        logging.info(f"Found {len(all_files)} images to process.")
        start_time = time.perf_counter()

        # 2. Initial Run
        results = self._run_batch(all_files, prompt, attempt=1)

        # 3. Retry Logic
        for attempt in range(2, self.max_retries + 1):
            failed_files = [f for f, status in results.items() if status != "ok"]

            if not failed_files:
                break

            logging.info(f"Retrying {len(failed_files)} failed images...")
            time.sleep(2)  # Brief pause before retrying

            retry_results = self._run_batch(failed_files, prompt, attempt)
            results.update(retry_results)

        # 4. Final Report
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        success_count = sum(1 for s in results.values() if s == "ok")
        minutes, seconds = divmod(elapsed, 60)

        logging.info("-" * 40)
        logging.info("Processing Complete.")
        logging.info(f"Success: {success_count}/{len(all_files)}")
        logging.info(f"Time: {int(minutes)} min {seconds:.1f} sec")
        logging.info("-" * 40)