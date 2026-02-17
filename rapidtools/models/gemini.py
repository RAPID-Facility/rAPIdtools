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
# 02-16-2025

import base64
import logging
import os
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from requests.exceptions import RetryError, HTTPError
from tqdm import tqdm

# Import shared configuration
from rapidtools.config import get_configured_session, DEFAULT_RETRY_TOTAL, \
    REQUESTS_TIMEOUT_VAL

GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
GEMINI_DEFAULT_MODEL = 'gemini-2.0-flash'

# Source of Truth for supported extensions and their MIME types:
MIME_MAP = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
    '.bmp': 'image/bmp',
}


class GeminiInference:
    """
    A client for performing inference on imagery using Google's Gemini API.
    """

    def __init__(
        self, 
        api_key: str | None = None, 
        model_id: str = GEMINI_DEFAULT_MODEL,
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None
    ):
        """
        Initialize the inference client.

        Args:
            api_key: Can be one of three things:
                     1. The raw API key string.
                     2. A path to a text file containing the API key.
                     3. None (attempts to load from GOOGLE_API_KEY env var).
            model_id: The model to use.
            max_workers: Parallel threads for batch processing.
            max_retries: Retry attempts for failed API calls.
            system_instruction: Optional system instruction to guide model behavior.
        """
        self.api_key = self._resolve_api_key(api_key)
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.system_instruction = system_instruction
        
        self.session = get_configured_session()
        self.session.headers.update({'x-goog-api-key': self.api_key})

        self._validate_model()

    def _resolve_api_key(self, key_input: str | None) -> str:
        """
        Resolves API Key from arg, env var, or file path.
        Prioritizes: 
            Explicit Arg > Env Var > File Path (if arg/env points to one).
        """
        # Get the raw input:
        raw_key = (key_input or os.environ.get('GOOGLE_API_KEY', '')).strip()
    
        if not raw_key:
            raise ValueError(
                'API Key is missing. Pass it as an argument or set the '
                'GOOGLE_API_KEY environment variable.'
            )
    
        # Determine if it's a file path or the key itself
        # Check for file existence only if it looks like a path (e.g., has / 
        # or . or \) or if the string is reasonably short:
        potential_path = Path(raw_key)
        resolved_key = raw_key
    
        try:
            if potential_path.is_file():
                resolved_key = potential_path.read_text(
                    encoding='utf-8'
                ).strip()
                
                if not resolved_key:
                    raise ValueError(
                        f"API key file found at '{raw_key}' but it is empty."
                    )
                    
        except OSError as e:
            # Ignore "File name too long" as it indicates an invalid key.
            # Alert the user for permission issues or other system errors:
            if e.errno != 36: # 36 is 'File name too long'
                logging.debug(
                    f'Note: Could not check if input is a file path: {e}'
                )
                
        # Remove any stray quotes (common when copy-pasting from .env files):
        resolved_key = resolved_key.strip("'\" \n\t")
    
        if not resolved_key:
            raise ValueError('Resolved API key is empty.')
    
        # Google AI keys (Gemini/Vertex) almost always start with AIza:
        if not resolved_key.startswith('AIza'):
            logging.warning(
                "API Key does not start with 'AIza'. Google API keys usually "
                'follow this format. Please check if you accidentally passed a'
                " file path that doesn't exist."
            )
    
        # Basic length check: Most Google API keys are ~39 characters:
        if len(resolved_key) < 20:
            raise ValueError(
                'The API key provided appears incomplete. Please check that '
                'you copied the entire key and try again.'
            )
    
        return resolved_key

    def _validate_model(self) -> None:
            """
            Soft-validates that the model exists. 
            Catches network errors gracefully but exposes auth/config errors.
            """
            if not self.api_key or not self.model_id:
                return
    
            # Normalize the current ID for comparison:
            target_model = self.model_id.replace('models/', '')
    
            try:
                # Fetch available models:
                available_models = self.list_available_models(self.api_key)
                
                # Normalize list for comparison:
                available_ids = {m.replace('models/', '') for m in available_models}
    
                # Check existence:
                if target_model not in available_ids and available_ids:
                    # Suggest a fix if the user input has a typo by showing the
                    # first five available models:
                    logging.warning(
                        f"Model '{self.model_id}' is not in the list of "
                        f"available models for this key.\n Available: "
                        f"{', '.join(sorted(list(available_ids))[:5])}..."
                    )
    
            except ValueError as e:
                # A ValueError indicates a structural/config issue that 
                # requires developer attention:
                logging.error(
                    f'Configuration error: Model validation failed. {e}'
                )
            except Exception as e:
                # Suppress transient/environmental errors (network, timeout) to
                # avoid noise, but retain debug visibility for troubleshooting:
                logging.debug(
                    f'Transient error during model validation: {e}'
                )
            
    @staticmethod
    def list_available_models(api_key: str) -> list[str]:
        """
        Fetche list of available Gemini models supported for content generation.

        This method queries the Google Gemini API to retrieve all models 
        associated with the provided API key, filters for those that support
        the 'generateContent' method, and strips the 'models/' prefix from 
        their names.

        Args:
            api_key (str): The Google API key used for authentication.

        Returns:
            list[str]: 
                A sorted list of model IDs (e.g., ['gemini-1.5-flash', 
                'gemini-pro']). Returns an empty list if the request fails or 
                auth is invalid.

        Example:
            >>> api_key = 'AIzaSy...'
            >>> models = GeminiClient.list_available_models(api_key)
            >>> print(models)
            ['gemini-1.0-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']
        """
        if not api_key:
            logging.error('Cannot get available models: No API key provided.')
            return []

        url = f'{GEMINI_BASE_URL}/models'

        try:
            session = get_configured_session()
            response = session.get(
                url, 
                headers={'x-goog-api-key': api_key}, 
                timeout=10
            )

            if response.status_code != 200:
                logging.warning(
                    f'Failed to fetch models. Status: {response.status_code}'
                )
                return []

            data = response.json()

            # Filter for models that support text generation and strip the prefix:
            models = [
                m['name'].replace('models/', '')
                for m in data.get('models', [])
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]

            return sorted(models)

        except Exception as e:
            logging.error(f"Error fetching Gemini models: {e}")
            return []

    def _get_mime_type(self, image_path: Path) -> str:
        """
        Determines the MIME type for Gemini API compatibility.
        
        Explicitly maps common extensions to ensure correct types 
        (e.g., .jpg -> image/jpeg) and falls back to standard library guessing
        for others.
        """
        ext = image_path.suffix.lower()
        
        # Check the explicit map first:
        if ext in self.MIME_MAP:
            return self.MIME_MAP[ext]

        # Fallback to standard library guessing:
        guess, _ = mimetypes.guess_type(image_path)
        
        # Final fallback:
        return guess or 'image/jpeg'

    def run_inference(
            self, 
            image_input: str | Path, 
            prompt: str, 
            json_mode: bool = False,
            max_retries: int | None = None
        ) -> str | None:
            """
            Perform inference on a single image (local file or URL).
    
            This method handles file reading/downloading, payload construction,
            and error handling. Retries for transient network errors are 
            handled automatically by the session adapter.
    
            Args:
                image_input (str | Path): 
                    Path to a local file OR a valid URL (http/https).
                prompt (str): 
                    The text instruction or question for the model.
                json_mode (bool, optional): 
                    If ``True``, requests JSON output. Defaults to ``False``.
                max_retries (int, optional):
                    Override the default retry count for this specific request.
                    If ``None``, uses the global configuration.
    
            Returns:
                str | None: The generated text response, or ``None`` on failure.
    
            Example:
                >>> import requests
                >>> from pathlib import Path
                >>> from rapidtools.models import GeminiModels
                >>> # (Assuming 'GeminiClient' is the class containing this method)
                >>>
                >>> # 1. Initialize Client
                >>> # Note: Replace 'YOUR_API_KEY' with a real key
                >>> client = GeminiClient(api_key='YOUR_API_KEY', model_id=GeminiModels.GEMINI_1_5_FLASH)
                >>>
                >>> # 2. Setup: Download a real test image to a local file
                >>> test_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg'
                >>> local_path = Path('test_cat.jpg')
                >>> _ = local_path.write_bytes(requests.get(test_url).content)
                >>>
                >>> # 3. Run Inference on Local File
                >>> try:
                ...     result = client.run_inference(local_path, 'Describe this image.')
                ...     print(result)
                ... finally:
                ...     local_path.unlink() # Cleanup
                'A close-up of a cat looking at the camera.'
                >>>
                >>> # 4. Run Inference directly on URL (No local download needed)
                >>> result_url = client.run_inference(test_url, 'What animal is this?')
                >>> print(result_url)
                'It is a cat.'
    
            """
            # Initialize default variables:
            image_bytes = None
            mime_type = 'image/jpeg' # Default fallback
            input_str = str(image_input)
            
            # Fetch image data (Handle URL vs local file):
            try:
                if input_str.startswith(('http://', 'https://')):
                    # If it is a URL: Download it using the configured session:
                    logging.info(f'Downloading image from URL: {input_str}')
                    try:
                        img_resp = self.session.get(
                            input_str, 
                            timeout=REQUESTS_TIMEOUT_VAL
                        )
                        img_resp.raise_for_status()
                        image_bytes = img_resp.content
                        # Use the content-type header if available, else guess
                        # from URL:
                        mime_type = img_resp.headers.get('Content-Type') or \
                            self._get_mime_type(Path(input_str))
                    except Exception as e:
                        logging.error(f'Failed to download image from URL: {e}')
                        return None
                else:
                    # If it is a local file:
                    path_obj = Path(image_input)
                    
                    # Fail fast if the file does not exist to avoid subsequent 
                    # errors:
                    if not path_obj.exists():
                        logging.error(f'File not found: {path_obj}')
                        return None
                    
                    mime_type = self._get_mime_type(path_obj)
                    
                    # Safely read binary data from disk:
                    try:
                        image_bytes = path_obj.read_bytes()
                    except OSError as e:
                        logging.error(
                            f'Failed to read local file \'{path_obj.name}\': {e}'
                        )
                        return None
    
            except Exception as e:
                logging.error(f'Unexpected error preparing image data: {e}')
                return None
    
            # Encode data:
            encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    
            # Construct payload:
            contents_parts = [
                {
                    'inline_data': {
                        'mime_type': mime_type,
                        'data': encoded_data
                    }
                },
                {'text': prompt}
            ]
    
            # Define the full JSON payload
            # Explicitly disable safety blocks to prevent over-censorship on
            # standard tasks:
            payload = {
                'contents': [{'parts': contents_parts}],
                'safetySettings': [
                    {'category': cat, 'threshold': 'BLOCK_NONE'}
                    for cat in [
                        'HARM_CATEGORY_HARASSMENT', 
                        'HARM_CATEGORY_HATE_SPEECH', 
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT', 
                        'HARM_CATEGORY_DANGEROUS_CONTENT'
                    ]
                ],
                'generationConfig': {
                    'temperature': 0.4, # Lower temperature = more deterministic/factual
                    'maxOutputTokens': 2048,
                    'responseMimeType': 
                        'application/json' if json_mode else 'text/plain'
                }
            }
    
            if self.system_instruction:
                payload['systemInstruction'] = {
                    'parts': [{'text': self.system_instruction}]
                }
    
            url = f'{GEMINI_BASE_URL}/models/{self.model_id}:generateContent'
            
            # Explicitly pass the key in headers (redundancy safety for session
            # handling):
            headers = {'x-goog-api-key': self.api_key}
    
            # Determine session to use (global or temporary with override):
            session_to_use = self.session
            should_close_session = False
            
            if max_retries is not None:
                # Create a temporary session with custom retry logic via helper:
                session_to_use = get_configured_session(retries=max_retries)
                should_close_session = True
    
            # Execute request:
            try:
                response = session_to_use.post(
                    url, 
                    json=payload, 
                    headers=headers, 
                    timeout=REQUESTS_TIMEOUT_VAL
                )
                
                # Raise an HTTPError if the status code indicates failure:
                response.raise_for_status()
                result = response.json()
    
                # Attempt to parse the text result from the deep JSON structure:
                try:
                    return result['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError):
                    feedback = result.get('promptFeedback', {})
                    block_reason = feedback.get('blockReason')
                    if block_reason:
                        logging.warning(
                            f'Blocked: {input_str} Reason: {block_reason}'
                        )
                    else:
                        logging.error(
                            f'Unexpected response format for {input_str}: {result}'
                        )
                    return None
    
            # Error handling:
            except RetryError:
                # This occurs after all configured retries are exhausted
                limit = max_retries if max_retries is not None \
                        else DEFAULT_RETRY_TOTAL
                logging.error(
                    f'Failed to process {input_str}: Max retries ({limit}) exceeded.'
                )
            except HTTPError as e:
                logging.error(
                    f'HTTP Error for {input_str}: {e} | Response: {e.response.text}'
                )
            except Exception as e:
                # Catch-all for any other unforeseen issues:
                logging.error(f'Unexpected error processing {input_str}: {e}')
            finally:
                # Clean up temporary session if we created one:
                if should_close_session:
                    session_to_use.close()
                
            return None

    def run_batch(
            self, 
            input_dir: str | Path, 
            output_dir: str | Path, 
            prompt: str,
            skip_existing: bool = True,
            json_mode: bool = False,
            save_manifest: bool = False
        ) -> dict[str, int]:
            """
            Runs inference on a directory of images in parallel.
    
            This method scans the input directory for supported images (defined in MIME_MAP), 
            processes them using a thread pool, and saves the text results to the output 
            directory. It handles rate limiting, progress tracking, and optional manifest generation.
    
            Args:
                input_dir (str | Path): Source directory containing images.
                output_dir (str | Path): Destination directory for text results.
                prompt (str): The text instruction to send to the model for all images.
                skip_existing (bool, optional): If ``True``, skips images that already have 
                                                a corresponding non-empty output file. 
                                                Defaults to ``True``.
                json_mode (bool, optional): If ``True``, requests JSON output. Defaults to ``False``.
                save_manifest (bool, optional): If ``True``, saves a 'batch_manifest.json' 
                                                log in the output directory. Defaults to ``False``.
    
            Returns:
                dict[str, int]: A dictionary containing execution statistics 
                                (e.g., {'ok': 10, 'failed': 2, 'skipped': 5}).
    
            Example:
                >>> import shutil
                >>> from pathlib import Path
                >>> # 1. Setup dummy directories
                >>> in_dir = Path('temp_input')
                >>> out_dir = Path('temp_output')
                >>> in_dir.mkdir(exist_ok=True)
                >>> (in_dir / 'test1.jpg').touch() # Create dummy images
                >>> (in_dir / 'test2.png').touch()
                >>>
                >>> # 2. Run Batch Inference
                >>> # (Assumes 'client' is initialized)
                >>> stats = client.run_batch(
                ...     input_dir=in_dir,
                ...     output_dir=out_dir,
                ...     prompt='Describe this image.',
                ...     save_manifest=True
                ... )
                >>> print(stats)
                {'ok': 2, 'failed': 0, 'skipped': 0}
                >>>
                >>> # 3. Cleanup
                >>> shutil.rmtree(in_dir)
                >>> shutil.rmtree(out_dir)
            """
            in_path = Path(input_dir)
            out_path = Path(output_dir)
    
            if not in_path.is_dir():
                raise ValueError(f'Input directory not found: {in_path}')
    
            out_path.mkdir(parents=True, exist_ok=True)
    
            # 1. Efficient File Discovery
            # Dynamically load valid extensions from our class constant
            valid_exts = set(self.MIME_MAP.keys())
            
            # Generator to find files (converted to list for tqdm total count)
            try:
                images = [
                    p for p in in_path.iterdir() 
                    if p.is_file() and p.suffix.lower() in valid_exts
                ]
            except OSError as e:
                logging.error(f'Error reading directory: {e}')
                return {'ok': 0, 'failed': 0, 'skipped': 0}
    
            if not images:
                logging.warning(f'No valid images found in {in_path}')
                return {'ok': 0, 'failed': 0, 'skipped': 0}
    
            # 2. Setup Stats & Manifest
            stats = {'ok': 0, 'failed': 0, 'skipped': 0}
            manifest_data = {}
    
            # 3. Worker Function
            def _process(img_path: Path) -> tuple[str, str | None]:
                """Returns (status, error_message/None)."""
                dest_file = out_path / (img_path.stem + '.txt')
    
                if skip_existing and dest_file.exists() and dest_file.stat().st_size > 0:
                    return 'skipped', None
    
                # Run inference (retries handled internally by run_inference)
                # We treat 'None' return as a failure.
                result_text = self.run_inference(
                    img_path, 
                    prompt, 
                    json_mode=json_mode
                )
    
                if result_text:
                    try:
                        dest_file.write_text(result_text, encoding='utf-8')
                        return 'ok', None
                    except OSError as e:
                        return 'failed', f'Write Error: {e}'
                
                return 'failed', 'API Error or Safety Block'
    
            # 4. Execution
            # We cap workers at 10 to prevent immediate rate limits, regardless of config
            effective_workers = min(self.max_workers, 10)
            logging.info(f'Starting batch of {len(images)} images using {effective_workers} threads.')
    
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                # Map keys (futures) to values (filenames) for tracking
                future_to_file = {
                    executor.submit(_process, img): img.name 
                    for img in images
                }
                
                with tqdm(total=len(images), unit='img', desc='Processing') as pbar:
                    for future in as_completed(future_to_file):
                        filename = future_to_file[future]
                        try:
                            status, error_msg = future.result()
                        except Exception as e:
                            status, error_msg = 'failed', str(e)
    
                        # Update stats
                        stats[status] += 1
                        
                        # Update manifest
                        if save_manifest:
                            manifest_data[filename] = {
                                'status': status,
                                'error': error_msg
                            }
    
                        pbar.set_postfix(ok=stats['ok'], fail=stats['failed'])
                        pbar.update(1)
    
            # 5. Save Manifest
            if save_manifest and manifest_data:
                manifest_path = out_path / 'batch_manifest.json'
                try:
                    import json
                    with open(manifest_path, 'w', encoding='utf-8') as f:
                        json.dump(manifest_data, f, indent=2)
                    logging.info(f'Manifest saved to {manifest_path}')
                except OSError as e:
                    logging.error(f'Failed to save manifest: {e}')
    
            logging.info(f'Batch complete. Stats: {stats}')
            return stats