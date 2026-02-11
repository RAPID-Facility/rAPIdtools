"""
Module for Gemini API interactions using rapidtools configuration.
"""

import base64
import os
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Import shared configuration
from rapidtools.config import get_configured_session


# Gemini-Specific Configuration:
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT_VAL = 60

class GeminiInference:
    """
    A client for performing inference on aerial, street-level, or oblique imagery
    using Google's Gemini API.
    """

    def __init__(
        self, 
        api_key: str, 
        model_id: str = "gemini-2.0-flash",
        max_workers: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize the inference client.

        Args:
            api_key: Google AI Studio API Key.
            model_id: The default model to use (e.g., 'gemini-1.5-flash').
            max_workers: Number of parallel threads for batch processing.
            max_retries: Number of retry attempts for failed API calls.
        """
        self.api_key = api_key
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # Use shared session factory from rapidtools.config
        self.session = get_configured_session()
        
        # Verify model availability
        available_models = GeminiInference.list_available_models(api_key)
        
        if model_id not in available_models:
            logging.warning(
                f"Requested model '{model_id}' not found in available list. "
                f"Using '{model_id}' anyway, but it might fail."
            )
            logging.info(f"Available models: {', '.join(available_models)}")
        
        self.model_id = model_id

    @staticmethod
    def list_available_models(api_key: str) -> List[str]:
        """
        Static method to fetch available Gemini models without initializing the class.
        
        Args:
            api_key: Google AI Studio API Key.
            
        Returns:
            List[str]: A list of model names.
        """
        # Create a temporary session just for this request using shared config
        session = get_configured_session()
        url = f"{GEMINI_BASE_URL}/models?key={api_key}"
        
        try:
            response = session.get(url, timeout=GEMINI_TIMEOUT_VAL)
            response.raise_for_status()
            data = response.json()
            
            # Filter for models that support 'generateContent'
            models = [
                m['name'].replace('models/', '') 
                for m in data.get('models', [])
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
            return sorted(models)
        except Exception as e:
            logging.error(f"Failed to fetch model list: {e}")
            return []

    def run_inference(
        self, 
        image_path: str, 
        prompt: str, 
        mime_type: str = "image/jpeg"
    ) -> Optional[str]:
        """Performs inference on a single image."""
        url = f"{GEMINI_BASE_URL}/models/{self.model_id}:generateContent?key={self.api_key}"
        
        try:
            with open(image_path, "rb") as f:
                encoded_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded_data
                            }
                        }
                    ]
                }],
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            }

            # Use shared timeout value
            response = self.session.post(url, json=payload, timeout=GEMINI_TIMEOUT_VAL)
            response.raise_for_status()
            
            result = response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                feedback = result.get("promptFeedback", {})
                if feedback.get("blockReason"):
                    logging.warning(f"Blocked: {image_path} Reason: {feedback['blockReason']}")
                return None

        except Exception as e:
            logging.error(f"Error processing {os.path.basename(image_path)}: {e}")
            return None

    def run_batch(self, input_dir: str, output_dir: str, prompt: str) -> None:
        """Runs inference on a directory of images in parallel."""
        os.makedirs(output_dir, exist_ok=True)
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.tif')
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

        if not images:
            logging.warning("No images found.")
            return

        logging.info(f"Processing {len(images)} images using {self.model_id}...")
        
        def _worker(filename):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
            
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return "skipped"

            result = self.run_inference(in_path, prompt)
            
            if result:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result)
                return "ok"
            return "failed"

        results = {"ok": 0, "failed": 0, "skipped": 0}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_worker, img): img for img in images}
            
            for future in tqdm(as_completed(futures), total=len(images)):
                status = future.result()
                results[status] += 1

        logging.info(f"Done. Stats: {results}")