# python version: 3.11+
# required libraries: pip install requests tqdm

import base64
import os
import requests
import logging
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Custom Exceptions for Better Error Handling ---
class ChatGPTClientError(Exception):
    """Base exception for the ChatGPT client."""
    pass

class ImageEncodingError(ChatGPTClientError):
    """Raised when an image cannot be encoded."""
    pass

class APIRequestError(ChatGPTClientError):
    """Raised when an API request fails."""
    pass

# --- The ChatGPT Client Class ---

class ChatGPTClient:
    """
    A client for interacting with the OpenAI GPT-4 Vision API to describe images.
    """
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    SUPPORTED_MODELS = {
        "gpt-4o": "GPT-4o (Latest and greatest)",
        "gpt-4-turbo": "GPT-4 Turbo with Vision",
    }

    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-4o",
        max_workers: int = 10,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        if model_id not in self.SUPPORTED_MODELS:
            logging.warning(f"Model '{model_id}' is not an official vision model. It may not work.")
            
        self.api_key = api_key
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # Use a single session for connection pooling and to set the auth header once
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Reads an image and returns its base64 encoded string."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise ImageEncodingError(f"Failed to encode image at {image_path}: {e}") from e

    def describe_image(
        self, 
        image_path: str, 
        prompt: str,
        max_tokens: int = 300,
        model_id: str | None = None
    ) -> str:
        """
        Sends a single image and a prompt to the OpenAI API and returns the description.

        Args:
            image_path: The path to the image file.
            prompt: The text prompt to guide the model's description.
            max_tokens: The maximum number of tokens to generate in the response.
            model_id: (Optional) The model to use, overriding the client's default.

        Returns:
            The text description from the model.
        
        Raises:
            ImageEncodingError: If the image file cannot be read or encoded.
            APIRequestError: If the API call fails or returns an unexpected format.
        """
        target_model_id = model_id or self.model_id
        base64_image = self._encode_image(image_path)
        
        payload = {
            "model": target_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }

        try:
            response = self.session.post(self.API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            # Try to include the API's error message for better debugging
            error_details = e.response.json().get("error", {}).get("message", "No details provided.")
            raise APIRequestError(f"API request failed: {e}. Details: {error_details}") from e
        except (KeyError, IndexError) as e:
            raise APIRequestError(f"Unexpected API response format: {data}") from e

    def batch_describe_directory(
        self,
        image_dir: str,
        prompt: str,
        results_dir: str,
    ) -> dict[str, str]:
        """
        Processes all images in a directory concurrently, with retries for failures.

        Args:
            image_dir: Path to the directory containing images.
            prompt: The text prompt to use for all images.
            results_dir: Path to the directory where description .txt files will be saved.

        Returns:
            A dictionary mapping image filenames to their final status ("ok" or "error: ...").
        """
        os.makedirs(results_dir, exist_ok=True)
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        final_results: dict[str, str] = {}
        images_to_process = image_files[:]

        for attempt in range(1, self.max_retries + 1):
            if not images_to_process:
                break
            
            logging.info(f"--- Starting attempt {attempt}/{self.max_retries} for {len(images_to_process)} images ---")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                with tqdm(total=len(images_to_process), desc=f"Attempt {attempt}") as pbar:
                    
                    futures = {
                        executor.submit(self._process_and_save, img, image_dir, prompt, results_dir): img
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

            images_to_process = [img for img, status in final_results.items() if status != "ok"]
        
        return final_results

    def _process_and_save(self, image_filename: str, image_dir: str, prompt: str, results_dir: str) -> str:
        """A helper method for the batch processing workflow."""
        image_path = os.path.join(image_dir, image_filename)
        description = self.describe_image(image_path=image_path, prompt=prompt)
        
        output_filename = os.path.splitext(image_filename)[0] + '_description.txt'
        output_path = os.path.join(results_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(description)
        return "ok"


# --- Example of How to Use the New Class ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # It's highly recommended to use environment variables for API keys
        # api_key = os.environ.get("OPENAI_API_KEY")
        # For this example, we'll read from a file.
        with open("openai_api_key.txt", "r") as f:
            api_key = f.read().strip()
            
        with open("orthomosaic_prompts.txt", "r") as f:
            prompt_text = f.read().strip()

        # 1. Create an instance of the client
        chatgpt_client = ChatGPTClient(api_key=api_key, model_id="gpt-4o")

        # 2. Process an entire directory
        print("\n--- Processing an entire directory with ChatGPT ---")
        start_time = time.perf_counter()
        
        results = chatgpt_client.batch_describe_directory(
            image_dir="eaton_aerial_images_feb25/overlaid_imagery",
            prompt=prompt_text,
            results_dir="chatgpt_outputs/eaton"
        )
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # --- Print Summary ---
        successful = sum(1 for status in results.values() if status == "ok")
        failed = len(results) - successful
        
        print("\n--- Batch Processing Complete ---")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        if failed > 0:
            print("Failed files:")
            for img, status in results.items():
                if status != "ok":
                    print(f"  - {img}: {status}")

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}. Please ensure API key and prompt files exist.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")