import base64
import os
import requests
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

# API URL Information:
API_BASE_URL = "https://generativelanguage.googleapis.com/v1/models/"
API_SUFFIX = ":generateContent?key="

# Supported Gemini Models:
SUPPORTED_GEMINI_MODELS = {
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

def show_supported_gemini_models() -> None:
    """
    Print all supported Gemini models with their IDs and names.
    """
    print("Supported Gemini Models:\n")
    for model_id, model_name in sorted(SUPPORTED_GEMINI_MODELS.items()):
        print(f"- {model_id}: {model_name}")

def encode_image(image_path: str) -> str:
    """
    Read an image file from and return its base64-encoded string.

    Args:
        image_path (str): The path to the image file (e.g., JPEG or PNG).

    Returns:
        str: The base64-encoded string representation of the image.

    Raises:
        RuntimeError: If the image cannot be read or encoded.
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

def build_model_url(model_id: str, api_key: str) -> str:
    """
    Construct the full Gemini API URL for a given model and API key.

    Args:
        model_id (str): The Gemini model identifier (e.g., "gemini-2.5-pro").
        api_key (str): The Google API key for authenticating requests.

    Returns:
        str: The complete URL to call the Gemini model's generateContent endpoint.
    """
    return f"{API_BASE_URL}{model_id}{API_SUFFIX}{api_key}"


def describe_image(
    image_path: str,
    prompt: str,
    model_id: str,
    api_key: str
) -> Optional[str]:
    """
    Get text description for an image from the specified Gemini model.

    Args:
        image_path (str): Path to the image file to be sent for description.
        prompt (str): Natural language prompt guiding the model's output,
                      e.g., "Describe the damage in this image."
        model_id (str): The Gemini model identifier to use for generation.
        api_key (str): Your Google Gemini API key for authentication.

    Returns:
        Optional[str]: The text response from the Gemini model, or None if
                       the request failed or the response was invalid.

    Notes:
        - This function handles HTTP and response parsing errors internally,
          printing error messages and returning None on failure.
        - Assumes the image is a JPEG; modify mime_type if using other formats.
    """
    encoded_image = encode_image(image_path)
    api_url = build_model_url(model_id, api_key)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded_image
                        }
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except KeyError:
        print("Unexpected response format:", response.json())
    return None

def process_image(
        image_filename: str,
        prompts: str,
        api_key:str
    ) -> Tuple[str, str]:
    """Describe one image and save the description to a text file."""
    try:
        image_path = os.path.join(image_dir, image_filename)
        description = describe_image(
            image_path=image_path,
            prompt=prompts,
            model_id='gemini-2.5-pro',
            api_key=api_key
        )

        description_output = os.path.join(
            api_results_dir,
            image_filename.split('.')[0] + '_description.txt'
        )

        with open(description_output, "w", encoding="utf-8") as f:
            f.write(description)

        return image_filename, "ok"
    except Exception as e:
        return image_filename, f"error: {e}"

def run_batch(
        image_list: List[str],
        prompts: str,
        api_key:str,
        attempt=1
    ):
    """Run one batch of image processing; returns dict of results."""
    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor, \
         tqdm(total=len(image_list),
              desc=f"Processing images (Attempt {attempt})") as pbar:

        futures = {
            executor.submit(process_image, img, prompts, api_key): img
            for img in image_list
        }

        for future in as_completed(futures):
            img = futures[future]
            try:
                _, status = future.result()
            except Exception as e:
                status = f"error: {e}"
            results[img] = status
            pbar.update(1)
    return results

# File and directory paths:
prompt_file = 'orthomosaic_prompts.txt'
api_key_file = 'rapid_gemini_api_key3.txt'
image_dir = 'eaton_aerial_images_feb25/overlaid_imagery'
api_results_dir = 'gemini_outputs/eaton'

# Prepare output directory:
os.makedirs(api_results_dir, exist_ok=True)

# Load prompt text and API key:
with open(prompt_file, "r", encoding="utf-8") as f:
    prompts = f.read().strip()

with open(api_key_file, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

# Gather image files and set worker count:
image_files = os.listdir(image_dir)
num_workers = min(MAX_WORKERS, max(1, len(image_files)))

# Get descriptions for each image:
start_time = time.perf_counter()
#results = run_batch(image_files, prompts, api_key, attempt=1)

# Retry failed images up to MAX_RETRIES times
for attempt in range(2, MAX_RETRIES + 1):
    failed = [img for img, status in results.items() if not status.startswith('ok')]
    if not failed:
        break  # all done
    print(f'\nRetrying {len(failed)} failed images (attempt {attempt})...')
    retry_results = run_batch(failed, prompts, api_key, attempt=attempt)
    results.update(retry_results)

end_time = time.perf_counter()
elapsed = end_time - start_time

minutes, seconds = divmod(elapsed, 60)
print(f'\nTotal processing time: {int(minutes)} min {seconds:.1f} sec')