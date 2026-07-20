"""
This script provides an example of running diagnostic inference on a post-disaster scene using rapidtools by:
1. Downloading a synthetic landslide reconnaissance image from the dataset registry.
2. Initializing a local Gemma-4 vision-language model for multimodal analysis.
3. Defining a custom diagnostic prompt to evaluate the scene based on domain expertise.
4. Executing the inference and outputting the AI's assessment to the screen.
"""

from rapidtools import Gemma4Inference, download_dataset

# Define your prompt:
PROMPT_STRING = "Describe the condition of this post-disaster scene and what went wrong."

# Download our generated disaster reconnaissance photo:
[image_path] = download_dataset('synthetic_landslide_image')

# Initialize the local inference model for Google's Gemma-4 vision-language model:
vlm_model = Gemma4Inference(
    model_id='google/gemma-4-E2B-it',
    temperature=0.4
)

# Send the image and your custom prompt to the Vision-Language Model
# for analysis, then print the AI's diagnostic response to the screen:
output = vlm_model.run_inference(
        image_inputs=image_path,
        prompt=PROMPT_STRING
    )

print(output.text)
