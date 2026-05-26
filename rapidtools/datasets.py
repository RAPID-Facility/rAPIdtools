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
# 05-25-2026

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RemoteFile:
    """Represents a single downloadable file within a dataset."""
    url: str
    filename: str

# Dataset registry mapping descriptive dataset names to their respective URLs 
# and target filenames. A dataset can consist of a single file or a list of
# multiple files:
DATASET_REGISTRY: dict[str, list[RemoteFile]] = {
    'aerial_chs_prompts': [
        RemoteFile(
            url='https://www.dropbox.com/scl/fi/wd8zd24xvgmhd0hrsw5eo/aerial_CHS_prompts.txt?rlkey=diybx8npfrlvam4wgltpw9qbe&st=wvmw5vz4&dl=0',
            filename='aerial_CHS_prompts.txt',
        )
    ],
    'altadena_sample_buildings': [
        RemoteFile(
            url='https://www.dropbox.com/scl/fi/4mr3r5as4ccebqxuvooaw/altadena_sample_buildings.geojson?rlkey=cpkyamwg7hdos984a552aj8d5&st=7np0z16s&dl=0',
            filename='altadena_sample_buildings.geojson',
        )
    ],
    'eaton_patch1': [
        RemoteFile(
            url='https://www.dropbox.com/scl/fi/qzehjo91mz1lrd27etcve/eaton_patch_20250214.tiff?rlkey=rcmkpdyaixgq9i18q997bnp2z&st=6mgdvhay&dl=0',
            filename='eaton_patch_20250214.tiff',
        )
    ],
    'eaton_patch2': [
        RemoteFile(
            url='https://www.dropbox.com/scl/fi/02sx7l5r594jl5qg3qfhq/Eaton_Trinity38_RGBortho_20250214.tiff?rlkey=5frnvs7drsxj7s8ofj2upqdvt&st=4814ge5y&dl=0',
            filename='Eaton_Trinity38_RGBortho_20250214.tiff',
        )
    ],
    'mapillary_token': [
        RemoteFile(
            url='https://www.dropbox.com/scl/fi/wkfxeyjv65rlj4zgsawh2/mapillary_token.txt?rlkey=ns1v51demkr8lm3grmonmw9py&st=vzq95fa9&dl=0    ',
            filename='mapillary_token.txt',
        )
    ]
}


def download_dataset(
    dataset_names: str | list[str], output_dir: str | Path = '.'
) -> list[Path]:
    """
    Downloads all files associated with one or more datasets from the registry.

    Uses atomic writing (downloading to a temporary file first) to ensure 
    that interrupted downloads do not result in corrupted files.

    Args:
        dataset_names (str | list[str]): 
            A single descriptive name of the dataset, or a list of names 
            (e.g., 'eaton_patch1' OR ['eaton_patch1', 'eaton_patch2']).
            Input is case-insensitive and ignores leading/trailing whitespace.
        output_dir (str | Path, optional): Directory where files should be saved.
            Defaults to the current working directory ('.').

    Returns:
        list[Path]: 
            A flat list of absolute paths to all successfully downloaded files.
        
    Raises:
        ValueError: 
            If any of the requested dataset names do not exist in the registry.
    """
    # Normalize input into a list so we can process it uniformly
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    available_datasets = list(DATASET_REGISTRY.keys())
    clean_names = []

    # SAFEGUARD 1 & 2: Normalize and validate ALL requested datasets before downloading
    for name in dataset_names:
        clean_name = name.strip().lower()
        
        if clean_name not in DATASET_REGISTRY:
            error_message = f"Dataset '{name}' not found in the registry."
            
            # Use difflib to find the closest matching dataset name
            suggestions = difflib.get_close_matches(
                clean_name, available_datasets, n=1, cutoff=0.5
            )
            
            if suggestions:
                error_message += f" Did you mean '{suggestions[0]}'?"
            else:
                error_message += f" Available datasets: {available_datasets}"
                
            logger.error(error_message)
            raise ValueError(error_message)
            
        clean_names.append(clean_name)

    # Deduplicate the list in case the user passed the same dataset twice
    # using a dictionary preserves the original order:
    clean_names = list(dict.fromkeys(clean_names))

    # Resolve output directory and create it if it does not exist:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate all individual files from the requested datasets
    files_to_download: list[RemoteFile] = []
    for clean_name in clean_names:
        files_to_download.extend(DATASET_REGISTRY[clean_name])

    logger.info(
        f"Preparing to download {len(files_to_download)} file(s) "
        f"across {len(clean_names)} dataset(s)..."
    )

    downloaded_paths: list[Path] = []

    for remote_file in files_to_download:
        final_path = out_dir / remote_file.filename
        temp_path = final_path.with_suffix('.tmp')

        # Skip if the fully downloaded file already exists:
        if final_path.exists():
            logger.info(f'File already exists at: {final_path}. Skipping.')
            downloaded_paths.append(final_path)
            continue

        # Bypass the Dropbox web view to force a direct file stream:
        direct_link = remote_file.url.replace(
            'www.dropbox.com', 'dl.dropboxusercontent.com'
        )

        try:
            # Always include a timeout to prevent indefinite hanging:
            response = requests.get(direct_link, stream=True, timeout=15)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192

            # Download to a temporary file first:
            with open(temp_path, 'wb') as file, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f'Downloading {remote_file.filename}',
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))

            # Rename the temp file to the final filename upon successful completion:
            temp_path.rename(final_path)
            downloaded_paths.append(final_path)

        except requests.exceptions.RequestException as req_err:
            logger.error(
                f'Network Error while downloading {remote_file.filename}: {req_err}'
            )
            
        except Exception as err:
            logger.error(
                f'Unexpected error downloading {remote_file.filename}: {err}'
            )
            
        finally:
            # Cleanup: Remove corrupted temporary file if download failed/cancelled:
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f'Cleaned up incomplete temporary file: {temp_path}')

    if downloaded_paths:
        logger.info(
            f"Successfully secured {len(downloaded_paths)}/"
            f"{len(files_to_download)} files."
        )
        
    return downloaded_paths