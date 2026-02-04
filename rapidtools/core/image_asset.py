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
# 02-04-2025

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image as PillowImage
from PIL import ImageOps as PillowImageOps
from skimage import measure

if TYPE_CHECKING:
    from .bounding_box import BoundingBox
from rapidtools.config import (
    DEFAULT_INSTANCE_CMAP,
    DEFAULT_SEMANTIC_CMAP,
    REQUESTS_HEADERS,
    REQUESTS_TIMEOUT_VAL,
    MaskType,
)


@dataclass(kw_only=True)
class ImageAsset:
    """
    A single image asset, its location, metadata, and segmentation info.
    """

    path: Path
    id: str | None = field(default=None)

    # User-facing metadata (location, photographer, tags, etc.):
    properties: dict[str, Any] = field(default_factory=dict)

    # Mapping dictionary for the semantic segmentation masks:
    # Key: Pixel Value (int) -> Value: Class Name (str)
    semantic_map: dict[int, str] | None = field(default=None, repr=False)

    # Mapping dictionary for the instance segmentation mask:
    # Key: Instance ID (int) -> Value: Metadata dict or label
    instance_map: dict[int, Any] | None = field(default=None, repr=False)

    # Flag for delayed image download for efficient image handling:
    allow_missing_file: bool = field(default=False, repr=False)

    # --- Internal Caches ---
    _pil_image: PillowImage.Image | None = field(default=None, repr=False, init=False)

    # Lazy-loaded NumPy arrays for segmentation masks:
    _semantic_mask: np.ndarray | None = field(default=None, repr=False, init=False)

    _instance_mask: np.ndarray | None = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """
        Validate and normalize initialization arguments.

        Behavior:
        - Converts a string path to a pathlib.Path instance.
        - Resolves the path to an absolute path if it is relative, logging
          a warning when resolution is needed.
        - Verifies that the file exists at the resolved path; raises an error
          if it does not.
        - If id is not provided, sets it to the filename stem.

        Raises:
            ValueError: If the image file does not exist at the specified path.
        """
        # Ensure that self.path is ALWAYS a Path object:
        if isinstance(self.path, str):
            self.path = Path(self.path)

        # Resolve the path to a full path:
        self.path = self.path.resolve()

        # Only check existence of the image if we are NOT allowing missing
        # files:
        if not self.allow_missing_file and not self.path.exists():
            raise ValueError(f'Image file does not exist at: {self.path}')

        # If no ID was provided, default to the filename without extension:
        if self.id is None:
            self.id = self.path.stem

    def __repr__(self) -> str:
        """
        Return a concise, developer-friendly string representation.

        Example:
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> img = ImageAsset(
            ...     id='img_01',
            ...     path='/tmp/photo.jpg',
            ...     allow_missing_file=True
            ... )
            >>> repr(img)
            "<ImageAsset id='img_01' filename='photo.jpg'>"
        """
        return f"<ImageAsset id='{self.id}' filename='{self.filename}'>"

    @property
    def directory(self) -> Path:
        """
        The parent directory of the image file.

        Returns:
            Path: The directory containing the image as a Path object.

        Example:
            In the example below, we set ``allow_missing_file=True``. This is
            useful for lazy-loading assets from an API or database where the
            local file may not yet exist, allowing for metadata manipulation
            without disk overhead.

            >>> from rapidtools.core import ImageAsset
            >>> image = ImageAsset(
            ...     path='/projects/data/image.jpg',
            ...     allow_missing_file=True
            ... )

            The property returns a standard ``pathlib.Path`` object:

            >>> image.directory
            PosixPath('/projects/data')

            This allows for seamless path manipulation, such as joining
            related files in the same folder:

            >>> image.directory / 'metadata.json'
            PosixPath('/projects/data/metadata.json')
        """

        return self.path.parent

    @property
    def filename(self) -> str:
        """
        The filename of the image, including its extension.

        Returns:
            str: The image filename as a string.

        Example:
            In the example below, we set ``allow_missing_file=True``. This
            allows the asset to be initialized and its naming metadata to
            be accessed even if the file has not yet been downloaded to the
            local file system.

            >>> from rapidtools.core import ImageAsset
            >>> image = ImageAsset(
            ...     path='/data/projects/panoramas/image_001.jpg',
            ...     allow_missing_file=True
            ... )

            The ``filename`` property extracts only the final component of the
            path:

            >>> image.filename
            'image_001.jpg'

            This property is useful for generating logs, status reports,
            or matching image data with external database records:

            >>> print(f'Processing asset: {image.filename}')
            Processing asset: image_001.jpg
        """
        return self.path.name

    @property
    def is_downloaded(self) -> bool:
        """
        Check whether the image file exists on disk and is non-empty.

        This property verifies that:

            - The path exists.
            - The path points to a regular file (not a directory).
            - The file size is greater than zero bytes.

        Returns:
            bool:
                ``True`` if a non-empty image file exists at image path,
                otherwise ``False``.

        Examples:
            ``is_downloaded`` is essential for workflows where assets are
            initialized lazily. Here, by using ``allow_missing_file=True``, we
            can  work with an asset's metadata and only trigger a download if
            this property returns ``False``.

            Case 1: The file exists on the local file system:

            >>> from rapidtools.core import ImageAsset
            >>> image = ImageAsset('exists.jpg')
            >>> image.is_downloaded
            True

            Case 2: The file is missing or has not been downloaded yet. We use
            ``allow_missing_file=True`` to avoid an initialization error:

            >>> image = ImageAsset(path='missing.jpg', allow_missing_file=True)
            >>> image.is_downloaded
            False

            This allows for simple conditional logic in processing pipelines.

            >>> if not image.is_downloaded:
            ...     image.download(
            ...         url=r'https://upload.wikimedia.org/wikipedia/commons/1'
            ...             r'/1e/Suzzallo_Reading_Room%2C_May_2016.jpg'
            ...     )
            ...     print(f'Downloaded: {image.is_downloaded}')
            INFO: Downloading missing...
            INFO: Successfully downloaded: missing.jpg
            Downloaded: True
        """
        return (
            self.path.exists() and self.path.is_file() and self.path.stat().st_size > 0
        )

    @property
    def stem(self) -> str:
        """
        The filename of the image without its extension.

        Returns:
            str: The filename stem as a string.

        Examples:
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> asset = ImageAsset(
            ...     id='scene',
            ...     path='/data/images/scene_01.tif',
            ...     allow_missing_file=True
            ... )
            >>> asset.stem
            'scene_01'
            >>>
            >>> asset = ImageAsset(
            ...     id='photo',
            ...     path='photo.jpg',
            ...     allow_missing_file=True
            ... )
            >>> asset.stem
            'photo'
        """
        return self.path.stem

    def download(
        self,
        url: str | None = None,
        url_key: str = 'thumb_original_url',
        overwrite: bool = False,
        session: requests.Session | None = None,
    ):
        """
        Download the image file to self.path.

        Args:
            url (str, optional):
                The direct download link. If provided, this takes precedence
                over properties.
            url_key (str):
                The key in self.properties to look for if ``'url'`` is ``None``.
                Defaults to ``'thumb_original_url'``.
                Useful for downloading specific sizes (e.g. ``'thumb_2048_url'``).
            overwrite (bool):
                If True, forces download even if file exists.
            session (requests.Session, optional):
                An existing requests session to speed up connections.

        Raises:
            ValueError: If no URL is provided or found in properties.
            IOError: If the download fails.

        Examples:
            Basic usage through automatic URL Lookup:

            >>> from rapidtools.core import ImageAsset
            >>>
            >>> asset = ImageAsset(
            ...     id='img_01',
            ...     path='data/img_01.jpg',
            ...     properties={
            ...         'thumb_original_url':
            ...             r'https://upload.wikimedia.org/wikipedia/commons/1'
            ...             r'/1e/Suzzallo_Reading_Room%2C_May_2016.jpg'
            ...             },
            ...     allow_missing_file=True
            ... )
            >>> asset.download()
            INFO: Downloading 'img_01' from https://upload.wikimedia.org/
            wikipedia/commons/1/1e/Suzzallo_Reading_Room%2C_May_2016.jpg...
            INFO: Successfully downloaded: img_01.jpg

            Usage through providing explicit URL:

            >>> asset.download(
            ...     url=r'https://upload.wikimedia.org/wikipedia/commons/1'
            ...     r'/1e/Suzzallo_Reading_Room%2C_May_2016.jpg'
            ... )
            INFO: Downloading 'img_01' from https://upload.wikimedia.org/
            wikipedia/commons/1/1e/Suzzallo_Reading_Room%2C_May_2016.jpg...
            INFO: Successfully downloaded: img_01.jpg

            Download using a session (recommended for batch operations):

            >>> import requests
            >>>
            >>> assets = [
            ...     ImageAsset(
            ...         id='wiki_A',
            ...         path='data/wiki_a.jpg',
            ...         allow_missing_file=True,
            ...         properties={
            ...             'src': (
            ...                 'https://upload.wikimedia.org/wikipedia/commons/'
            ...                 'a/a9/Example.jpg'
            ...             ),
            ...         },
            ...     ),
            ...     ImageAsset(
            ...         id='wiki_B',
            ...         path='data/wiki_b.jpg',
            ...         allow_missing_file=True,
            ...         properties={
            ...             'src': (
            ...                 'https://upload.wikimedia.org/wikipedia/commons/5'
            ...                 '/50/Smile_Image.png'
            ...             ),
            ...         },
            ...     )
            ... ]
            >>>
            >>> with requests.Session() as s:
            ...     s.headers.update({'User-Agent': 'RapidTools/1.0'})
            ...     for asset in assets:
            ...         asset.download(url_key='src', session=s)
            INFO: Downloading 'wiki_A' from https://upload.wikimedia.org/
            wikipedia/commons/a/a9/Example.jpg...
            INFO: Successfully downloaded: wiki_a.jpg
            INFO: Downloading 'wiki_B' from https://upload.wikimedia.org/
            wikipedia/commons/5/50/Smile_Image.png...
            INFO: Successfully downloaded: wiki_b.jpg
        """
        # Check if the image needs to be downloaded:
        if self.is_downloaded and not overwrite:
            logging.info(f'Skipping download, file already exists: {self.filename}')
            return

        # Determine the image URL:
        # Priority: Explicit URL arg -> Property lookup -> Fail
        target_url = url or self.properties.get(url_key)

        # Check if a valid URL is found, if not show available keys for
        # debugging:
        if not target_url:
            available_keys = ', '.join(k for k in self.properties.keys() if 'url' in k)
            msg = (
                f"Cannot download '{self.id}': No explicit URL provided and "
                f"property '{url_key}' is missing."
            )
            if available_keys:
                msg += f' Did you mean one of these? [{available_keys}]'
            raise ValueError(msg)

        # Ensure directory exists:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Use a temporary filename to ensure atomicity.
        # If the script crashes mid-download, we won't be left with a
        # half-written 'image.jpg':
        temp_path = self.path.with_suffix(f'{self.path.suffix}.tmp')

        # Prepare download environment:
        requester = session if session else requests

        # Download image:
        try:
            logging.info(f"Downloading '{self.id}' from {target_url}...")

            with requester.get(
                target_url, headers=REQUESTS_HEADERS, stream=True, timeout=30
            ) as response:
                response.raise_for_status()

                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Rename the atomic file:
            temp_path.replace(self.path)
            logging.info(f'Successfully downloaded: {self.filename}')

        except Exception as e:
            # If download fails (e.g., partial file), clean up:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass  # Ignore cleanup errors during exception handling

            logging.error(f'Download failed for {self.id}: {e}')
            raise

    def get_mask_path(
        self, mask_type: MaskType, override_path: Path | str | None = None
    ) -> Path:
        """
        Determine the filename for the mask.

        Default Convention:
          - Image:    photo.jpg
          - Semantic: photo_semantic.png
          - Instance: photo_instance.png

        Args:
            mask_type:
                The type of mask (``'semantic'`` or ``'instance'``).
            override_path:
                If provided, this specific path is used instead of the
                automatic naming convention.

        Returns:
            Path: The resolved path object for the mask.

        Raises:
            ValueError: If ``mask_type`` is not a valid ``MaskType``.

        Examples:
            Create an ``ImageAsset``:

            >>> from rapidtools.core import ImageAsset
            >>> from rapidtools.config import MaskType
            >>> asset = ImageAsset(
            ...     id='A1',
            ...     path='/tmp/scene_01.jpg',
            ...     allow_missing_file=True
            ... )

            Getting mask path by providing a mask type:

            >>> path = asset.get_mask_path('instance')
            >>> print(path)
            /tmp/scene_01_instance.png

            Getting mask path using a manual override. This option is useful
            when masks are stored in a completely different folder:

            >>> path = asset.get_mask_path(
            ...     'semantic',
            ...     override_path='/masks/custom_01.png'
            ... )
            >>> print(path)
            /masks/custom_01.png
        """
        # Use the manual override if provided:
        if override_path is not None:
            return Path(override_path)

        # Otherwise, use the standard convention:
        try:
            valid_type = MaskType(mask_type)
        except ValueError:
            raise ValueError(
                f"Invalid mask_type '{mask_type}'. Expected one of: "
                f"{[m.value for m in MaskType]}"
            ) from None

        return self.path.with_name(f'{self.stem}_{valid_type.value}.png')

    def get_property(
        self, key: str | Iterable[str], default: Any = None
    ) -> Any | dict[str, Any]:
        """
        Safely retrieve one or more metadata properties from the asset.

        This method is polymorphic:

            1. If ``key`` is a ``str``, it returns the single value.
            2. If ``key`` is a list/tuple, it returns a dictionary of
               ``{key: value}``.

        Args:
            key (str or Iterable[str]):
                A single property name or a list of names to retrieve.
            default (Any, optional):
                The value to return for any key that is not found.
                Defaults to ``None``.

        Returns:
            Any or dict[str, Any]:
                - The value (if input was a single string).
                - A dictionary of values (if input was an iterable).

        Examples:
            Set up an ``ImageAsset``

            >>> from rapidtools.core import ImageAsset
            >>> asset = ImageAsset(
            ...     id='img_01',
            ...     path='photo.jpg',
            ...     properties={'width': 1920, 'height': 1080, 'iso': 100},
            ...     allow_missing_file=True
            ... )

            Request the values of individual properties:

            >>> asset.get_property('width')
            1920
            >>> asset.get_property('shutter_speed', default='Auto')
            'Auto'

            Request the values of multiple properties at once:

            >>> asset.get_property(['width', 'height'])
            {'width': 1920, 'height': 1080}

            Request the values of multiple properties, including some that
            exist and some that are missing:

            >>> asset.get_property(['iso', 'aperture'], default='Unknown')
            {'iso': 100, 'aperture': 'Unknown'}
        """
        # Case 1: Single string:
        if isinstance(key, str):
            return self.properties.get(key, default)

        # Case 2: Iterable (list, tuple, set):
        return {k: self.properties.get(k, default) for k in key}

    def load_image_from_disk(
        self,
        force_reload: bool = False,
        convert_mode: str | None = None,
        apply_exif_orientation: bool = True,
    ) -> PillowImage.Image:
        """
        Load the image from disk using Pillow and caches it.

        Behavior:

            - Lazy-caches by default (returns cached image if present).
            - Always fully decodes and then closes the underlying file handle.
              Optionally converts to a target mode (e.g., ``'RGB'``).
            - Optional force reload to bypass cache.

        Args:
            force_reload:
                If ``True``, re-reads the file from disk even if an image is
                currently cached. Defaults to ``False``.
            convert_mode:
                If provided, converts image to the specified PIL mode. Some
                options are ``L`` (Grayscale), ``1`` (Black and White),
                ``CMYK`` (Cyan, Magenta, Yellow, Key/Black), ``I``
                (32-bit signed integer pixels), and ``HSV`` (Hue, Saturation,
                Value color space).
            apply_exif_orientation (bool):
                If ``True``, reads the EXIF tags and rotates the image to appear
                upright (e.g., fixing sideways smartphone photos).
                Defaults to ``True``.

        Returns:
            PillowImage.Image: The loaded PIL image.

        Raises:
            FileNotFoundError:
                If the file does not exist or has not been downloaded.
            OSError:
                If the file exists but cannot be opened (corruption/permissions).

        Examples:
            Create a dummy red 100x100 RGB image for testing:

            >>> import os
            >>> from PIL import Image
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> temp_path = 'temp_example.jpg'
            >>> Image.new('RGB', (100, 100), color='red').save(temp_path)
            >>>
            >>> asset = ImageAsset(id='img_01', path=temp_path)

            Basic loading from a file. The first call reads the file from disk,
            subsequent calls return the cached result:

            >>> img1 = asset.load_image_from_disk()
            >>> print(f'{img1.format} {img1.size} {img1.mode}')
            INFO: Loading image data from: temp_example.jpg
            None (100, 100) RGB
            >>>
            >>> img2 = asset.load_image_from_disk()
            >>> print(img1 is img2)
            True

            Loading an image with mode conversion and force reload. Here force
            reload is enabled to ignore the cache, and the image is converted
            to Grayscale (``'L'``):

            >>> img_gray = asset.load_image_from_disk(
            ...     force_reload=True,
            ...     convert_mode='L'
            ... )
            INFO: Loading image data from: temp_example.jpg
            >>> print(img_gray.mode)
            L

            ``load_image_from_disk`` raises a ``FileNotFoundError`` if the file
            does not exist locally (or has not yet been downloaded) or raises an
            ``OSError`` if the file exists but cannot be opened (e.g., due to
            corruption or insufficient permissions). The following example shows
            a sample error handling situation:

            >>> missing_asset = ImageAsset(
            ...     id=1,
            ...     path='ghost.jpg',
            ...     allow_missing_file=True,
            ... )
            >>> try:
            ...     missing_asset.load_image_from_disk()
            ... except FileNotFoundError as e:
            ...     print('Caught error')
            Caught error

            Finally, run the following line to delete the temporary image
            file created for this example:

            >>> os.remove(temp_path)
        """
        # Return cache if available:
        if self._pil_image is not None and not force_reload:
            return self._pil_image

        # Validate file existence:
        if not self.is_downloaded:
            raise FileNotFoundError(
                f'File {self.path} not found. Please call the download method '
                'to retrieve the image file.'
            )

        try:
            logging.info(f'Loading image data from: {self.path}')

            # Open the image as a context manager to ensure file handle safety:
            with PillowImage.open(self.path) as opened_img:

                # Assign to 'img' and explicitly hint it as the generic base class.
                # This allows 'img' to accept results from .convert() and
                # .exif_transpose():
                img: PillowImage.Image = opened_img

                # Handle EXIF rotation (crucial for phone photos):
                if apply_exif_orientation:
                    # exif_transpose returns a copy with the rotation applied,
                    # or the original image if no EXIF data is present:
                    transposed = PillowImageOps.exif_transpose(img)

                    # Handle edge case where older Pillow versions might return
                    # None if no EXIF:
                    if transposed:
                        img = transposed

                # Handle mode conversion:
                if convert_mode is not None and img.mode != convert_mode:
                    img = img.convert(convert_mode)

                # Force load pixel data into memory:
                img.load()

            # Cache and return:
            self._pil_image = img
            return self._pil_image

        except OSError as e:
            logging.error(f'Failed to load image file at {self.path}: {e}')
            self._pil_image = None
            raise

    def load_image_from_url(
        self,
        url: str | None = None,
        url_key: str = 'thumb_original_url',
        convert_mode: str | None = None,
        force_reload: bool = False,
        apply_exif_orientation: bool = True,
        verbose: bool = True,
        session: requests.Session | None = None,
    ) -> PillowImage.Image:
        """
        Download the image into memory and cache it.

        This method bypasses ``path`` for the image entirely.
        Note that the ``is_downloaded`` property of the image will remain
        ``False`` after this method completes execution (unless the file
        happened to exist previously).

        Args:
            url (str, optional):
                Direct URL to download from. If ``None``, checks
                image properties to find the URL.
            url_key (str):
                Property key to use for URL if direct url is ``None``.
                Defaults to 'thumb_original_url'.
            convert_mode (str, optional):
                Mode to convert the loaded image to (e.g., ``'RGB'``).
            force_reload (bool):
                If ``True``, re-downloads even if _pil_image is already set.
            apply_exif_orientation (bool):
                If ``True``, rotates the image based on EXIF data (e.g.,
                fixing sideways phone photos). Defaults to ``True``.
            verbose (bool):
                If ``True``, logs standard info messages. Set to ``False``
                to suppress the "Downloading into memory" log. Defaults to
                ``True``.
            session (requests.Session, optional):
                Session for efficient connection pooling during download.

        Returns:
            PillowImage.Image: The loaded PIL image object.

        Raises:
            ValueError: If the URL cannot be determined.
            requests.RequestException: If the network request fails.
            IOError: If the bytes cannot be decoded as an image.

        Examples:
            Setup an asset defined only by metadata (no local file yet):

            >>> from rapidtools.core import ImageAsset
            >>> asset = ImageAsset(
            ...     id='wiki_cat',
            ...     path='virtual_cat.jpg',
            ...     allow_missing_file=True,
            ...     properties={
            ...         'thumb_original_url': (
            ...             'https://upload.wikimedia.org/wikipedia/commons/'
            ...             '3/3a/Cat03.jpg'
            ...         ),
            ...         'backup_url': (
            ...             'https://upload.wikimedia.org/wikipedia/commons/'
            ...             '4/47/PNG_transparency_demonstration_1.png'
            ...         ),
            ...     },
            ... )

            Loading image from the URL in properties (loads from
            ``'thumb_original_url'`` by default) into memory:

            >>> img = asset.load_image_from_url()
            INFO: Downloading wiki_cat into memory...
            >>> print(img.size)
            (1600, 1598)
            >>> print(asset.is_downloaded)
            False

            Load from a specific key, convert to Grayscale (``'L'``), force a
            re-download, and suppress the log message:

            >>> img_small = asset.load_image_from_url(
            ...     url_key='backup_url',
            ...     convert_mode='L',
            ...     force_reload=True,
            ...     verbose=False
            ... )
            >>> print(img_small.mode)
            L

            Load an image using a Session (recommended for batch
            processing. This example uses the TCP connection for speed:

            >>> import requests
            >>> with requests.Session() as session:
            ...     asset.load_image_from_url(session=session)
        """
        # If an image is already loaded and force reload is set to False,
        # exit method execution:
        if self._pil_image is not None and not force_reload:
            return self._pil_image

        # Determine the image URL:
        target_url = url or self.properties.get(url_key)

        if not target_url:
            available_keys = ', '.join(k for k in self.properties.keys() if 'url' in k)
            raise ValueError(
                f"Cannot load image for {self.id}: URL not provided and "
                f"property '{url_key}' not found.\nAvailable URL-like keys"
                f": {available_keys}"
            )

        try:
            # Only log if verbose is True
            if verbose:
                logging.info(f'Downloading {self.id} into memory...')

            # Use provided session or create a temporary one:
            requester = session if session else requests

            # Download bytes into memory:
            response = requester.get(
                target_url,
                headers=REQUESTS_HEADERS,
                stream=True,
                timeout=REQUESTS_TIMEOUT_VAL,
            )
            response.raise_for_status()

            # Create an in-memory file-like object:
            image_bytes = BytesIO(response.content)

            # Load into Pillow:
            raw_img = PillowImage.open(image_bytes)
            img: PillowImage.Image = raw_img

            # Perform EXIF rotation:
            if apply_exif_orientation:
                # We check for None to satisfy strict optional checking:
                transposed = PillowImageOps.exif_transpose(img)
                if transposed is not None:
                    img = transposed

            # Force decode and convert:
            img.load()  # Force decode

            if convert_mode is not None and img.mode != convert_mode:
                img = img.convert(convert_mode)

            # Cache and return:
            self._pil_image = img
            return self._pil_image

        except Exception as e:
            logging.error(f'Failed to load image from URL for {self.id}: {e}')
            self._pil_image = None
            raise

    def load_mask(
        self,
        mask_type: MaskType,
        force_reload: bool = False,
        custom_path: Path | str | None = None,
    ) -> np.ndarray:
        """
        Lazy-load the segmentation mask from disk into a NumPy array.

        This method handles caching automatically. It checks if the mask is
        already loaded in memory (e.g. `_semantic_mask`) before attempting
        file I/O.

        Args:
            mask_type (str):
                The type of mask to load. Standard values are ``'semantic'`` or
                ``'instance'``. Defaults to ``'semantic'``.
            force_reload (bool):
                If ``True``, ignores the internal cache and re-reads the file
                from disk. Defaults to ``False``.
            custom_path (Path, or str, or None):
                If provided, loads the mask from this specific path instead
                of deriving the filename from the image asset's path.

        Returns:
            np.ndarray:
                The segmentation mask data as a NumPy array.

        Raises:
            ValueError:
                If ``mask_type`` is invalid.
            FileNotFoundError:
                If the mask file does not exist at the calculated or provided
                path.
            OSError:
                If the file exists but cannot be opened or decoded
                (e.g. corruption).

        Examples:
            Create a mock environment with a real mask file by creating a dummy
            asset and a semantic mask file matching the expected naming
            convention:

            >>> import os
            >>> import numpy as np
            >>> from PIL import Image
            >>> from rapidtools.config import MaskType
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> asset = ImageAsset(
            ...     id='scene_01',
            ...     path='test_scene.jpg',
            ...     allow_missing_file=True
            ... )
            >>>
            >>> mask_data = np.zeros((10, 10), dtype=np.uint8)
            >>> mask_data[0:5, :] = 1  # Top half is class 1
            >>> mask_path = 'test_scene_semantic.png'
            >>> Image.fromarray(mask_data).save(mask_path)

            Load the semantic segmentation mask from 'test_scene_semantic.png'
            Please note that the ``load_mask`` method infers this filename
            automatically from the image filename and the provided mask_type
            by appending _semantic before the file extension
            (e.g., test_scene.png turns into test_scene_semantic.png):

            >>> loaded_mask = asset.load_mask(MaskType.SEMANTIC)
            >>> print(loaded_mask.shape)
            (10, 10)
            >>> print(loaded_mask[0, 0])
            1

            To delete the generated mask file, please run:

            >>> os.remove(mask_path)
        """

        # Normalize and validate input:
        try:
            valid_type = MaskType(mask_type)
        except ValueError:
            raise ValueError(
                f"Invalid mask_type '{mask_type}'. Expected one of: "
                f"{[m.value for m in MaskType]}"
            ) from None

        # Map Enum to internal cache attributes:
        cache_map = {
            MaskType.SEMANTIC: '_semantic_mask',
            MaskType.INSTANCE: '_instance_mask',
        }
        cache_attr = cache_map.get(valid_type)

        # Return cached if available:
        if cache_attr:
            current_val = getattr(self, cache_attr)
            if current_val is not None and not force_reload:
                return current_val

        # Locate file:
        mask_path = self.get_mask_path(valid_type, override_path=custom_path)

        if not mask_path.exists():
            raise FileNotFoundError(f'No {mask_type} mask found at {mask_path}')

        # Load and cache:
        try:
            with PillowImage.open(mask_path) as img:
                # Convert to numpy array:
                mask_data = np.array(img)

            # Cache it only if we have a slot for it
            if cache_attr:
                setattr(self, cache_attr, mask_data)

            return mask_data

        except OSError as e:
            logging.error(f'Failed to load {valid_type.value} mask for {self.id}: {e}')
            raise

    def merge(
        self,
        other: ImageAsset,
        ignore_id_mismatch: bool = False,
        overwrite_path: bool = False,
        overwrite_properties: bool = True,
    ) -> None:
        """
        Merge data from another ImageAsset.

        By default, this method ensures IDs match before merging to prevent
        accidental data corruption.

        Args:
            other (ImageAsset):
                The source asset to merge data from.
            ignore_id_mismatch (bool):
                If ``True``, allows merging even if self.id != other.id.
                Defaults to ``False``.
            overwrite_path (bool):
                If ``True``, updates self.path with other.path and adopts
                its internal caches (images/masks).
                Defaults to ``False`` (preserves original path).
            overwrite_properties (bool):
                If ``True``, other.properties, semantic_map, and instance_map
                will overwrite existing values in self.
                If ``False``, only missing values are filled.
                Defaults to ``True``.

        Raises:
            TypeError:
                If ``other`` is not an ``ImageAsset``.
            ValueError:
                If IDs do not match and ``ignore_id_mismatch`` is ``False``.

        Examples:
            Create two ImageAsset objects:

            >>> from rapidtools.core import ImageAsset
            >>>
            >>> asset_a = ImageAsset(
            ...     id='1',
            ...     path='a.jpg',
            ...     properties={'temp': 20},
            ...     allow_missing_file=True,
            ... )
            >>> asset_b = ImageAsset(
            ...     id='1',
            ...     path='b.jpg',
            ...     properties={'temp': 25, 'gps': [0, 0]},
            ...     allow_missing_file=True,
            ... )

            Merge ``asset_b``'s properties into ``asset_a`` using the default
            behavior that overwrites conflicts:

            >>> asset_a.merge(asset_b, overwrite_properties=True)
            >>> print(asset_a.properties)
            {'temp': 25, 'gps': [0, 0]}

            Merge without overwriting existing keys. Here, ``'gps'`` is added
            (it was not present on ``asset_a``), ``'temp'`` is NOT overwritten
            (stays 20):

            >>> asset_a = ImageAsset(
            ...     id='1',
            ...     path='a.jpg',
            ...     properties={'temp': 20},
            ...     allow_missing_file=True,
            ... )
            >>> asset_a.merge(asset_b, overwrite_properties=False)
            >>> print(asset_a.properties)
            {'temp': 20, 'gps': [0, 0]}
        """
        if not isinstance(other, type(self)):
            raise TypeError(f'Cannot merge {type(other).__name__} into ImageAsset.')

        # Check asset ID correspondence:
        if self.id != other.id and not ignore_id_mismatch:
            raise ValueError(
                f"Cannot combine assets: IDs do not match ('{self.id}' vs "
                f"'{other.id}'). Set ignore_id_mismatch=True to bypass this"
                "requirement."
            )

        # Merge asset properties:
        if overwrite_properties:
            # Update existing keys and add new ones:
            self.properties.update(other.properties)
        else:
            # Only add keys that do not exist:
            for key, value in other.properties.items():
                self.properties.setdefault(key, value)

        # Merge segmentation maps. If 'other' has a map, we decide whether to
        # take it based on the overwrite flag:
        if other.semantic_map is not None:
            if self.semantic_map is None or overwrite_properties:
                self.semantic_map = other.semantic_map

        if other.instance_map is not None:
            if self.instance_map is None or overwrite_properties:
                self.instance_map = other.instance_map

        # Merge asset path:
        if overwrite_path:
            self.path = other.path

            # Since the path changed, the old cache is invalid. Either adopt
            # the other's cache (if it exists) or reset to None:
            self._pil_image = other._pil_image

            # Similarly, if the path changed, the old mask likely does not
            # apply. Adopt the other's mask (whether it is data or None):
            self._semantic_mask = other._semantic_mask
            self._instance_mask = other._instance_mask

    def save_interactive_html(
        self,
        output_path: Path | str | None = None,
        mask_type: MaskType = MaskType.SEMANTIC,
        opacity: float = 0.5,
        min_area: int = 10,
    ) -> None:
        """
         Create an interactive HMTL file to view segmentation masks.

         Generates an interactive HTML file where masks are converted to SVG
         polygons. Hovering over a polygon shows its label from the asset's
         mapping data.

         This method converts the raster NumPy mask into vector polygons
         on the fly.

         Args:
             output_path (Path, or str, or None):
                 File path to save the HTML. Defaults to
                 {filename}_{mask_type}.html.
             mask_type (str):
                 Type of segmentation mask displayed, i.e., 'semantic' or
                 'instance'.
             opacity (float):
                 Opacity of the polygon fill (0.0 to 1.0).
             min_area (int):
                 Minimum number of vertices to generate a polygon for. Helps
                 reduce noise/dust in the SVG.

        Examples:
             Create a dummy image and a corresponding mask:

             >>> import os
             >>> import numpy as np
             >>> from PIL import Image
             >>> from rapidtools.core import ImageAsset
             >>> from rapidtools.config import MaskType
             >>>
             >>> img_path = 'html_test.jpg'
             >>> Image.new('RGB', (100, 100), color='white').save(img_path)
             >>> mask_data = np.zeros((100, 100), dtype=np.uint8)
             >>> mask_data[10:50, 10:50] = 1 # Square object
             >>>
             >>> asset = ImageAsset(id='viz', path=img_path, allow_missing_file=True)
             >>> asset.set_mask(mask_data, MaskType.SEMANTIC, map_data={1: 'Square'})

             Create an interactive HTML:

             >>> asset.save_interactive_html(mask_type=MaskType.SEMANTIC)
             INFO: Loading image data from: html_test.jpg
             INFO: Generating HTML polygons for 1 semantics...
             INFO: Saved interactive HTML to: html_test_semantic.html
             >>> print(os.path.exists('html_test_semantic.html'))
             True

             Please run the following lines to remove the created files:

             >>> for f in [img_path, 'html_test_semantic.html']:
             ...     if os.path.exists(f): os.remove(f)
        """
        # Input validation:
        try:
            valid_type = MaskType(mask_type)
        except ValueError:
            raise ValueError(f"Invalid mask_type '{mask_type}'.") from None

        # Load image and mask:
        try:
            # Load raw image for the background:
            pil_img = self.load_image_from_disk()
            mask = self.load_mask(valid_type)
        except FileNotFoundError:
            logging.error('Cannot generate HTML: Image or Mask not found.')
            return

        img_width, img_height = pil_img.size

        # Convert base image to Base64 so it can be embedded directly in
        # HTML:
        buffered = BytesIO()
        # Convert to RGB to ensure JPEG compatibility. Please note that
        # this drops the alpha channel if present:
        pil_img.convert('RGB').save(buffered, format='JPEG', quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Setup HTML structure:
        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: sans-serif;
                        background: #222;
                        color: #eee;
                        margin: 0;
                        padding: 20px;
                    }}
                    .container {{
                        position: relative;
                        width: {img_width}px;
                        height: {img_height}px;
                        margin: 0 auto;
                        box-shadow: 0 0 20px rgba(0,0,0,0.5);
                    }}
                    .bg-img {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        z-index: 1;
                    }}
                    .svg-layer {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        z-index: 2;
                    }}
                    polygon {{
                        stroke-width: 1;
                        stroke: rgba(255, 255, 255, 0.5);
                        transition: all 0.2s ease;
                        cursor: pointer;
                        /* Keeps stroke thin on zoom */
                        vector-effect: non-scaling-stroke;
                    }}
                    polygon:hover {{
                        stroke: #fff;
                        stroke-width: 2;
                        fill-opacity: 0.8 !important;
                    }}
                </style>
            </head>
            <body>
                <h2 style="text-align:center">
                    {self.filename} ({mask_type})
                </h2>
                <div class="container">
                    <img src="data:image/jpeg;base64,{img_str}" class="bg-img">
                    <svg
                        class="svg-layer"
                        viewBox="0 0 {img_width} {img_height}"
                        xmlns="http://www.w3.org/2000/svg"
                    >
            """

        # Processing logic (Raster -> Vector):
        # Identify unique objects in the mask (ignoring 0/background):
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]

        # Select Colormap consistent with show():
        cmap_name = (
            DEFAULT_SEMANTIC_CMAP if mask_type == 'semantic' else DEFAULT_INSTANCE_CMAP
        )
        colormap = plt.get_cmap(cmap_name)

        logging.info(f'Generating HTML polygons for {len(unique_ids)} {mask_type}s...')

        for obj_id in unique_ids:
            # Create binary mask for this specific object ID:
            obj_mask = (mask == obj_id).astype(np.uint8)

            # Pad the binary mask with zeros. This ensures contours are
            # closed even if they touch the image edge:
            padded_mask = np.pad(
                obj_mask, pad_width=1, mode='constant', constant_values=0
            )

            # Find contours using marching squares (skimage)
            # level=0.5 finds the boundary between 0 and 1:
            contours = measure.find_contours(padded_mask, level=0.5)

            # Map IDs to colors, ensuring IDs > 0 never map to index 0
            # (background):Ensure obj_id is a standard Python int for
            # .get() lookups:
            lookup_id = int(obj_id)
            color_idx = ((lookup_id - 1) % 255) + 1
            hex_color = mcolors.to_hex(colormap(color_idx))

            # Determine label/tooltip:
            if mask_type == 'semantic':
                label_text = (
                    self.semantic_map.get(lookup_id, 'unknown')
                    if self.semantic_map
                    else 'unknown'
                )
                tooltip = f'{label_text} (ID: {lookup_id})'
            else:
                info = self.instance_map.get(lookup_id) if self.instance_map else None
                label_text = str(info) if info else f'Object {lookup_id}'
                tooltip = f'{label_text}'

            for contour in contours:
                # Filter noise: Skip contours with too few points:
                if len(contour) < min_area:
                    continue

                # Subtract the 1-pixel padding offset pt[1] is Column (X),
                # pt[0] is Row (Y). Also, clip them to 0 -> width/height
                # to be safe:
                points = []
                for pt in contour:
                    # Clip to valid image bounds:
                    x = max(0, min(img_width, pt[1] - 1))
                    y = max(0, min(img_height, pt[0] - 1))
                    points.append(f'{x:.1f},{y:.1f}')

                points_str = ' '.join(points)

                html_content += (
                    f'<polygon points="{points_str}" fill="{hex_color}" '
                    f'fill-opacity="{opacity}" onclick="alert(\'{tooltip}\')">'
                    f'<title>{label_text}</title></polygon>\n'
                )

        # Create the footer and save:
        html_content += """
                    </svg>
                </div>
            </body>
            </html>
            """

        # Determine output path:
        if output_path:
            target_path = Path(output_path)
        else:
            suffix = f'_{mask_type}.html'
            target_path = self.path.with_name(f'{self.path.stem}{suffix}')

        # Write the HMTL file:
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logging.info(f'Saved interactive HTML to: {target_path}')

    def save_image(
        self,
        output_path: Path | str | None = None,
        format: str | None = None,
        quality: int = 100,
    ) -> None:
        """
        Saves the cached PIL image to disk.

        This method writes the in-memory image data to a file. It automatically
        creates any necessary parent directories.

        Args:
            output_path (Path, or str, or None):
                The file path where the image will be saved. If ``None``,
                defaults to ``self.path``, which will **overwrite** the
                original image file.
            format (str, or None):
                The image format to use (e.g., ``'JPEG'``, ``'PNG'``). If
                ``None``, the format is inferred from the file extension of the
                output path.
            quality (int):
                The compression quality (1-100) for lossy formats like JPEG.
                Higher numbers mean better quality but larger file size.
                Defaults to 100.

        Raises:
            ValueError:
                If no image is currently loaded in memory (i.e.,
                ``load_image_from_disk()``, ``load_image_from_url`` has not
                been called).
            OSError:
                If the file cannot be written (e.g., permission errors).

        Examples:
            Create an asset with a cached image:

            >>> import os
            >>> from PIL import Image
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> asset = ImageAsset(id='A1', path='input.jpg', allow_missing_file=True)
            >>> asset._pil_image = Image.new('RGB', (100, 100), color='blue')

            **Example 1: Save to a new path**

            >>> asset.save_image(output_path='output.png')
            INFO: Saving image to output.png
            >>> print(os.path.exists('output.png'))
            True

            **Example 2: Save as JPEG with specific quality**

            >>> asset.save_image(output_path='compressed.jpg', quality=50)
            INFO: Saving image to compressed.jpg

            **Cleanup**

            >>> for f in ['output.png', 'compressed.jpg']:
            ...     if os.path.exists(f): os.remove(f)
        """
        if self._pil_image is None:
            raise ValueError(
                "No image loaded in memory. Please call "
                "'load_image_from_disk()' or 'load_image_from_url()' before"
                " saving."
            )

        # Extract the target save path:
        target_path = Path(output_path) if output_path else self.path

        # Ensure the parent directory exists:
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the iamge:
        logging.info(f'Saving image to {target_path}')
        self._pil_image.save(target_path, format=format, quality=quality)

    def save_mask(
        self, mask_type: MaskType, output_path: Path | str | None = None
    ) -> None:
        """
        Save the specified image mask to disk as a PNG image.

        This method retrieves the mask (from memory cache or by reloading the
        source file) and writes it to the specified location. It strictly
        enforces the PNG format to ensure lossless storage of integer class IDs.

        Args:
            mask_type (MaskType):
                The type of mask to save (e.g., ``'instance'`` or ``'semantic'``).
            output_path (Path, or str, or None):
                The destination path. If ``None``, the method calculates the
                default path (e.g., ``image_name_semantic.png``) in the same
                directory as the source image.

        Raises:
            ValueError:
                If ``mask_type`` is invalid or if the mask data cannot be found
                (neither in memory nor on disk).
            OSError:
                If the file cannot be written (e.g., permission denied).

        Examples:
            Create an ``ImageAsset`` with an in-memory dummy mask (10x10 numpy
            array):

            >>> import os
            >>> import numpy as np
            >>> from rapidtools.core import ImageAsset
            >>> from rapidtools.config import MaskType
            >>>
            >>> asset = ImageAsset(
            ...     id='A1',
            ...     path='temp_asset.jpg',
            ...     allow_missing_file=True,
            ... )
            >>>
            >>> mask_data = np.zeros((10, 10), dtype=np.uint8)
            >>> asset.set_mask(mask_data, MaskType.SEMANTIC)

            Save the mask using the default naming convention:

            >>> asset.save_mask('semantic')
            INFO: Saving semantic mask to temp_asset_semantic.png
            >>> print(os.path.exists('temp_asset_semantic.png'))
            True

            Save to a specific custom path:

            >>> asset.save_mask('semantic', output_path='custom_mask.png')
            INFO: Saving semantic mask to custom_mask.png

            Remove the created files:

            >>> for f in ['temp_asset_semantic.png', 'custom_mask.png']:
            ...     if os.path.exists(f): os.remove(f)
        """

        # Validate input:
        try:
            valid_type = MaskType(mask_type)
        except ValueError:
            raise ValueError(
                f"Invalid mask_type '{mask_type}'. "
                f"Expected one of: {[m.value for m in MaskType]}"
            ) from None

        # Ensure the data exists (this checks cache and disk):
        try:
            mask_data = self.load_mask(valid_type)
        except FileNotFoundError:
            raise ValueError(
                f'Cannot save {valid_type} mask: No data found in memory.'
            ) from None

        # Determine path:
        target_path = (
            Path(output_path) if output_path else self.get_mask_path(mask_type)
        )

        # Prepare output directory:
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert NumPy array to PIL image and save as PNG (lossless):
        img = PillowImage.fromarray(mask_data)

        logging.info(f'Saving {valid_type} mask to {target_path}')
        img.save(target_path, format='PNG')

    def set_mask(
        self,
        data: np.ndarray,
        mask_type: MaskType,
        map_data: dict[int, Any] | None = None,
    ) -> None:
        """
        Manually attaches a segmentation mask from memory.

        Use this when you have generated a mask in code
        (e.g., model inference) instead of loading it from a file.

        Args:
            data:
                The array containing the mask.
            mask_type:
                Mask type, i.e., ``'semantic'`` or ``'instance'``.
            map_data:
                Optional dictionary to update the class/instance mapping.
                If provided, it replaces the existing map for this asset.

        Raises:
            TypeError: If ``data`` is not a NumPy array.
            ValueError: If ``mask_type`` is not 'semantic' or 'instance'.

        Example:
            Setup a virtual asset and mock segmentation data (representing
            output from an AI model):

            >>> import numpy as np
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> asset = ImageAsset(
            ...     id='img_01',
            ...     path='virtual.jpg',
            ...     allow_missing_file=True
            ... )

            Create a dummy 10x10 mask where 1=Sky, 2=Grass:

            >>> fake_mask = np.zeros((10, 10), dtype=np.uint8)
            >>> fake_mask[:5, :] = 1  # Top half is Sky
            >>> fake_mask[5:, :] = 2  # Bottom half is Grass

            Define what the numbers mean (i.e., a segmentation map):

            >>> labels = {1: 'Sky', 2: 'Grass'}

            Attach the mask and the labels to the asset:

            >>> asset.set_mask(fake_mask, 'semantic', map_data=labels)
            >>> print(asset.semantic_map)
            {1: 'Sky', 2: 'Grass'}

            Verify the mask is cached and retrievable:

            >>> retrieved = asset.load_mask('semantic')
            >>> print(retrieved.shape)
            (10, 10)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'Expected numpy.ndarray for mask data, got {type(data).__name__}.'
            )

        # Define attribute targets based on mask type
        # format: 'type': ('internal_cache_attr', 'public_map_attr'):
        target_attrs = {
            MaskType.SEMANTIC: ('_semantic_mask', 'semantic_map'),
            MaskType.INSTANCE: ('_instance_mask', 'instance_map'),
        }

        try:
            enum_key = MaskType(mask_type)
        except ValueError:
            raise ValueError(
                f"Unknown mask type '{mask_type}'. "
                f"Supported types: {list(target_attrs.keys())}"
            ) from None

        # Unpack attribute names:
        mask_attr, map_attr = target_attrs[enum_key]

        # Set the mask data:
        setattr(self, mask_attr, data)

        # Update the segmentation map if provided:
        if map_data is not None:
            setattr(self, map_attr, map_data)

    def show(
        self,
        output_type: Literal[
            'image', 'semantic', 'instance', 'overlay_semantic', 'overlay_instance'
        ] = 'image',
        alpha: float = 0.5,
        cmap: str | None = None,
    ) -> None:
        """
        Show the asset in the system's default image viewer.

        Args:
            output_type (str):
                The type of visualization to generate. Options:

                - ``'image'``: Show raw image only.
                - ``'semantic'``: Show semantic mask only (colored).
                - ``'instance'``: Show instance mask only.
                - ``'overlay_semantic'``: Image + Semantic Mask.
                - ``'overlay_instance'``: Image + Instance Mask.
            alpha (float):
                Transparency of the mask in overlay modes (0.0 to 1.0), where
                0 is fully transparent, 0.5 is 50% Transparent (see-through),
                and 1 is a solid color (blocks anything underneath it)
            cmap (str, optional):
                Matplotlib colormap name (e.g., ``'tab20'``, ``'jet'``).

        Raises:
            ValueError:
                If the provided ``cmap`` is not a valid Matplotlib colormap name.
            OSError:
                If a mask file exists but cannot be opened (e.g., file corruption),
                as this method only catches ``FileNotFoundError`` for masks.

        Examples:
            Create a dummy base image (100x100 gray square) and dummy semantic
            mask (inner square is class 1) and initialize an ``ImageAsset``
            with this image and its mask:

            >>> import os
            >>> import numpy as np
            >>> from PIL import Image
            >>> from rapidtools.core import ImageAsset
            >>> from rapidtools.config import MaskType
            >>>
            >>> img_path = "temp_viz_test.jpg"
            >>> Image.new('RGB', (100, 100), color='gray').save(img_path)
            >>>
            >>> mask_path = "temp_viz_test_semantic.png"
            >>> mask_data = np.zeros((100, 100), dtype=np.uint8)
            >>> mask_data[25:75, 25:75] = 1
            >>> Image.fromarray(mask_data).save(mask_path)
            >>>
            >>> asset = ImageAsset(id='viz_test', path=img_path)

            View base image:

            >>> asset.show(output_type='image')
            Loading image data from: temp_viz_test.jpg

            View semantic mask:

            >>> asset.show(output_type='semantic')

            Overlay the semantic mask with custom settings:

            >>> asset.show(
            ...     output_type='overlay_semantic',
            ...     alpha=0.3,
            ...     cmap='viridis'
            ... )

            Remove dummy image and mask files:

            >>> os.remove(img_path)
            >>> os.remove(mask_path)
        """
        # Check if the speficied output type is valid:
        valid_options = [
            'image',
            MaskType.SEMANTIC,
            MaskType.INSTANCE,
            f'overlay_{MaskType.SEMANTIC}',
            f'overlay_{MaskType.INSTANCE}',
        ]

        if output_type not in valid_options:
            logging.warning(
                f"The output type '{output_type}' is not supported. "
                f"Please specify one of: {', '.join(map(repr, valid_options))}"
            )
            return

        # Check if the image is requested, if so check if the base image exists
        # and load it if it exists:
        if output_type == 'image':
            try:
                self.load_image_from_disk().show(title=f'{self.filename} - Base image')
            except Exception as e:
                logging.error(f'Could not load/show base image: {e}')
            return

        # Determine mask type and defaults:
        if MaskType.SEMANTIC in output_type:
            target_mask_type = MaskType.SEMANTIC
            default_cmap = DEFAULT_SEMANTIC_CMAP
        else:
            target_mask_type = MaskType.INSTANCE
            default_cmap = DEFAULT_INSTANCE_CMAP

        # Helper to colorize a mask:
        def create_visual_mask(
            mask_type: MaskType, color_map_name: str
        ) -> PillowImage.Image | None:
            try:
                mask_data = self.load_mask(mask_type)
            except FileNotFoundError:
                logging.warning(f'No {mask_type} mask found to show.')
                return None

            # Get colormap and put it in a look up table for efficiency:
            colormap = plt.get_cmap(color_map_name)
            lut = (colormap(np.arange(256)) * 255).astype(np.uint8)

            # Apply colormap. Note that if you have > 255 instances, this
            # modulo wrap-around ensures the code does not crash, though colors
            # will repeat:
            if mask_data.dtype != np.uint8 or np.any(mask_data > 255):
                # (mask-1)%255 + 1 keeps objects in range 1-255
                mask_idx = (
                    (mask_data.astype(np.int32, copy=False) - 1) % 255 + 1
                ).astype(np.uint8)
                mask_idx[mask_data == 0] = 0  # Background is exactly 0
            else:
                mask_idx = mask_data

            # Apply look up table:
            colored_uint8 = lut[mask_idx]

            # Handle Transparency directly in NumPy (Channel 3 is Alpha)
            # Set Alpha to 0 where mask is background:
            colored_uint8[mask_data == 0, 3] = 0

            # Apply user alpha to visible objects:
            if alpha < 1.0:
                # Apply transparency to the visible parts
                visible_mask = mask_data > 0
                colored_uint8[visible_mask, 3] = (
                    colored_uint8[visible_mask, 3] * alpha
                ).astype(np.uint8)

            # Create PIL Image once from the finalized array:
            return PillowImage.fromarray(colored_uint8, mode='RGBA')

        # Create the mask image:
        final_image = create_visual_mask(target_mask_type, cmap or default_cmap)

        if final_image is None:
            return

        # Handle overlay iamges:
        if 'overlay' in output_type:
            try:
                base_img = self.load_image_from_disk().convert('RGBA')
                final_image = PillowImage.alpha_composite(base_img, final_image)
            except Exception as e:
                logging.warning(
                    f'Could not load base image for overlay ({e}). '
                    'Displaying mask only.'
                )

        # Display the image:
        if final_image:
            final_image.show(title=f'{self.filename} - {output_type}')

    def summary(self) -> None:
        """
        Print a human-readable summary of the image asset.

        This method prints details including the asset's ID, filename, path,
        status (downloaded/in-memory), file size (if available), and
        segmentation mask availability.

        Example:
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> img = ImageAsset(
            ...     id='img_01',
            ...     path='/tmp/photo.jpg',
            ...     allow_missing_file=True
            ... )
            >>> img.summary()
            --- ImageAsset Summary ---
            ID:          img_01
            Filename:    photo.jpg
            Status:      Missing (Virtual)
            Path:        /tmp/photo.jpg
            Masks:       Semantic [Not Loaded], Instance [Not Loaded]
            Properties (0):
              (No properties)
            --------------------------
        """
        header = '--- ImageAsset Summary ---'
        print(f'\n{header}')
        print(f'ID:          {self.id}')
        print(f'Filename:    {self.filename}')

        # Determine status:
        if self._pil_image:
            status = 'Loaded in Memory'
        elif self.is_downloaded:
            status = 'On Disk'
        elif self.allow_missing_file:
            status = 'Missing (Virtual)'
        else:
            status = 'Missing (Error)'

        print(f'Status:      {status}')
        print(f'Path:        {self.path}')

        # File size:
        if self.path.exists() and self.path.is_file():
            size_bytes = self.path.stat().st_size
            if size_bytes > 1024 * 1024:
                size_str = f'{size_bytes / (1024 * 1024):.2f} MB'
            elif size_bytes > 1024:
                size_str = f'{size_bytes / 1024:.2f} KB'
            else:
                size_str = f'{size_bytes} bytes'
            print(f'File Size:   {size_str}')

        # Segmentation info:
        sem_status = 'Available' if self._semantic_mask is not None else 'Not Loaded'
        inst_status = 'Available' if self._instance_mask is not None else 'Not Loaded'
        print(f'Masks:       Semantic [{sem_status}], Instance [{inst_status}]')

        # Properties:
        print(f'Properties ({len(self.properties)}):')
        if self.properties:
            print(json.dumps(self.properties, indent=4, default=str))
        else:
            print('  (No properties)')

        print('-' * len(header) + '\n')


class ImageCollection:
    """
    A container class for managing a group of ImageAsset objects.

    This class provides utilities for filtering, batch downloading, and
    exporting metadata for a collection of images.
    """

    def __init__(self, assets: list[ImageAsset] | None = None) -> None:
        """
        Initializes the ImageCollection.

        Args:
            assets: An optional list of ImageAsset objects to initialize the
                collection with. Defaults to an empty list.
        """
        self._assets: list[ImageAsset] = assets if assets else []

    def __contains__(self, item: str | ImageAsset) -> bool:
        """
        Checks if an image ID or ImageAsset object exists in the collection.
        """
        # Check by string ID by looping through the assets to see if any ID
        # matches the string:
        if isinstance(item, str):
            return any(asset.id == item for asset in self._assets)

        # Check by Asset object by comparing IDs to ensure matching assets are
        # captured even if they are different memory instances:
        if hasattr(item, 'id'):
            return any(asset.id == item.id for asset in self._assets)

        return False

    def __getitem__(self, index: int) -> ImageAsset:
        return self._assets[index]

    def __iter__(self) -> Iterator[ImageAsset]:
        return iter(self._assets)

    def __len__(self) -> int:
        return len(self._assets)

    def __repr__(self) -> str:
        return f'<ImageCollection containing {len(self)} image assets>'

    def add(
        self, items: ImageAsset | list[ImageAsset], overwrite: bool = False
    ) -> None:
        """
        Adds one or more assets to the collection.

        This method accepts either a single ImageAsset object or a list of
        objects. It checks for duplicates based on the asset ID.

        Args:
            items: A single ImageAsset object or a list of ImageAsset objects
                to add.
            overwrite: If True, existing assets with the same ID will be
                replaced by the new ones. If False, duplicates are ignored
                (the existing asset is kept). Defaults to False.
        """
        # Normalize input to a list for uniform processing:
        assets_to_add = items if isinstance(items, list) else [items]

        # Create a lookup map of {id: index} for the current collection:
        id_map = {asset.id: i for i, asset in enumerate(self._assets)}

        for asset in assets_to_add:
            # Handle assets with no ID (append without checking):
            if asset.id is None:
                self._assets.append(asset)
                continue

            if asset.id in id_map:
                if overwrite:
                    # Replace the element at the specific index:
                    index = id_map[asset.id]
                    self._assets[index] = asset
                else:
                    # Log that we are ignoring the duplicate:
                    logging.info(f'Skipping duplicate asset with ID: {asset.id}')
            else:
                # Add new asset:
                self._assets.append(asset)
                # Update map to handle duplicates within the input list itself:
                id_map[asset.id] = len(self._assets) - 1

    def merge(
        self,
        other: ImageCollection,
        overwrite_path: bool = False,
        overwrite_properties: bool = True,
        add_new: bool = False,
    ) -> None:
        """
        Merge with another ImageCollection.

        This method iterates through the incoming collection and either merges
        properties into existing assets (if IDs match) or adds new assets
        (if ``add_new`` is ``True``).

        Args:
            other (ImageCollection):
                The collection to merge with.
            overwrite_path (bool):
                If ``True``, existing assets will update their file paths
                to match the incoming assets. Defaults to ``False``.
            overwrite_properties (bool):
                If ``True``, incoming properties (metadata) overwrite existing
                ones on match. Defaults to ``True``.
            add_new (bool):
                If ``True``, assets in ``other`` that do not share an ID
                with any asset in this collection will be appended.
                If ``False``, unmatched assets are ignored. Defaults to
                ``False``.

        Examples:
            >>> master_images.merge(new_images, add_new=True)
            INFO: Merge complete: 5 merged, 2 added, 0 skipped.
        """
        # Call the internal core to do the work
        merged, added, skipped = self._merge_core(
            other,
            overwrite_path=overwrite_path,
            overwrite_properties=overwrite_properties,
            add_new=add_new,
        )

        logging.info(
            f'Merge complete: {merged} merged, {added} added, ' f'{skipped} skipped.'
        )

    def filter(self, func: Callable[[ImageAsset], bool]) -> ImageCollection:
        """
        Filters the collection using a custom function.

        Args:
            func: A callable that takes an ImageAsset as input and returns
                True if the asset should be kept, or False otherwise.

        Returns:
            A new ImageCollection instance containing only the assets for
            which the function returned True.

        Example:
            >>> subset = collection.filter(
            ...     lambda x: x.properties.get('damage') == 'High'
            ... )
        """
        filtered_assets = [asset for asset in self._assets if func(asset)]
        return ImageCollection(filtered_assets)

    def filter_by_property(self, key: str, value: Any) -> ImageCollection:
        """
        Filters assets where a specific property matches a given value.

        Args:
            key: The property key to check (e.g., 'event_name').
            value: The value the property must match.

        Returns:
            A new ImageCollection containing the matching assets.
        """
        return self.filter(lambda a: a.properties.get(key) == value)

    def filter_downloaded(self) -> ImageCollection:
        """
        Filters for assets that currently exist on disk.

        Returns:
            A new ImageCollection containing only assets where
            `is_downloaded` is True.
        """
        return self.filter(lambda a: a.is_downloaded)

    def filter_by_bbox(self, bbox: BoundingBox) -> ImageCollection:
        """
        Filters images that fall within a geographic bounding box.

        This method filters the collection to include only images whose
        coordinates (longitude/latitude) fall within the provided BoundingBox.

        Args:
            bbox: The BoundingBox object defining the geographic area.

        Returns:
            A new ImageCollection containing only the assets within the bounds.
        """
        # Extract bounds from the internal shapely geometry of the BoundingBox:
        min_lon, min_lat, max_lon, max_lat = bbox.shapely.bounds

        def inside_box(asset: ImageAsset) -> bool:
            lat = asset.properties.get('latitude')
            lon = asset.properties.get('longitude')

            # If coordinates are missing, exclude the image
            if lat is None or lon is None:
                return False

            return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

        return self.filter(inside_box)

    def get_ids(self, ignore_none: bool = True) -> list[str | int | None]:
        """
        Retrieves a list of IDs for all assets in the collection.

        Args:
            ignore_none: If ``True``, assets with no ID (None) are excluded
                         from the list. Defaults to ``True``.

        Returns:
            list: A list of IDs corresponding to the assets.
        """
        if ignore_none:
            return [asset.id for asset in self._assets if asset.id is not None]
        return [asset.id for asset in self._assets]

    def remove(self, items: ImageAsset | str | list[ImageAsset | str]) -> None:
        """
        Remove one or more assets from the collection.

        This method accepts a single ImageAsset, a single ID string, or a
        list containing a mix of both.

        Args:
            items:
                An ImageAsset object, a string ID, or a list of these to
                remove.

        Examples:
            Initialize an ImageCollection:

            >>> from rapidtools.core import ImageAsset, ImageCollection
            >>>
            >>> # Initialize a collection with some assets
            >>> img1 = ImageAsset(id='img_001', path='path/to/1.jpg',
            ... allow_missing_file=True)
            >>> img2 = ImageAsset(id='img_002', path='path/to/2.jpg',
            ... allow_missing_file=True)
            >>> img3 = ImageAsset(id='img_003', path='path/to/3.jpg',
            ... allow_missing_file=True)
            >>> collection = ImageCollection([img1, img2, img3])

            Remove by ID:
            >>> collection.remove('img_001')
            Removed 1 assets from the collection.
            >>> len(collection)
            2

            Remove by Object:
            >>> collection.remove(img2)
            Removed 1 assets from the collection.
            >>> len(collection)
            1

            Remove a list of mixed types:
            >>> collection.add([img1, img2])
            >>> collection.remove(['img_001', img2])
            Removed 2 assets from the collection.
            >>> len(collection)
            1
        """
        removed_count = self._remove_core(items)

        if removed_count > 0:
            suffix = 's' if removed_count != 1 else ''
            logging.info(f'Removed {removed_count} asset{suffix} from the collection.')
        else:
            logging.warning('No matching assets found to remove.')

    def subset(self, indices: list[int]) -> ImageCollection:
        """
        Creates a new ImageCollection containing assets at the specific indices.

        Args:
            indices: A list of integer indices (e.g., [0, 5, 10]).

        Returns:
            A new ImageCollection instance.

        Raises:
            IndexError: If any index is out of bounds.
        """
        # Extract the assets using list comprehension:
        selected_assets = [self._assets[i] for i in indices]

        # Return a new instance of the class:
        return ImageCollection(selected_assets)

    def download_all(self, max_workers: int = 5, overwrite: bool = False) -> None:
        """
        Downloads all images in the collection using multi-threading.

        Args:
            max_workers: The maximum number of parallel download threads.
                Defaults to 5.
            overwrite: If True, existing files will be re-downloaded and
                overwritten. Defaults to False.
        """
        total = len(self)
        logging.info(
            f'Starting batch download for {total} images with {max_workers} ' 'workers.'
        )

        # Create a single session to reuse TCP connections
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map futures to assets for error reporting
                future_to_asset = {
                    executor.submit(
                        asset.download, overwrite=overwrite, session=session
                    ): asset
                    for asset in self._assets
                }

                for i, future in enumerate(as_completed(future_to_asset)):
                    asset = future_to_asset[future]
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f'Failed to download {asset.id}: {e}')

                    # Optional: Print progress every 10 images
                    if (i + 1) % 10 == 0:
                        logging.info(f'Progress: {i + 1}/{total} completed.')

    def to_dataframe(self) -> Any:
        """
        Converts the collection metadata to a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame where each row represents an asset.
                Columns include 'id', 'path', 'is_downloaded', and all keys
                found in the asset properties.

        Raises:
            ImportError: If the pandas library is not installed.
        """
        data = []
        for asset in self._assets:
            # Flattens structure: ID, Path + all Properties
            row = {
                'id': asset.id,
                'path': str(asset.path),
                'is_downloaded': asset.is_downloaded,
                **asset.properties,
            }
            data.append(row)

        return pd.DataFrame(data)

    def to_json(self, filepath: str | Path) -> None:
        """
        Exports the collection metadata to a JSON file.

        Args:
            filepath: The path where the JSON file will be saved.
        """
        data = [
            {'id': a.id, 'path': str(a.path), 'properties': a.properties}
            for a in self._assets
        ]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _merge_core(
        self,
        other: ImageCollection,
        overwrite_path: bool,
        overwrite_properties: bool,
        add_new: bool,
    ) -> tuple[int, int, int]:
        """
        Internal logic to merge with another ImageCollection.

        See :meth:`merge` for argument definitions.

        Returns:
            tuple[int, int, int]:
                A tuple containing counts for
            (merged, added, skipped).
        """
        # Create a lookup map for the current assets for fast access:
        current_map = {
            asset.id: asset for asset in self._assets if asset.id is not None
        }

        count_merged = 0
        count_added = 0
        count_skipped = 0

        for incoming_asset in other:
            # Merge if a match is found:
            if incoming_asset.id is not None and incoming_asset.id in current_map:

                existing_asset = current_map[incoming_asset.id]
                existing_asset.merge(
                    incoming_asset,
                    ignore_id_mismatch=False,
                    overwrite_path=overwrite_path,
                    overwrite_properties=overwrite_properties,
                )
                count_merged += 1

            # If no match is found and add_new=True, add the asset:
            elif add_new:
                self._assets.append(incoming_asset)
                # Update map in case 'other' has duplicates of this new ID:
                if incoming_asset.id is not None:
                    current_map[incoming_asset.id] = incoming_asset
                count_added += 1

            # If no match is found and add_new=False, skip:
            else:
                count_skipped += 1

        return count_merged, count_added, count_skipped

    def _remove_core(self, items: ImageAsset | str | list[ImageAsset | str]) -> int:
        """
        Core logic to remove assets.

        This performs the list filtering without logging side effects.
        See :meth:`remove` for argument definitions.

        Returns:
            int: The number of assets removed.
        """
        # Normalize input to a list
        items_to_remove = items if isinstance(items, list) else [items]

        ids_to_remove = set()
        object_addresses_to_remove = set()

        for item in items_to_remove:
            if isinstance(item, str):
                ids_to_remove.add(item)
            elif hasattr(item, 'id'):
                # Store the memory address of the object for identity
                # comparison:
                object_addresses_to_remove.add(id(item))

                # Also track the ID string if it exists:
                if item.id is not None:
                    ids_to_remove.add(item.id)

        initial_count = len(self._assets)

        # Rebuild the list preserving order.
        # We keep the asset if:
        # 1. Its ID string is NOT in the removal set
        #    AND
        # 2. Its memory address is NOT in the removal set
        self._assets = [
            asset
            for asset in self._assets
            if asset.id not in ids_to_remove
            and id(asset) not in object_addresses_to_remove
        ]

        return initial_count - len(self._assets)
