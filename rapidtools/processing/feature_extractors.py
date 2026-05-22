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
# 05-16-2026

import logging
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm import tqdm

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.data_sources import OrthomosaicReader
from rapidtools.models import SAM3Inference


class SAM3OrthoFeatureExtractor:
    """
    Pipeline component that uses local SAM 3 to discover and extract real-world 
    assets from high-resolution orthomosaic rasters using a multi-scale approach.
    """

    def __init__(
        self,
        prompt: str,
        patch_size: float = 30.0,
        unit: str = 'feet',
        overlap_ratio: float = 0.15,
        model_id: str = 'facebook/sam3',
        device: str = 'auto',
        batch_size: int = 4,
        load_in_4bit: bool = True,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_missing_data_ratio: float = 0.95 
    ):
        self.prompt = prompt
        self.patch_size = patch_size
        self.unit = unit
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.max_missing_data_ratio = max_missing_data_ratio
        
        logging.info(
            f"Initializing SAM3OrthoFeatureExtractor for '{self.prompt}' "
            f"at scale: {self.patch_size} {self.unit} with threshold {self.threshold}..."
        )
        
        self.model = SAM3Inference(
            model_id=model_id,
            device=device,
            load_in_4bit=load_in_4bit
        )

    def __call__(self, raster_path: str | Path) -> PhysicalAssetCollection:
        """
        Scan the raster, extract features, and merge overlaps into a new collection.
        """
        raw_polygons = []
        
        with OrthomosaicReader(raster_path) as reader:
            tile_generator = reader.generate_tiles(
                patch_size=self.patch_size, 
                unit=self.unit, 
                overlap_ratio=self.overlap_ratio, 
                max_missing_data_ratio=self.max_missing_data_ratio,
                pad_edge_tiles=True 
            )
            
            batch_images = []
            batch_bounds = []
            
            for pil_image, wgs84_bounds in tqdm(tile_generator, desc=f"Scanning for '{self.prompt}'"):
                batch_images.append(pil_image)
                batch_bounds.append(wgs84_bounds)
                
                if len(batch_images) == self.batch_size:
                    self._process_batch(batch_images, batch_bounds, raw_polygons)
                    batch_images, batch_bounds = [], []
                    
            # Process any remaining images in the final partial batch
            if batch_images:
                self._process_batch(batch_images, batch_bounds, raw_polygons)

        logging.info(f"Extracted {len(raw_polygons)} raw polygons. Merging overlaps...")
        final_collection = PhysicalAssetCollection()
        
        if raw_polygons:
            # unary_union automatically melts overlapping geometries together
            merged_geometry = unary_union(raw_polygons)
            
            # Unpack the results safely depending on what unary_union returns
            if merged_geometry.geom_type == 'Polygon':
                final_geometries = [merged_geometry]
            elif merged_geometry.geom_type == 'MultiPolygon':
                final_geometries = list(merged_geometry.geoms)
            else:
                final_geometries = [
                    g for g in merged_geometry.geoms 
                    if g.geom_type in ['Polygon', 'MultiPolygon']
                ]

            # Wrap the unified geometries into PhysicalAsset objects
            for geom in final_geometries:
                asset = PhysicalAsset(
                    id=f"{self.prompt.replace(' ', '_')}_{uuid.uuid4().hex[:8]}",
                    geometry=geom,
                    attributes={
                        "asset_type": self.prompt,
                        "source_model": self.model.model_id,
                        "extraction_scale": f"{self.patch_size}_{self.unit}"
                    }
                )
                final_collection.add(asset)

        logging.info(f"Extraction complete. Yielded {len(final_collection)} unique assets.")
        return final_collection

    def _process_batch(self, batch_images: list, batch_bounds: list, raw_polygons: list):
        """
        Helper to safely save PIL images to disk, run inference, and map 
        pixel masks back to real-world Shapely polygons.
        """
        temp_paths = []
        
        # 1. Save in-memory PIL images to temporary files
        for img in batch_images:
            fd, path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            # Save as JPEG for fast disk I/O
            img.save(path, format='JPEG')
            temp_paths.append(path)
            
        try:
            # 2. Run inference using the file paths WITH threshold parameters
            outputs = self.model.run_inference(
                image_inputs=temp_paths, 
                prompt=self.prompt,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold
            )
            
            if not outputs or getattr(outputs, 'masks', None) is None:
                return
                
            # 3. Translate pixel masks to geographic polygons
            for image_masks, bounds in zip(outputs.masks, batch_bounds):
                if image_masks is None or len(image_masks) == 0:
                    continue
                
                # --- FIX: Force 2D arrays to become 3D arrays (1, H, W) ---
                if image_masks.ndim == 2:
                    image_masks = image_masks[np.newaxis, ...]
                    
                min_lon, min_lat, max_lon, max_lat = bounds
                
                # --- FIX: Safely grab height and width from the last two dimensions ---
                height, width = image_masks.shape[-2:]
                
                # Build an affine transform for this specific tile
                transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
                
                for instance_mask in image_masks:
                    # Binarize the float/boolean mask
                    binary_mask = (instance_mask > 0.5).astype(np.uint8)
                    
                    # Extract the shapes natively using rasterio
                    for geom_dict, val in rasterio.features.shapes(binary_mask, transform=transform):
                        if val == 1:
                            poly = shape(geom_dict)
                            if poly.is_valid and not poly.is_empty:
                                raw_polygons.append(poly)
        finally:
            # 4. Always clean up temporary files to prevent disk bloat
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)