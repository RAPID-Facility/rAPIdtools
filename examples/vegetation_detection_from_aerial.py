from rapidtools.processing.feature_extractors import SAM3OrthoFeatureExtractor

def main():
    extractor = SAM3OrthoFeatureExtractor(
        prompt="vegetation",  # Multiple descriptive terms
        patch_size=50,                        # Drop from 300 to 100 meters to keep texture visible
        unit="meters", 
        overlap_ratio=0.25,
        batch_size=1,
        threshold=0.1,         # Force SAM to accept lower-confidence object boxes
        mask_threshold=0.1     # Force SAM to generate masks even if unsure
    )
    
    vegetation_assets = extractor('eaton_patch_20250214.tiff')
    vegetation_assets.to_geojson('vegetation_merged.geojson')
    
    print(f"Successfully saved {len(vegetation_assets)} merged polygons")

if __name__ == '__main__':
    main()