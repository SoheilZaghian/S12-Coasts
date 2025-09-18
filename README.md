# S12-Coasts Benchmark Dataset Generator

The S12-Coasts project provides an automated pipeline to build a global coastline segmentation benchmark. It creates a multimodal dataset using Sentinel-1 SAR and Sentinel-2 optical imagery, integrating vector shoreline data from NOAA (CUSP), OpenStreetMap, and HydroLAKES to generate reference water masks. The resulting dataset contains 6,637 image patches (512×512 pixels) covering 33 diverse coastal and inland-water regions worldwide. This broad coverage and automated framework overcome the scarcity of large-scale labeled datasets in coastline mapping, enabling training and evaluation of deep-learning models (e.g. SegFormer) with robust transferability across different environments.

## Data and Patch Extraction

The code retrieves Sentinel-1 and Sentinel-2 data via Google Earth Engine. It stratifies coastline vectors to place patch centroids along shorelines, then extracts 512×512-pixel tiles (at 10 m GSD) from both sensors. Sentinel-1 C-band SAR scenes (dual-polarization VV/VH) are preprocessed (radiometric calibration, speckle filtering, terrain correction, resampled to 10 m). Sentinel-2 MSI scenes are accessed as atmospherically corrected surface reflectance, with scenes exceeding 30% cloud cover excluded and patches chosen to minimize the time gap between the S1/S2 acquisitions. In practice, each centroid defines a 5120 m×5120 m (512×512 px) window that is clipped from the aligned S1/S2 imagery, forming the image component of S12-Coasts.

## Water Boundaries and Mask Generation

Vector water boundaries from OpenStreetMap, NOAA (CUSP), and HydroLAKES are used as ground truth for coastlines. HydroLAKES provides polygonal lake outlines that can be rasterized directly into binary water masks. In contrast, OSM and NOAA supply coastline polylines, which only mark the water edges. To convert these into full-water masks, the pipeline applies an Ensemble Object-Based Clustering (EOBC) approach. First, each polyline is rasterized and connected components are labeled. Then, multiple spectral indices (MNDWI, NDVI, AEWI, WRI, NDPI from Sentinel-2) and Sentinel-1 backscatter features are computed for each region. These features feed an ensemble of clustering algorithms (K-means, GMM, etc.) that partition regions into water vs. non-water. The cluster with the highest mean MNDWI is labeled as water. Finally, a majority-vote across all clustering outputs produces a single robust binary water mask for each patch. The mask component is then paired with the corresponding S1/S2 image, completing the dataset.

Usage

To reproduce the S12-Coasts dataset, users can run the provided pipeline scripts:

Extract patches: Run the patch-generation script to sample Sentinel-1/2 imagery at coastal centroids (using Google Earth Engine). This produces aligned 512×512 image tiles.

Generate masks: Execute the mask-generation script. It ingests the same centroids and vector shorelines, rasterizes polylines/polygons, and applies the EOBC clustering to output binary water masks.

Compile dataset: The output images and masks are saved in the specified directories, matching the S12-Coasts structure.

See the repository README for details on configuration (e.g. region lists) and execution. By following these steps, users can fully regenerate the S12-Coasts benchmark (images + water masks) for any set of coastal regions, enabling consistent training and evaluation of coastline segmentation methods.
