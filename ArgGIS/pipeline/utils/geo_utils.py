#!/usr/bin/env python3
"""
Geo Utilities Module

Lightweight helpers that work with any GeoDataFrame (separate from shapefiles).
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon, MultiLineString
from rtree import index
import logging
from typing import List, Dict, Union, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GeoUtils")

def ensure_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Ensure a GeoDataFrame has the specified CRS, reprojecting if necessary.
    
    Args:
        gdf: GeoDataFrame to check
        crs: Target CRS
        
    Returns:
        GeoDataFrame: GeoDataFrame with the specified CRS
    """
    if gdf is None:
        logger.warning("Cannot ensure CRS: GeoDataFrame is None")
        return None
        
    if gdf.crs is None:
        logger.warning("GeoDataFrame has no CRS, setting to specified CRS")
        gdf.crs = crs
        return gdf
        
    if str(gdf.crs) != str(crs):
        logger.info(f"Reprojecting from {gdf.crs} to {crs}")
        return gdf.to_crs(crs)
    else:
        return gdf

def explode_multipolygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode MultiPolygons into individual Polygons.
    
    Args:
        gdf: GeoDataFrame to explode
        
    Returns:
        GeoDataFrame: Exploded GeoDataFrame
    """
    if gdf is None:
        logger.warning("Cannot explode MultiPolygons: GeoDataFrame is None")
        return None
        
    # Check if there are any MultiPolygons
    if not any(gdf.geometry.type.isin(['MultiPolygon'])):
        logger.info("No MultiPolygons to explode")
        return gdf
        
    logger.info("Exploding MultiPolygons")
    exploded = gdf.explode(index_parts=True)
    logger.info(f"Exploded {len(gdf)} features into {len(exploded)} features")
    return exploded

def generate_spatial_index(gdf: gpd.GeoDataFrame):
    """Generate a spatial index for a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame to index
        
    Returns:
        rtree.index.Index: Spatial index
    """
    if gdf is None:
        logger.warning("Cannot generate spatial index: GeoDataFrame is None")
        return None
        
    logger.info("Generating spatial index")
    spatial_index = index.Index()
    for idx, geom in enumerate(gdf.geometry):
        spatial_index.insert(idx, geom.bounds)
        
    return spatial_index

def calculate_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate centroids for all geometries.
    
    Args:
        gdf: GeoDataFrame to process
        
    Returns:
        GeoDataFrame: GeoDataFrame with centroids
    """
    if gdf is None:
        logger.warning("Cannot calculate centroids: GeoDataFrame is None")
        return None
        
    logger.info("Calculating centroids")
    centroids = gdf.copy()
    centroids.geometry = centroids.geometry.centroid
    return centroids

def validate_geometries(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Validate geometries and report issues.
    
    Args:
        gdf: GeoDataFrame to validate
        
    Returns:
        DataFrame: Validation results
    """
    if gdf is None:
        logger.warning("Cannot validate geometries: GeoDataFrame is None")
        return None
        
    from shapely import validation
    
    logger.info("Validating geometries")
    validation_results = []
    for idx, geom in enumerate(gdf.geometry):
        is_valid = geom.is_valid
        validation_results.append({
            'index': idx,
            'is_valid': is_valid,
            'issue': None if is_valid else validation.explain_validity(geom)
        })
    
    invalid_count = sum(1 for r in validation_results if not r['is_valid'])
    logger.info(f"Found {invalid_count} invalid geometries out of {len(validation_results)}")
    return pd.DataFrame(validation_results)

def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix invalid geometries.
    
    Args:
        gdf: GeoDataFrame to fix
        
    Returns:
        GeoDataFrame: Fixed GeoDataFrame
    """
    if gdf is None:
        logger.warning("Cannot fix geometries: GeoDataFrame is None")
        return None
        
    logger.info("Fixing invalid geometries")
    fixed = gdf.copy()
    fixed.geometry = fixed.geometry.buffer(0)  # Buffer with 0 distance to fix self-intersections
    return fixed

def clip_to_bounds(gdf: gpd.GeoDataFrame, bounds: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Clip a GeoDataFrame to specified bounds.
    
    Args:
        gdf: GeoDataFrame to clip
        bounds: Bounds as (minx, miny, maxx, maxy)
        
    Returns:
        GeoDataFrame: Clipped GeoDataFrame
    """
    if gdf is None:
        logger.warning("Cannot clip to bounds: GeoDataFrame is None")
        return None
        
    from shapely.geometry import box
    
    logger.info(f"Clipping to bounds: {bounds}")
    bbox = box(*bounds)
    return gpd.clip(gdf, bbox)

def simplify_geometries(gdf: gpd.GeoDataFrame, tolerance: float, preserve_topology: bool = True) -> gpd.GeoDataFrame:
    """Simplify geometries to reduce complexity.
    
    Args:
        gdf: GeoDataFrame to simplify
        tolerance: Simplification tolerance
        preserve_topology: Whether to preserve topology
        
    Returns:
        GeoDataFrame: Simplified GeoDataFrame
    """
    if gdf is None:
        logger.warning("Cannot simplify geometries: GeoDataFrame is None")
        return None
        
    logger.info(f"Simplifying geometries with tolerance {tolerance}")
    simplified = gdf.copy()
    simplified.geometry = simplified.geometry.simplify(tolerance, preserve_topology=preserve_topology)
    return simplified