#!/usr/bin/env python3
"""
Join Utilities Module

Utilities for spatial and attribute joins between shapefiles and tabular data.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
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

logger = logging.getLogger("JoinUtils")

def join_attributes(gdf: gpd.GeoDataFrame, df: pd.DataFrame, key: str, 
                   how: str = 'left', lsuffix: str = '_left', rsuffix: str = '_right') -> gpd.GeoDataFrame:
    """Join attributes from a DataFrame to a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame to join to
        df: DataFrame to join from
        key: Column name to join on
        how: Join type ('left', 'right', 'inner', 'outer')
        lsuffix: Suffix for left DataFrame columns
        rsuffix: Suffix for right DataFrame columns
        
    Returns:
        GeoDataFrame: Joined GeoDataFrame
    """
    if gdf is None or df is None:
        logger.warning("Cannot join attributes: GeoDataFrame or DataFrame is None")
        return None
        
    if key not in gdf.columns:
        logger.warning(f"Key '{key}' not found in GeoDataFrame")
        return gdf
        
    if key not in df.columns:
        logger.warning(f"Key '{key}' not found in DataFrame")
        return gdf
        
    logger.info(f"Joining attributes with key '{key}' and join type '{how}'")
    result = gdf.merge(df, on=key, how=how, suffixes=(lsuffix, rsuffix))
    logger.info(f"Join resulted in {len(result)} features")
    return result

def safe_spatial_join(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                     how: str = "inner", op: str = "intersects",
                     lsuffix: str = '_1', rsuffix: str = '_2') -> gpd.GeoDataFrame:
    """Perform a spatial join with error handling and CRS alignment.
    
    Args:
        gdf1: First GeoDataFrame
        gdf2: Second GeoDataFrame
        how: Join type ('inner', 'left', 'right')
        op: Spatial operation ('intersects', 'contains', 'within', etc.)
        lsuffix: Suffix for left DataFrame columns
        rsuffix: Suffix for right DataFrame columns
        
    Returns:
        GeoDataFrame: Joined GeoDataFrame
    """
    if gdf1 is None or gdf2 is None:
        logger.warning("Cannot perform spatial join: GeoDataFrame is None")
        return None
        
    # Ensure both GeoDataFrames have the same CRS
    if gdf1.crs != gdf2.crs:
        logger.info(f"Reprojecting second GeoDataFrame to match CRS: {gdf1.crs}")
        gdf2 = gdf2.to_crs(gdf1.crs)
        
    try:
        logger.info(f"Performing spatial join with operation '{op}' and join type '{how}'")
        joined = gpd.sjoin(gdf1, gdf2, how=how, op=op, lsuffix=lsuffix, rsuffix=rsuffix)
        logger.info(f"Spatial join resulted in {len(joined)} features")
        return joined
    except Exception as e:
        logger.error(f"Error performing spatial join: {e}")
        return None

def join_nearest(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                max_distance: Optional[float] = None,
                lsuffix: str = '_1', rsuffix: str = '_2') -> gpd.GeoDataFrame:
    """Join each feature in gdf1 to its nearest neighbor in gdf2.
    
    Args:
        gdf1: First GeoDataFrame
        gdf2: Second GeoDataFrame
        max_distance: Maximum distance to consider (in CRS units)
        lsuffix: Suffix for left DataFrame columns
        rsuffix: Suffix for right DataFrame columns
        
    Returns:
        GeoDataFrame: Joined GeoDataFrame with distance column
    """
    if gdf1 is None or gdf2 is None:
        logger.warning("Cannot perform nearest join: GeoDataFrame is None")
        return None
        
    # Ensure both GeoDataFrames have the same CRS
    if gdf1.crs != gdf2.crs:
        logger.info(f"Reprojecting second GeoDataFrame to match CRS: {gdf1.crs}")
        gdf2 = gdf2.to_crs(gdf1.crs)
        
    from rtree import index
    
    logger.info("Creating spatial index for nearest neighbor join")
    # Create spatial index for gdf2
    spatial_index = index.Index()
    for idx, geom in enumerate(gdf2.geometry):
        spatial_index.insert(idx, geom.bounds)
        
    # Find nearest neighbors
    neighbors = []
    logger.info(f"Finding nearest neighbors for {len(gdf1)} features")
    for idx1, row1 in gdf1.iterrows():
        geom1 = row1.geometry
        nearest_idx = None
        nearest_dist = float('inf')
        
        # Use spatial index to find potential neighbors
        for idx2 in spatial_index.intersection(geom1.bounds):
            geom2 = gdf2.iloc[idx2].geometry
            dist = geom1.distance(geom2)
            
            if dist < nearest_dist and (max_distance is None or dist <= max_distance):
                nearest_dist = dist
                nearest_idx = idx2
        
        if nearest_idx is not None:
            neighbors.append({
                'left_idx': idx1,
                'right_idx': gdf2.index[nearest_idx],
                'distance': nearest_dist
            })
    
    if not neighbors:
        logger.warning("No nearest neighbors found within distance threshold")
        return None
        
    # Create join DataFrame
    join_df = pd.DataFrame(neighbors)
    
    # Merge with original GeoDataFrames
    result = gdf1.loc[join_df.left_idx].merge(
        gdf2.loc[join_df.right_idx],
        left_index=True,
        right_index=True,
        suffixes=(lsuffix, rsuffix)
    )
    
    # Add distance column
    result['distance'] = join_df.distance.values
    
    logger.info(f"Nearest join resulted in {len(result)} features")
    return result

def spatial_overlay(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                   how: str = 'intersection',
                   keep_geom_type: bool = True) -> gpd.GeoDataFrame:
    """Perform a spatial overlay operation.
    
    Args:
        gdf1: First GeoDataFrame
        gdf2: Second GeoDataFrame
        how: Overlay operation ('intersection', 'union', 'difference', 'symmetric_difference')
        keep_geom_type: Whether to keep only geometries of the same type as gdf1
        
    Returns:
        GeoDataFrame: Result of overlay operation
    """
    if gdf1 is None or gdf2 is None:
        logger.warning("Cannot perform spatial overlay: GeoDataFrame is None")
        return None
        
    # Ensure both GeoDataFrames have the same CRS
    if gdf1.crs != gdf2.crs:
        logger.info(f"Reprojecting second GeoDataFrame to match CRS: {gdf1.crs}")
        gdf2 = gdf2.to_crs(gdf1.crs)
        
    try:
        logger.info(f"Performing spatial overlay with operation '{how}'")
        result = gpd.overlay(gdf1, gdf2, how=how, keep_geom_type=keep_geom_type)
        logger.info(f"Overlay resulted in {len(result)} features")
        return result
    except Exception as e:
        logger.error(f"Error performing spatial overlay: {e}")
        return None