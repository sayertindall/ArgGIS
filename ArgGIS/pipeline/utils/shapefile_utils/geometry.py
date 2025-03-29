#!/usr/bin/env python3
"""
Geometry Module

Provides functionality for geometry operations on shapefiles,
including simplification, buffering, validation, and centroids.
"""

import geopandas as gpd
import pandas as pd
from shapely import validation
from typing import Dict, Optional, Union, Any

from .logger import logger
from .base import BaseShapefile

class GeometryMixin(BaseShapefile):
    """Mixin class for geometry operations on shapefiles.
    
    This class provides methods for simplifying geometries, buffering,
    validation, and calculating centroids.
    """
    
    def get_centroids(self) -> gpd.GeoDataFrame:
        """Calculate centroids of all geometries.
        
        Returns:
            GeoDataFrame: Centroids of all geometries
        """
        if self.gdf is None:
            logger.warning("Cannot get centroids: GeoDataFrame is None")
            return None
            
        logger.info("Calculating centroids")
        centroids = self.gdf.copy()
        centroids.geometry = centroids.geometry.centroid
        return centroids
        
    def buffer_features(self,
                       distance: float,
                       unit: str = 'km',
                       dissolve: bool = False) -> gpd.GeoDataFrame:
        """Buffer features by a specified distance.
        
        Args:
            distance: Buffer distance
            unit: Distance unit ('km', 'm', or 'mi')
            dissolve: Whether to dissolve the buffers
            
        Returns:
            GeoDataFrame: Buffered geometries
        """
        if self.gdf is None:
            logger.warning("Cannot buffer features: GeoDataFrame is None")
            return None
            
        logger.info(f"Buffering features by {distance} {unit}")
        # Reproject to equal area projection for accurate buffering
        equal_area = self.reproject('EPSG:6933')  # World Equal Area
        
        # Convert distance to meters
        if unit == 'km':
            distance_m = distance * 1000
        elif unit == 'mi':
            distance_m = distance * 1609.34
        else:  # m
            distance_m = distance
        
        # Buffer the geometries
        buffered = equal_area.copy()
        buffered.geometry = buffered.geometry.buffer(distance_m)
        
        # Dissolve if requested
        if dissolve:
            logger.info("Dissolving buffered geometries")
            buffered = buffered.dissolve()
        
        # Reproject back to original CRS
        buffered = buffered.to_crs(self.original_crs)
        
        return buffered
    
    def simplify_geometries(self,
                           tolerance: float,
                           preserve_topology: bool = True) -> gpd.GeoDataFrame:
        """Simplify geometries to reduce complexity.
        
        Args:
            tolerance: Simplification tolerance
            preserve_topology: Whether to preserve topology
            
        Returns:
            GeoDataFrame: Simplified geometries
        """
        if self.gdf is None:
            logger.warning("Cannot simplify geometries: GeoDataFrame is None")
            return None
            
        logger.info(f"Simplifying geometries with tolerance {tolerance}")
        simplified = self.gdf.copy()
        simplified.geometry = simplified.geometry.simplify(tolerance, preserve_topology=preserve_topology)
        return simplified
    
    def validate_geometries(self) -> pd.DataFrame:
        """Validate geometries and report issues.
        
        Returns:
            DataFrame: Validation results
        """
        if self.gdf is None:
            logger.warning("Cannot validate geometries: GeoDataFrame is None")
            return None
            
        logger.info("Validating geometries")
        validation_results = []
        for idx, geom in enumerate(self.gdf.geometry):
            is_valid = geom.is_valid
            validation_results.append({
                'index': idx,
                'is_valid': is_valid,
                'issue': None if is_valid else validation.explain_validity(geom)
            })
        
        invalid_count = sum(1 for r in validation_results if not r['is_valid'])
        logger.info(f"Found {invalid_count} invalid geometries out of {len(validation_results)}")
        return pd.DataFrame(validation_results)
    
    def fix_geometries(self) -> gpd.GeoDataFrame:
        """Fix invalid geometries.
        
        Returns:
            GeoDataFrame: Fixed geometries
        """
        if self.gdf is None:
            logger.warning("Cannot fix geometries: GeoDataFrame is None")
            return None
            
        logger.info("Fixing invalid geometries")
        fixed = self.gdf.copy()
        fixed.geometry = fixed.geometry.buffer(0)  # Buffer with 0 distance to fix self-intersections
        return fixed