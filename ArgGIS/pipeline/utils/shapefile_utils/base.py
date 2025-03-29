#!/usr/bin/env python3
"""
Base Module

Provides the core functionality for working with shapefiles,
including loading, metadata extraction, and CRS handling.
"""

import os
import geopandas as gpd
import pandas as pd
from pyproj import CRS
from typing import Dict, List, Optional, Union, Any

from .logger import logger

class BaseShapefile:
    """Base class for shapefile operations.
    
    This class provides core functionality for loading shapefiles,
    extracting metadata, and handling coordinate reference systems.
    """
    
    def __init__(self, shapefile_path: str):
        """Initialize the BaseShapefile class.
        
        Args:
            shapefile_path (str): Path to the shapefile
        """
        self.shapefile_path = shapefile_path
        self._gdf = None
        self.original_crs = None
        self.components = self._get_shapefile_components()
        logger.info(f"Initialized BaseShapefile for {shapefile_path}")
    
    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Lazy load the GeoDataFrame."""
        if self._gdf is None:
            self._gdf = self.load_shapefile()
        return self._gdf
    
    def load_shapefile(self) -> gpd.GeoDataFrame:
        """Load and validate the shapefile.
        
        Returns:
            GeoDataFrame: Loaded shapefile data
        """
        try:
            logger.info(f"Loading shapefile from {self.shapefile_path}")
            gdf = gpd.read_file(self.shapefile_path)
            self.original_crs = gdf.crs
            logger.info(f"Loaded shapefile with {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.error(f"Error loading shapefile: {e}")
            return None
    
    def _get_shapefile_components(self) -> Dict[str, str]:
        """Get all component files of a shapefile.
        
        Returns:
            dict: Dictionary mapping component types to file paths
        """
        base_path = os.path.splitext(self.shapefile_path)[0]
        components = {}
        
        extensions = {
            'shp': 'Main geometry file',
            'shx': 'Index file',
            'dbf': 'Attribute data',
            'prj': 'Projection information',
            'cpg': 'Character encoding',
            'sbn': 'Spatial index',
            'sbx': 'Spatial index',
            'xml': 'Metadata',
            'qix': 'Quadtree spatial index'
        }
        
        for ext, desc in extensions.items():
            file_path = f"{base_path}.{ext}"
            if os.path.exists(file_path):
                components[ext] = file_path
        
        logger.debug(f"Found {len(components)} shapefile components")
        return components
    
    def get_metadata(self) -> Dict:
        """Get metadata about the shapefile.
        
        Returns:
            dict: Metadata including CRS, bounds, columns, etc.
        """
        if self.gdf is None:
            logger.warning("Cannot get metadata: GeoDataFrame is None")
            return {}
            
        logger.info("Extracting shapefile metadata")
        return {
            'crs': str(self.gdf.crs),
            'bounds': self.gdf.total_bounds.tolist(),
            'feature_count': len(self.gdf),
            'columns': list(self.gdf.columns),
            'geometry_types': list(self.gdf.geometry.type.unique()),
            'components': self.components
        }
        
    def get_column_info(self) -> pd.DataFrame:
        """Get detailed information about each column in the shapefile.
        
        Returns:
            DataFrame: Column information including data type, unique values, etc.
        """
        if self.gdf is None:
            logger.warning("Cannot get column info: GeoDataFrame is None")
            return None
            
        logger.info("Extracting column information")
        column_info = []
        for col in self.gdf.columns:
            if col != 'geometry':
                dtype = self.gdf[col].dtype
                unique_count = self.gdf[col].nunique()
                non_null_count = self.gdf[col].count()
                null_percentage = (len(self.gdf) - non_null_count) / len(self.gdf) * 100
                
                column_info.append({
                    'name': col,
                    'dtype': str(dtype),
                    'unique_values': unique_count,
                    'null_percentage': null_percentage
                })
        
        return pd.DataFrame(column_info)
    
    def reproject(self, target_crs: Union[str, int]) -> gpd.GeoDataFrame:
        """Reproject the data to a different coordinate system.
        
        Args:
            target_crs: Target coordinate reference system (EPSG code or proj string)
            
        Returns:
            GeoDataFrame: Reprojected data
        """
        if self.gdf is None:
            logger.warning("Cannot reproject: GeoDataFrame is None")
            return None
            
        logger.info(f"Reprojecting from {self.gdf.crs} to {target_crs}")
        return self.gdf.to_crs(target_crs)