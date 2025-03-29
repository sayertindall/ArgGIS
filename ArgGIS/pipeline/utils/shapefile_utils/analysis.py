#!/usr/bin/env python3
"""
Analysis Module

Provides functionality for spatial analysis on shapefiles,
including area calculations, length calculations, grid creation, and statistics.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from rtree import index
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from .logger import logger
from .base import BaseShapefile

class AnalysisMixin(BaseShapefile):
    """Mixin class for analysis operations on shapefiles.
    
    This class provides methods for calculating areas, lengths, creating grids,
    and performing spatial statistics.
    """
    
    def calculate_areas(self, unit: str = 'km2') -> pd.Series:
        """Calculate areas of all geometries.
        
        Args:
            unit: Area unit ('km2', 'ha', or 'm2')
            
        Returns:
            Series: Areas of all geometries
        """
        if self.gdf is None:
            logger.warning("Cannot calculate areas: GeoDataFrame is None")
            return None
            
        logger.info(f"Calculating areas in {unit}")
        # Reproject to equal area projection for accurate calculations
        equal_area = self.reproject('EPSG:6933')  # World Equal Area
        areas = equal_area.geometry.area
        
        # Convert to requested unit
        if unit == 'km2':
            return areas / 1_000_000
        elif unit == 'ha':
            return areas / 10_000
        else:  # m2
            return areas
    
    def calculate_lengths(self, unit: str = 'km') -> pd.Series:
        """Calculate lengths of all line geometries.
        
        Args:
            unit: Length unit ('km', 'm', or 'mi')
            
        Returns:
            Series: Lengths of all geometries
        """
        if self.gdf is None:
            logger.warning("Cannot calculate lengths: GeoDataFrame is None")
            return None
            
        # Check if geometries are lines
        if not any(self.gdf.geometry.type.isin(['LineString', 'MultiLineString'])):
            logger.warning("This method is intended for line geometries")
            
        logger.info(f"Calculating lengths in {unit}")
        # Reproject to equal area projection for accurate calculations
        equal_area = self.reproject('EPSG:6933')  # World Equal Area
        lengths = equal_area.geometry.length
        
        # Convert to requested unit
        if unit == 'km':
            return lengths / 1_000
        elif unit == 'mi':
            return lengths / 1_609.34
        else:  # m
            return lengths
    
    def create_grid(self, cell_size: float, unit: str = 'km') -> gpd.GeoDataFrame:
        """Create a grid covering the extent of the data.
        
        Args:
            cell_size: Size of each grid cell
            unit: Unit of cell size ('km', 'm', or 'mi')
            
        Returns:
            GeoDataFrame: Grid cells
        """
        if self.gdf is None:
            logger.warning("Cannot create grid: GeoDataFrame is None")
            return None
            
        logger.info(f"Creating grid with cell size {cell_size} {unit}")
        # Convert cell size to degrees (approximate)
        if unit == 'km':
            cell_size_deg = cell_size / 111
        elif unit == 'mi':
            cell_size_deg = (cell_size * 1.60934) / 111
        else:  # m
            cell_size_deg = (cell_size / 1000) / 111
            
        # Get bounds
        minx, miny, maxx, maxy = self.gdf.total_bounds
        
        # Create grid cells
        grid_cells = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                grid_cells.append(box(x, y, x + cell_size_deg, y + cell_size_deg))
                y += cell_size_deg
            x += cell_size_deg
            
        # Create GeoDataFrame
        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=self.gdf.crs)
        logger.info(f"Created grid with {len(grid)} cells")
        
        return grid
    
    def aggregate_by_field(self, field: str, agg_functions: Dict[str, str]) -> pd.DataFrame:
        """Aggregate data by a field.
        
        Args:
            field: Field to group by
            agg_functions: Dictionary of field names and aggregation functions
            
        Returns:
            DataFrame: Aggregated data
        """
        if self.gdf is None:
            logger.warning("Cannot aggregate: GeoDataFrame is None")
            return None
            
        logger.info(f"Aggregating data by field '{field}'")
        return self.gdf.groupby(field).agg(agg_functions)
    
    def get_summary_statistics(self, numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate summary statistics for numeric columns.
        
        Args:
            numeric_columns: List of numeric columns to analyze
            
        Returns:
            DataFrame: Summary statistics
        """
        if self.gdf is None:
            logger.warning("Cannot get summary statistics: GeoDataFrame is None")
            return None
            
        if numeric_columns is None:
            numeric_columns = self.gdf.select_dtypes(include=[np.number]).columns
            
        logger.info(f"Calculating summary statistics for {len(numeric_columns)} columns")
        return self.gdf[numeric_columns].describe()
    
    def nearest_neighbor_analysis(self, other_gdf: gpd.GeoDataFrame, 
                                max_distance: Optional[float] = None) -> pd.DataFrame:
        """Find nearest neighbors between this GeoDataFrame and another.
        
        Args:
            other_gdf: Other GeoDataFrame to find neighbors in
            max_distance: Maximum distance to consider (in CRS units)
            
        Returns:
            DataFrame: Nearest neighbor results
        """
        if self.gdf is None or other_gdf is None:
            logger.warning("Cannot perform nearest neighbor analysis: GeoDataFrame is None")
            return None
            
        # Ensure both GeoDataFrames have the same CRS
        if self.gdf.crs != other_gdf.crs:
            logger.info(f"Reprojecting other GeoDataFrame to match CRS: {self.gdf.crs}")
            other_gdf = other_gdf.to_crs(self.gdf.crs)
            
        logger.info("Creating spatial index for nearest neighbor analysis")
        # Create spatial index for other_gdf
        spatial_index = index.Index()
        for idx, geom in enumerate(other_gdf.geometry):
            spatial_index.insert(idx, geom.bounds)
            
        # Find nearest neighbors
        neighbors = []
        logger.info(f"Finding nearest neighbors for {len(self.gdf)} features")
        for idx1, row1 in self.gdf.iterrows():
            geom1 = row1.geometry
            nearest_idx = None
            nearest_dist = float('inf')
            
            # Use spatial index to find potential neighbors
            for idx2 in spatial_index.intersection(geom1.bounds):
                geom2 = other_gdf.iloc[idx2].geometry
                dist = geom1.distance(geom2)
                
                if dist < nearest_dist and (max_distance is None or dist <= max_distance):
                    nearest_dist = dist
                    nearest_idx = other_gdf.index[idx2]
            
            if nearest_idx is not None:
                neighbors.append({
                    'source_idx': idx1,
                    'target_idx': nearest_idx,
                    'distance': nearest_dist
                })
        
        logger.info(f"Found {len(neighbors)} nearest neighbors")
        return pd.DataFrame(neighbors)