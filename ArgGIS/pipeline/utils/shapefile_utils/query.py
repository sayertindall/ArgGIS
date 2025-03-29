#!/usr/bin/env python3
"""
Query Module

Provides functionality for spatial and attribute queries on shapefiles.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from typing import Dict, List, Optional, Union, Any

from .logger import logger
from .base import BaseShapefile

class QueryMixin(BaseShapefile):
    """Mixin class for query operations on shapefiles.
    
    This class provides methods for spatial and attribute queries.
    """
    
    def spatial_query(self, 
                      geometry: Union[Point, Polygon, LineString],
                      operation: str = 'intersects',
                      buffer_km: float = 0) -> gpd.GeoDataFrame:
        """Perform spatial queries on the data.
        
        Args:
            geometry: Query geometry (Point, Polygon, or LineString)
            operation: Spatial operation ('intersects', 'within', 'contains', 'crosses', 'touches')
            buffer_km: Buffer distance in kilometers
            
        Returns:
            GeoDataFrame: Query results
        """
        if self.gdf is None:
            logger.warning("Cannot perform spatial query: GeoDataFrame is None")
            return None
            
        logger.info(f"Performing spatial query with operation '{operation}'")
        if buffer_km > 0:
            # Convert km to degrees (approximate)
            buffer_deg = buffer_km / 111
            geometry = geometry.buffer(buffer_deg)
            logger.info(f"Applied buffer of {buffer_km} km to query geometry")
        
        if operation == 'intersects':
            mask = self.gdf.geometry.intersects(geometry)
        elif operation == 'within':
            mask = self.gdf.geometry.within(geometry)
        elif operation == 'contains':
            mask = self.gdf.geometry.contains(geometry)
        elif operation == 'crosses':
            mask = self.gdf.geometry.crosses(geometry)
        elif operation == 'touches':
            mask = self.gdf.geometry.touches(geometry)
        else:
            logger.error(f"Unsupported operation: {operation}")
            raise ValueError(f"Unsupported operation: {operation}")
            
        result = self.gdf[mask]
        logger.info(f"Spatial query returned {len(result)} features")
        return result
    
    def attribute_query(self,
                        conditions: Dict[str, Union[str, List[str]]],
                        case_sensitive: bool = False) -> gpd.GeoDataFrame:
        """Query features based on attribute values.
        
        Args:
            conditions: Dictionary of field names and values to match
            case_sensitive: Whether string matching should be case-sensitive
            
        Returns:
            GeoDataFrame: Filtered data
        """
        if self.gdf is None:
            logger.warning("Cannot perform attribute query: GeoDataFrame is None")
            return None
            
        logger.info(f"Performing attribute query with {len(conditions)} conditions")
        mask = pd.Series([True] * len(self.gdf))
        
        for field, value in conditions.items():
            if field not in self.gdf.columns:
                logger.warning(f"Field '{field}' not found in GeoDataFrame")
                continue
                
            if isinstance(value, list):
                if case_sensitive:
                    field_mask = self.gdf[field].isin(value)
                else:
                    field_mask = self.gdf[field].str.lower().isin([v.lower() for v in value])
            else:
                if case_sensitive:
                    field_mask = self.gdf[field] == value
                else:
                    field_mask = self.gdf[field].str.lower() == value.lower()
                    
            mask &= field_mask
            
        result = self.gdf[mask]
        logger.info(f"Attribute query returned {len(result)} features")
        return result
    
    def filter_by_attribute(self, attribute: str, value: Any, operator: str = '==') -> gpd.GeoDataFrame:
        """Filters features by attribute values.
        
        Args:
            attribute: Attribute/column name
            value: Value to compare against
            operator: Comparison operator ('==', '!=', '>', '<', '>=', '<=')
            
        Returns:
            GeoDataFrame: Filtered GeoDataFrame
        """
        if self.gdf is None:
            logger.warning("Cannot filter by attribute: GeoDataFrame is None")
            return None
            
        if attribute not in self.gdf.columns:
            logger.warning(f"Attribute '{attribute}' not found in GeoDataFrame")
            return self.gdf
            
        logger.info(f"Filtering by attribute '{attribute}' with operator '{operator}'")
        
        if operator == '==':
            mask = self.gdf[attribute] == value
        elif operator == '!=':
            mask = self.gdf[attribute] != value
        elif operator == '>':
            mask = self.gdf[attribute] > value
        elif operator == '<':
            mask = self.gdf[attribute] < value
        elif operator == '>=':
            mask = self.gdf[attribute] >= value
        elif operator == '<=':
            mask = self.gdf[attribute] <= value
        else:
            logger.error(f"Unsupported operator: {operator}")
            raise ValueError(f"Unsupported operator: {operator}")
            
        result = self.gdf[mask]
        logger.info(f"Filter returned {len(result)} features")
        return result
    
    def clip_by_boundary(self, boundary: Union[gpd.GeoDataFrame, Polygon]) -> gpd.GeoDataFrame:
        """Clip the data by a boundary.
        
        Args:
            boundary: Boundary to clip by (GeoDataFrame or Polygon)
            
        Returns:
            GeoDataFrame: Clipped data
        """
        if self.gdf is None:
            logger.warning("Cannot clip by boundary: GeoDataFrame is None")
            return None
            
        logger.info("Clipping data by boundary")
        if isinstance(boundary, gpd.GeoDataFrame):
            # Ensure both GeoDataFrames have the same CRS
            if self.gdf.crs != boundary.crs:
                logger.info(f"Reprojecting boundary to match CRS: {self.gdf.crs}")
                boundary = boundary.to_crs(self.gdf.crs)
                
            # Dissolve boundary if it has multiple geometries
            if len(boundary) > 1:
                logger.info("Dissolving boundary with multiple geometries")
                boundary = boundary.dissolve()
                
            boundary_geom = boundary.geometry.iloc[0]
        else:
            boundary_geom = boundary
            
        result = gpd.clip(self.gdf, boundary_geom)
        logger.info(f"Clipping returned {len(result)} features")
        return result