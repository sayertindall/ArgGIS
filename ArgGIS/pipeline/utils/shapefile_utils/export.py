#!/usr/bin/env python3
"""
Export Module

Provides functionality for exporting shapefiles to various formats
and performing operations like dissolving and joining.
"""

import geopandas as gpd
import pandas as pd
from typing import Dict, Optional, Union, Any

from .logger import logger
from .base import BaseShapefile

class ExportMixin(BaseShapefile):
    """Mixin class for export operations on shapefiles.
    
    This class provides methods for exporting data to various formats
    and performing operations like dissolving and joining.
    """
    
    def export(self,
               output_path: str,
               driver: str = 'GeoJSON') -> None:
        """Export the data to various formats.
        
        Args:
            output_path: Path for the output file
            driver: Output format driver ('GeoJSON', 'ESRI Shapefile', etc.)
        """
        if self.gdf is None:
            logger.warning("Cannot export: GeoDataFrame is None")
            return None
            
        logger.info(f"Exporting data to {output_path} with driver {driver}")
        self.gdf.to_file(output_path, driver=driver)
        logger.info(f"Data exported to {output_path}")
    
    def dissolve_by_field(self,
                         field: str,
                         aggfunc: Dict[str, str] = None) -> gpd.GeoDataFrame:
        """Dissolve geometries based on a field.
        
        Args:
            field: Field to dissolve by
            aggfunc: Aggregation functions for other fields
            
        Returns:
            GeoDataFrame: Dissolved geometries
        """
        if self.gdf is None:
            logger.warning("Cannot dissolve: GeoDataFrame is None")
            return None
            
        logger.info(f"Dissolving geometries by field '{field}'")
        dissolved = self.gdf.dissolve(by=field, aggfunc=aggfunc)
        logger.info(f"Dissolved {len(self.gdf)} features into {len(dissolved)} features")
        return dissolved
    
    def spatial_join(self,
                    other_gdf: gpd.GeoDataFrame,
                    how: str = 'inner',
                    op: str = 'intersects',
                    lsuffix: str = 'left',
                    rsuffix: str = 'right') -> gpd.GeoDataFrame:
        """Perform a spatial join with another GeoDataFrame.
        
        Args:
            other_gdf: Other GeoDataFrame to join with
            how: Join type ('inner', 'left', or 'right')
            op: Spatial operation ('intersects', 'contains', 'within')
            lsuffix: Suffix for left DataFrame columns
            rsuffix: Suffix for right DataFrame columns
            
        Returns:
            GeoDataFrame: Joined data
        """
        if self.gdf is None or other_gdf is None:
            logger.warning("Cannot perform spatial join: GeoDataFrame is None")
            return None
            
        # Ensure both GeoDataFrames have the same CRS
        if self.gdf.crs != other_gdf.crs:
            logger.info(f"Reprojecting other GeoDataFrame to match CRS: {self.gdf.crs}")
            other_gdf = other_gdf.to_crs(self.gdf.crs)
            
        logger.info(f"Performing spatial join with operation '{op}' and join type '{how}'")
        joined = gpd.sjoin(self.gdf, other_gdf, how=how, op=op, lsuffix=lsuffix, rsuffix=rsuffix)
        logger.info(f"Spatial join resulted in {len(joined)} features")
        return joined
    
    def export_to_geojson(self, output_path: str) -> None:
        """Export the data to GeoJSON format.
        
        Args:
            output_path: Path for the output file
        """
        logger.info(f"Exporting data to GeoJSON: {output_path}")
        self.export(output_path, driver='GeoJSON')
    
    def export_to_shapefile(self, output_path: str) -> None:
        """Export the data to ESRI Shapefile format.
        
        Args:
            output_path: Path for the output file
        """
        logger.info(f"Exporting data to Shapefile: {output_path}")
        self.export(output_path, driver='ESRI Shapefile')
    
    def export_to_csv(self, output_path: str, geometry_format: str = 'WKT') -> None:
        """Export the data to CSV format with geometry as WKT or GeoJSON.
        
        Args:
            output_path: Path for the output file
            geometry_format: Format for geometry column ('WKT' or 'GeoJSON')
        """
        if self.gdf is None:
            logger.warning("Cannot export to CSV: GeoDataFrame is None")
            return None
            
        logger.info(f"Exporting data to CSV: {output_path} with geometry as {geometry_format}")
        df = self.gdf.copy()
        
        if geometry_format == 'WKT':
            df['geometry'] = df.geometry.apply(lambda x: x.wkt)
        elif geometry_format == 'GeoJSON':
            import json
            df['geometry'] = df.geometry.apply(lambda x: json.dumps(x.__geo_interface__))
        else:
            logger.warning(f"Unsupported geometry format: {geometry_format}, using WKT")
            df['geometry'] = df.geometry.apply(lambda x: x.wkt)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data exported to CSV: {output_path}")
    
    def export_metadata(self, output_path: str, format: str = 'json') -> None:
        """Export metadata about the shapefile to a file.
        
        Args:
            output_path: Path for the output file
            format: Output format ('json' or 'md')
        """
        if self.gdf is None:
            logger.warning("Cannot export metadata: GeoDataFrame is None")
            return None
            
        metadata = self.get_metadata()
        column_info = self.get_column_info()
        
        logger.info(f"Exporting metadata to {output_path} in {format} format")
        
        if format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': metadata,
                    'columns': column_info.to_dict('records')
                }, f, indent=2)
        elif format == 'md':
            with open(output_path, 'w') as f:
                f.write(f"# Shapefile Metadata\n\n")
                f.write(f"## General Information\n\n")
                f.write(f"- CRS: {metadata['crs']}\n")
                f.write(f"- Bounds: {metadata['bounds']}\n")
                f.write(f"- Feature Count: {metadata['feature_count']}\n")
                f.write(f"- Geometry Types: {', '.join(metadata['geometry_types'])}\n\n")
                
                f.write(f"## Column Information\n\n")
                f.write(f"| Column | Data Type | Unique Values | Null % |\n")
                f.write(f"| ------ | --------- | ------------- | ------ |\n")
                
                for _, row in column_info.iterrows():
                    f.write(f"| {row['name']} | {row['dtype']} | {row['unique_values']} | {row['null_percentage']:.1f}% |\n")
        else:
            logger.warning(f"Unsupported format: {format}, using json")
            import json
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': metadata,
                    'columns': column_info.to_dict('records')
                }, f, indent=2)
                
        logger.info(f"Metadata exported to {output_path}")