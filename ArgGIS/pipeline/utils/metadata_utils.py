#!/usr/bin/env python3
"""
Metadata Utilities Module

Utilities for generating metadata summaries for datasets.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Union, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MetadataUtils")

def summarize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary of columns in a DataFrame.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        DataFrame: Summary of columns including data type, unique values, missing values, etc.
    """
    if df is None:
        logger.warning("Cannot summarize columns: DataFrame is None")
        return None
        
    logger.info("Generating column summary")
    summary = []
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        non_null_count = df[col].count()
        null_percentage = (len(df) - non_null_count) / len(df) * 100
        
        # Get sample values
        if non_null_count > 0:
            sample_values = df[col].dropna().sample(min(3, non_null_count)).tolist()
        else:
            sample_values = []
        
        summary.append({
            'column': col,
            'dtype': str(dtype),
            'unique_values': unique_count,
            'null_percentage': null_percentage,
            'sample_values': str(sample_values)
        })
    
    return pd.DataFrame(summary)

def export_metadata_dict(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Export metadata about a GeoDataFrame to a JSON file.
    
    Args:
        gdf: GeoDataFrame to generate metadata for
        out_path: Path for the output file
    """
    if gdf is None:
        logger.warning("Cannot export metadata: GeoDataFrame is None")
        return None
        
    logger.info(f"Exporting metadata to {out_path}")
    
    # Generate metadata
    metadata = {
        'general': {
            'feature_count': len(gdf),
            'column_count': len(gdf.columns),
            'crs': str(gdf.crs),
            'geometry_types': list(gdf.geometry.type.unique()),
            'bounds': gdf.total_bounds.tolist(),
            'export_date': datetime.now().isoformat()
        },
        'columns': []
    }
    
    # Add column information
    column_info = summarize_columns(gdf)
    metadata['columns'] = column_info.to_dict('records')
    
    # Write to file
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Metadata exported to {out_path}")

def export_metadata_markdown(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Export metadata about a GeoDataFrame to a Markdown file.
    
    Args:
        gdf: GeoDataFrame to generate metadata for
        out_path: Path for the output file
    """
    if gdf is None:
        logger.warning("Cannot export metadata: GeoDataFrame is None")
        return None
        
    logger.info(f"Exporting metadata to {out_path}")
    
    # Generate column information
    column_info = summarize_columns(gdf)
    
    # Write to file
    with open(out_path, 'w') as f:
        f.write(f"# Dataset Metadata\n\n")
        f.write(f"## General Information\n\n")
        f.write(f"- **Feature Count:** {len(gdf)}\n")
        f.write(f"- **Column Count:** {len(gdf.columns)}\n")
        f.write(f"- **CRS:** {gdf.crs}\n")
        f.write(f"- **Geometry Types:** {', '.join(gdf.geometry.type.unique())}\n")
        f.write(f"- **Bounds:** {gdf.total_bounds.tolist()}\n")
        f.write(f"- **Export Date:** {datetime.now().isoformat()}\n\n")
        
        f.write(f"## Column Information\n\n")
        f.write(f"| Column | Data Type | Unique Values | Null % | Sample Values |\n")
        f.write(f"| ------ | --------- | ------------- | ------ | ------------- |\n")
        
        for _, row in column_info.iterrows():
            f.write(f"| {row['column']} | {row['dtype']} | {row['unique_values']} | {row['null_percentage']:.1f}% | {row['sample_values']} |\n")
        
    logger.info(f"Metadata exported to {out_path}")

def generate_data_manifest(directory: str, out_path: str, include_patterns: List[str] = None) -> None:
    """Generate a manifest of data files in a directory.
    
    Args:
        directory: Directory to scan for data files
        out_path: Path for the output file
        include_patterns: List of file patterns to include (e.g., ['.shp', '.csv'])
    """
    logger.info(f"Generating data manifest for {directory}")
    
    if include_patterns is None:
        include_patterns = ['.shp', '.csv', '.xlsx', '.geojson', '.json', '.gpkg']
    
    manifest = {
        'directory': directory,
        'scan_date': datetime.now().isoformat(),
        'files': []
    }
    
    # Scan directory
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(pattern) for pattern in include_patterns):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                file_size = os.path.getsize(file_path)
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                
                manifest['files'].append({
                    'name': file,
                    'path': rel_path,
                    'size_bytes': file_size,
                    'modified': file_modified,
                    'type': os.path.splitext(file)[1][1:]
                })
    
    # Write to file
    with open(out_path, 'w') as f:
        if out_path.endswith('.json'):
            json.dump(manifest, f, indent=2)
        elif out_path.endswith('.md'):
            f.write(f"# Data Manifest\n\n")
            f.write(f"**Directory:** {directory}\n")
            f.write(f"**Scan Date:** {manifest['scan_date']}\n\n")
            
            f.write(f"## Files\n\n")
            f.write(f"| Name | Path | Size | Modified | Type |\n")
            f.write(f"| ---- | ---- | ---- | -------- | ---- |\n")
            
            for file_info in manifest['files']:
                size_kb = file_info['size_bytes'] / 1024
                size_str = f"{size_kb:.1f} KB"
                f.write(f"| {file_info['name']} | {file_info['path']} | {size_str} | {file_info['modified']} | {file_info['type']} |\n")
        
    logger.info(f"Data manifest exported to {out_path} with {len(manifest['files'])} files")