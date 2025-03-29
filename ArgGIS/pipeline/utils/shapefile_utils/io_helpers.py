#!/usr/bin/env python3
"""
IO Helpers Module

Provides utility functions for finding shapefiles, getting shapefile components,
and unzipping files.
"""

import os
import zipfile
from typing import Dict, List, Optional, Union, Any

from .logger import logger

def find_shapefiles(directory: str) -> List[str]:
    """Find all shapefiles in a directory and its subdirectories.
    
    Args:
        directory (str): Directory to search in
        
    Returns:
        list: List of paths to shapefiles
    """
    logger.info(f"Searching for shapefiles in {directory}")
    shapefiles = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.shp'):
                shapefile_path = os.path.join(root, file)
                shapefiles.append(shapefile_path)
                logger.debug(f"Found shapefile: {shapefile_path}")
    
    logger.info(f"Found {len(shapefiles)} shapefiles")
    return shapefiles


def get_shapefile_components(shapefile_path: str) -> Dict[str, str]:
    """Get all component files of a shapefile.
    
    Args:
        shapefile_path (str): Path to the .shp file
        
    Returns:
        dict: Dictionary mapping component types to file paths
    """
    logger.info(f"Getting components for shapefile: {shapefile_path}")
    base_path = os.path.splitext(shapefile_path)[0]
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
            logger.debug(f"Found component: {ext} - {file_path}")
    
    logger.info(f"Found {len(components)} components")
    return components


def unzip_file(zip_path: str, extract_dir: str = None) -> Dict[str, Any]:
    """Unzip a file to a specified directory.
    
    Args:
        zip_path (str): Path to the zip file
        extract_dir (str, optional): Directory to extract files to. If None, extracts to same directory as zip file
        
    Returns:
        dict: Dictionary containing:
            - 'success' (bool): Whether unzip was successful
            - 'files' (list): List of extracted files
            - 'error' (str): Error message if any
    """
    try:
        logger.info(f"Unzipping file: {zip_path}")
        if not os.path.exists(zip_path):
            logger.error(f"Zip file not found: {zip_path}")
            return {'success': False, 'files': [], 'error': f'Zip file not found: {zip_path}'}
            
        if extract_dir is None:
            extract_dir = os.path.dirname(zip_path)
            
        if not os.path.exists(extract_dir):
            logger.info(f"Creating extraction directory: {extract_dir}")
            os.makedirs(extract_dir)
            
        extracted_files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            extracted_files = zip_ref.namelist()
            
        logger.info(f"Successfully extracted {len(extracted_files)} files to {extract_dir}")
        return {
            'success': True,
            'files': extracted_files,
            'error': None
        }
        
    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {zip_path}")
        return {'success': False, 'files': [], 'error': f'Invalid zip file: {zip_path}'}
    except Exception as e:
        logger.error(f"Error unzipping file: {e}")
        return {'success': False, 'files': [], 'error': str(e)}