#!/usr/bin/env python3
"""
Shapefile Utilities Package

A comprehensive package for working with shapefiles in Python.
This package provides functionality for loading, validating, transforming,
querying, and visualizing spatial data stored in shapefile format.
"""

# Import all modules
from .base import BaseShapefile
from .query import QueryMixin
from .geometry import GeometryMixin
from .analysis import AnalysisMixin
from .plotting import PlottingMixin
from .export import ExportMixin
from .io_helpers import find_shapefiles, get_shapefile_components, unzip_file
from .logger import logger

# Define the composed ShapefileUtils class
class ShapefileUtils(BaseShapefile, QueryMixin, GeometryMixin, 
                     AnalysisMixin, PlottingMixin, ExportMixin):
    """
    A utility class for working with shapefiles of any type.
    
    This class combines all the functionality from the various mixins
    to provide a comprehensive set of tools for working with shapefiles.
    """
    
    def __init__(self, shapefile_path: str):
        """
        Initialize the ShapefileUtils class.
        
        Args:
            shapefile_path (str): Path to the shapefile
        """
        # Initialize the base class
        super().__init__(shapefile_path)
        logger.info(f"Initialized ShapefileUtils for {shapefile_path}")

# Export the public API
__all__ = [
    'ShapefileUtils',
    'BaseShapefile',
    'find_shapefiles',
    'get_shapefile_components',
    'unzip_file',
    'logger'
]