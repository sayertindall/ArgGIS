#!/usr/bin/env python3
"""
Logger Module for ShapefileUtils

Provides centralized logging configuration for all shapefile utility modules.
"""

import logging

# Configure the basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Create a logger instance for the ShapefileUtils package
logger = logging.getLogger("ShapefileUtils")

# Export the logger for use in other modules
__all__ = ['logger']