#!/usr/bin/env python3
"""
Plotting Module

Provides functionality for visualizing shapefiles using matplotlib and folium.
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
import folium
from folium.plugins import MarkerCluster, HeatMap
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .logger import logger
from .base import BaseShapefile

class PlottingMixin(BaseShapefile):
    """Mixin class for plotting operations on shapefiles.
    
    This class provides methods for creating static and interactive visualizations.
    """
    
    def plot(self,
             column: Optional[str] = None,
             categorical: bool = False,
             legend: bool = True,
             title: Optional[str] = None,
             figsize: Tuple[int, int] = (15, 10),
             cmap: str = 'viridis',
             basemap: bool = False,
             alpha: float = 0.7) -> Tuple[plt.Figure, plt.Axes]:
        """Create a basic plot of the data.
        
        Args:
            column: Column to use for coloring
            categorical: Whether the column contains categorical data
            legend: Whether to show a legend
            title: Plot title
            figsize: Figure size
            cmap: Colormap to use
            basemap: Whether to add a basemap
            alpha: Transparency level
            
        Returns:
            tuple: Figure and axes objects
        """
        if self.gdf is None:
            logger.warning("Cannot plot: GeoDataFrame is None")
            return None
            
        logger.info(f"Creating plot{' with column: ' + column if column else ''}")
        fig, ax = plt.subplots(figsize=figsize)
        
        self.gdf.plot(
            column=column,
            categorical=categorical,
            legend=legend,
            ax=ax,
            cmap=cmap,
            alpha=alpha
        )
        
        if basemap:
            logger.info("Adding basemap")
            # Reproject to Web Mercator for basemap compatibility
            if self.gdf.crs != 'EPSG:3857':
                gdf_web = self.gdf.to_crs('EPSG:3857')
                gdf_web.plot(ax=ax, alpha=0)
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
            else:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        
        if title:
            ax.set_title(title)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        
        return fig, ax
    
    def create_choropleth(self,
                         column: str,
                         title: str = None,
                         cmap: str = 'viridis',
                         figsize: Tuple[int, int] = (15, 10),
                         legend_title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a choropleth map.
        
        Args:
            column: Column to use for coloring
            title: Map title
            cmap: Colormap to use
            figsize: Figure size
            legend_title: Title for the legend
            
        Returns:
            tuple: Figure and axes objects
        """
        if self.gdf is None or column not in self.gdf.columns:
            logger.warning(f"Cannot create choropleth: GeoDataFrame is None or column '{column}' not found")
            return None
            
        logger.info(f"Creating choropleth map with column '{column}'")
        fig, ax = plt.subplots(figsize=figsize)
        
        self.gdf.plot(
            column=column,
            cmap=cmap,
            legend=True,
            legend_kwds={'label': legend_title or column},
            ax=ax
        )
        
        if title:
            ax.set_title(title)
            
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        
        return fig, ax
    
    def create_folium_map(self,
                         popup_columns: List[str] = None,
                         tooltip_column: str = None,
                         style_function: Callable = None,
                         cluster: bool = False,
                         choropleth_column: str = None) -> folium.Map:
        """Create an interactive Folium map.
        
        Args:
            popup_columns: Columns to include in popups
            tooltip_column: Column to use for tooltips
            style_function: Function to style features
            cluster: Whether to cluster point features
            choropleth_column: Column to use for choropleth coloring
            
        Returns:
            folium.Map: Interactive map
        """
        if self.gdf is None:
            logger.warning("Cannot create Folium map: GeoDataFrame is None")
            return None
            
        logger.info("Creating Folium map")
        # Convert to WGS84 for Folium compatibility
        gdf_wgs84 = self.gdf.to_crs('EPSG:4326')
        
        # Calculate center of the data
        center = [
            (gdf_wgs84.total_bounds[1] + gdf_wgs84.total_bounds[3]) / 2,
            (gdf_wgs84.total_bounds[0] + gdf_wgs84.total_bounds[2]) / 2
        ]
        
        # Create map
        m = folium.Map(location=center, zoom_start=10)
        
        # Default style function
        if style_function is None:
            style_function = lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'fillOpacity': 0.2, 'weight': 2}
        
        # Handle point geometries with clustering
        if 'Point' in gdf_wgs84.geometry.type.unique() and cluster:
            logger.info("Clustering point geometries")
            points = gdf_wgs84[gdf_wgs84.geometry.type == 'Point']
            non_points = gdf_wgs84[gdf_wgs84.geometry.type != 'Point']
            
            # Add clustered points
            if not points.empty:
                marker_cluster = MarkerCluster().add_to(m)
                
                for idx, row in points.iterrows():
                    popup_content = ''
                    if popup_columns:
                        popup_content = '<br>'.join([f"{col}: {row[col]}" for col in popup_columns if col in row])
                    
                    tooltip_content = str(row[tooltip_column]) if tooltip_column and tooltip_column in row else None
                    
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=folium.Popup(popup_content, max_width=300) if popup_content else None,
                        tooltip=tooltip_content
                    ).add_to(marker_cluster)
            
            # Add non-point geometries
            if not non_points.empty:
                folium.GeoJson(
                    non_points,
                    style_function=style_function,
                    popup=folium.GeoJsonPopup(fields=popup_columns) if popup_columns else None,
                    tooltip=folium.GeoJsonTooltip(fields=[tooltip_column]) if tooltip_column else None
                ).add_to(m)
        
        # Choropleth map
        elif choropleth_column and choropleth_column in gdf_wgs84.columns:
            logger.info(f"Creating choropleth map with column '{choropleth_column}'")
            folium.Choropleth(
                geo_data=gdf_wgs84,
                name='choropleth',
                data=gdf_wgs84,
                columns=['geometry', choropleth_column],
                key_on='feature.properties.' + choropleth_column,
                fill_color='YlGnBu',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=choropleth_column
            ).add_to(m)
            
            # Add hover functionality
            folium.GeoJson(
                gdf_wgs84,
                style_function=lambda x: {'fillOpacity': 0.0, 'weight': 0},
                tooltip=folium.GeoJsonTooltip(fields=[choropleth_column])
            ).add_to(m)
        
        # Regular GeoJSON
        else:
            logger.info("Adding regular GeoJSON layer")
            folium.GeoJson(
                gdf_wgs84,
                style_function=style_function,
                popup=folium.GeoJsonPopup(fields=popup_columns) if popup_columns else None,
                tooltip=folium.GeoJsonTooltip(fields=[tooltip_column]) if tooltip_column else None
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        logger.info("Folium map created successfully")
        
        return m
        
    def create_heatmap(self, weight_column: Optional[str] = None) -> folium.Map:
        """Create a heatmap from point data.
        
        Args:
            weight_column: Column to use for heatmap weights
            
        Returns:
            folium.Map: Interactive heatmap
        """
        if self.gdf is None:
            logger.warning("Cannot create heatmap: GeoDataFrame is None")
            return None
            
        # Check if geometries are points
        if not any(self.gdf.geometry.type.isin(['Point'])):
            logger.warning("Heatmap requires point geometries")
            return None
            
        logger.info("Creating heatmap")
        # Convert to WGS84 for Folium compatibility
        gdf_wgs84 = self.gdf.to_crs('EPSG:4326')
        points = gdf_wgs84[gdf_wgs84.geometry.type == 'Point']
        
        # Calculate center of the data
        center = [
            (points.total_bounds[1] + points.total_bounds[3]) / 2,
            (points.total_bounds[0] + points.total_bounds[2]) / 2
        ]
        
        # Create map
        m = folium.Map(location=center, zoom_start=10)
        
        # Prepare heatmap data
        heat_data = []
        for idx, row in points.iterrows():
            if weight_column and weight_column in row:
                heat_data.append([row.geometry.y, row.geometry.x, row[weight_column]])
            else:
                heat_data.append([row.geometry.y, row.geometry.x, 1.0])
        
        # Add heatmap layer
        HeatMap(heat_data).add_to(m)
        logger.info("Heatmap created successfully")
        
        return m