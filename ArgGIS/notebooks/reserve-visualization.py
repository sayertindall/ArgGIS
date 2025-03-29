import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import geopandas as gpd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ReserveViz")

class ReserveDataVisualizer:
    """
    A comprehensive tool for visualizing and tracking oil and gas reserve data
    across multiple years and locations.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 map_file: Optional[Union[str, Path]] = None):
        """
        Initialize the visualizer with path configurations.
        
        Args:
            data_dir: Directory containing processed CSV files
            output_dir: Directory for output visualizations
            map_file: Optional path to GeoJSON file for mapping (Argentina provinces)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.map_file = Path(map_file) if map_file else None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default parameters
        self.figsize = (12, 8)
        self.dpi = 100
        self.color_palette = sns.color_palette("viridis", 10)
        
        # Set figure style
        sns.set_style("whitegrid")
        
        # Load and cache data
        self.data_cache = {}
    
    def load_data(self, refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all processed reserve CSV files into a dictionary.
        
        Args:
            refresh: If True, reload all data even if cached
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames by filename
        """
        if self.data_cache and not refresh:
            return self.data_cache
        
        data_dict = {}
        
        for file_path in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                
                # Ensure 'value' column is numeric
                if 'value' in df.columns:
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                # Ensure 'year' column is present
                if 'year' not in df.columns:
                    # Try to extract from filename
                    year_match = None
                    for part in file_path.stem.split('_'):
                        if part.isdigit() and len(part) == 4:
                            year_match = int(part)
                            break
                    
                    if year_match:
                        df['year'] = year_match
                
                data_dict[file_path.stem] = df
                logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        self.data_cache = data_dict
        return data_dict
    
    def combine_datasets(self, scope: Optional[str] = None) -> pd.DataFrame:
        """
        Combine all datasets into a single DataFrame for analysis.
        
        Args:
            scope: Optional filter for scope ('end_of_concession' or 'end_of_life')
            
        Returns:
            pd.DataFrame: Combined DataFrame
        """
        data_dict = self.load_data()
        
        if not data_dict:
            logger.warning("No data available to combine")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(data_dict.values(), ignore_index=True)
        
        # Filter by scope if specified
        if scope and 'scope' in combined_df.columns:
            combined_df = combined_df[combined_df['scope'] == scope]
        
        # Ensure 'year' is integer
        if 'year' in combined_df.columns:
            combined_df['year'] = pd.to_numeric(combined_df['year'], errors='coerce').astype('Int64')
        
        return combined_df
    
    def plot_reserves_by_year(self, 
                             metric: str = 'oil_m3', 
                             scope: str = 'end_of_concession',
                             reserve_type: Optional[str] = None,
                             cumulative: bool = False,
                             normalize: bool = False,
                             show_trend: bool = True) -> plt.Figure:
        """
        Create a bar chart of reserves by year for a specific metric.
        
        Args:
            metric: Metric to plot (e.g., 'oil_m3', 'gas_mm3')
            scope: Scope to filter by ('end_of_concession' or 'end_of_life')
            reserve_type: Optional filter for reserve type (e.g., 'comprobadas', 'probables')
            cumulative: If True, show cumulative values
            normalize: If True, normalize values (percentage of maximum)
            show_trend: If True, add trend line
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        combined_df = self.combine_datasets(scope)
        
        if combined_df.empty:
            logger.warning("No data available for plotting")
            return None
        
        # Filter data
        mask = combined_df['standardized_metric'] == metric
        if reserve_type:
            # Check which column might contain reserve type
            for col in ['comprobadas', 'probables', 'posibles']:
                if col in combined_df.columns:
                    mask &= (combined_df[col] == reserve_type) | (combined_df[col].str.contains(reserve_type, na=False))
        
        plot_df = combined_df[mask].copy()
        
        if plot_df.empty:
            logger.warning(f"No data for metric {metric} and reserve type {reserve_type}")
            return None
        
        # Group by year and calculate sum of values
        yearly_data = plot_df.groupby('year')['value'].sum().reset_index()
        yearly_data = yearly_data.sort_values('year')
        
        # Calculate cumulative values if requested
        if cumulative:
            yearly_data['value'] = yearly_data['value'].cumsum()
        
        # Normalize if requested
        if normalize:
            max_value = yearly_data['value'].max()
            if max_value > 0:
                yearly_data['value'] = yearly_data['value'] / max_value * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot bar chart
        bars = ax.bar(yearly_data['year'], yearly_data['value'], color=self.color_palette[0], alpha=0.7)
        
        # Add trend line if requested
        if show_trend and len(yearly_data) > 2:
            x = yearly_data['year']
            y = yearly_data['value']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", linewidth=2)
        
        # Add labels and title
        metric_name = metric.replace('_', ' ').title()
        scope_name = scope.replace('_', ' ').title()
        reserve_name = reserve_type.title() if reserve_type else "All Types"
        
        title = f"{metric_name} Reserves by Year ({scope_name})"
        if reserve_type:
            title += f" - {reserve_name}"
        if cumulative:
            title += " (Cumulative)"
        if normalize:
            title += " (Normalized)"
        
        ax.set_title(title)
        ax.set_xlabel("Year")
        
        # Set y-axis label based on metric and normalization
        if normalize:
            ax.set_ylabel("Percentage of Maximum (%)")
        else:
            unit = metric.split('_')[-1] if '_' in metric else ''
            if unit == 'm3':
                ax.set_ylabel("Volume (cubic meters)")
            elif unit == 'mm3':
                ax.set_ylabel("Volume (million cubic meters)")
            elif unit == 'bbl':
                ax.set_ylabel("Volume (barrels)")
            elif unit == 'bcf':
                ax.set_ylabel("Volume (billion cubic feet)")
            elif unit == 'boe':
                ax.set_ylabel("Barrel of Oil Equivalent (BOE)")
            else:
                ax.set_ylabel("Value")
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}', ha='center', va='bottom', rotation=0)
        
        # Adjust aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save the figure
        output_filename = f"reserves_{metric}_by_year_{scope}"
        if reserve_type:
            output_filename += f"_{reserve_type.lower()}"
        if cumulative:
            output_filename += "_cumulative"
        if normalize:
            output_filename += "_normalized"
        
        output_path = self.output_dir / f"{output_filename}.png"
        fig.tight_layout()
        fig.savefig(output_path)
        
        logger.info(f"Saved plot to {output_path}")
        
        return fig
    
    def plot_reserve_composition(self,
                                year: int,
                                scope: str = 'end_of_concession',
                                use_boe: bool = True,
                                top_n: int = 10) -> plt.Figure:
        """
        Create a pie chart showing the composition of reserves for a specific year.
        
        Args:
            year: Year to analyze
            scope: Scope to filter by ('end_of_concession' or 'end_of_life')
            use_boe: If True, use barrel of oil equivalent for comparison
            top_n: Number of top entities to show individually (others grouped)
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        combined_df = self.combine_datasets(scope)
        
        if combined_df.empty:
            logger.warning("No data available for plotting")
            return None
        
        # Filter by year
        plot_df = combined_df[combined_df['year'] == year].copy()
        
        if plot_df.empty:
            logger.warning(f"No data for year {year}")
            return None
        
        # Use BOE or original values
        value_col = 'value_boe' if use_boe and 'value_boe' in plot_df.columns else 'value'
        
        # Determine entity column (basin, province, operator, etc.)
        entity_cols = ['cuenca', 'provincia', 'operador']
        entity_col = None
        
        for col in entity_cols:
            if col in plot_df.columns and plot_df[col].notna().sum() > 0:
                entity_col = col
                break
        
        if not entity_col:
            logger.warning("No suitable entity column found for composition plot")
            return None
        
        # Group by entity and calculate sum
        composition = plot_df.groupby(entity_col)[value_col].sum().reset_index()
        composition = composition.sort_values(value_col, ascending=False)
        
        # Limit to top N and group others
        if len(composition) > top_n:
            top_entities = composition.head(top_n)
            others_sum = composition.iloc[top_n:][value_col].sum()
            
            # Create a DataFrame for "Others"
            others_df = pd.DataFrame({
                entity_col: ['Others'],
                value_col: [others_sum]
            })
            
            composition = pd.concat([top_entities, others_df], ignore_index=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            composition[value_col],
            labels=composition[entity_col],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("viridis", len(composition)),
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Styling
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
        
        # Add title
        entity_name = entity_col.replace('_', ' ').title()
        scope_name = scope.replace('_', ' ').title()
        value_type = "Barrel of Oil Equivalent" if use_boe else "Original Units"
        
        title = f"Reserve Composition by {entity_name} in {year} ({scope_name})\n{value_type}"
        ax.set_title(title)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect('equal')
        
        # Save the figure
        output_filename = f"reserve_composition_{entity_col}_{year}_{scope}"
        if use_boe:
            output_filename += "_boe"
        
        output_path = self.output_dir / f"{output_filename}.png"
        fig.tight_layout()
        fig.savefig(output_path)
        
        logger.info(f"Saved composition plot to {output_path}")
        
        return fig
    
    def plot_reserves_by_basin(self,
                              year: int,
                              metric: str = 'oil_m3',
                              scope: str = 'end_of_concession',
                              normalize: bool = False,
                              horizontal: bool = True) -> plt.Figure:
        """
        Create a bar chart of reserves by basin for a specific year and metric.
        
        Args:
            year: Year to analyze
            metric: Metric to plot (e.g., 'oil_m3', 'gas_mm3')
            scope: Scope to filter by ('end_of_concession' or 'end_of_life')
            normalize: If True, show normalized values (percentage of total)
            horizontal: If True, create horizontal bar chart
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        combined_df = self.combine_datasets(scope)
        
        if combined_df.empty:
            logger.warning("No data available for plotting")
            return None
        
        # Filter data
        mask = (combined_df['year'] == year) & (combined_df['standardized_metric'] == metric)
        plot_df = combined_df[mask].copy()
        
        if plot_df.empty:
            logger.warning(f"No data for year {year} and metric {metric}")
            return None
        
        # Determine basin column
        basin_cols = ['cuenca', 'basin']
        basin_col = None
        
        for col in basin_cols:
            if col in plot_df.columns and plot_df[col].notna().sum() > 0:
                basin_col = col
                break
        
        if not basin_col:
            logger.warning("No basin column found for plotting")
            return None
        
        # Group by basin and calculate sum
        basin_data = plot_df.groupby(basin_col)['value'].sum().reset_index()
        basin_data = basin_data.sort_values('value', ascending=True if horizontal else False)
        
        # Normalize if requested
        if normalize and basin_data['value'].sum() > 0:
            basin_data['value'] = basin_data['value'] / basin_data['value'].sum() * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot bars
        if horizontal:
            bars = ax.barh(basin_data[basin_col], basin_data['value'], color=self.color_palette[1], alpha=0.8)
        else:
            bars = ax.bar(basin_data[basin_col], basin_data['value'], color=self.color_palette[1], alpha=0.8)
        
        # Add labels
        basin_col_name = basin_col.replace('_', ' ').title()
        metric_name = metric.replace('_', ' ').title()
        scope_name = scope.replace('_', ' ').title()
        
        title = f"{metric_name} by {basin_col_name} in {year} ({scope_name})"
        if normalize:
            title += " (Normalized)"
        
        ax.set_title(title)
        
        if horizontal:
            ax.set_xlabel("Value" if not normalize else "Percentage (%)")
            ax.set_ylabel(basin_col_name)
        else:
            ax.set_xlabel(basin_col_name)
            ax.set_ylabel("Value" if not normalize else "Percentage (%)")
        
        # Format axis with commas
        if horizontal:
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        else:
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add data labels
        for bar in bars:
            if horizontal:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2., 
                       f'{width:,.1f}{"%" if normalize else ""}',
                       ha='left', va='center', fontsize=9)
            else:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.1f}{"%" if normalize else ""}',
                       ha='center', va='bottom', fontsize=9)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save the figure
        orientation = "horizontal" if horizontal else "vertical"
        output_filename = f"reserves_by_{basin_col}_{year}_{metric}_{scope}_{orientation}"
        if normalize:
            output_filename += "_normalized"
        
        output_path = self.output_dir / f"{output_filename}.png"
        fig.tight_layout()
        fig.savefig(output_path)
        
        logger.info(f"Saved basin plot to {output_path}")
        
        return fig
    
    def plot_reserves_time_series(self,
                                metric: str = 'oil_m3',
                                scope: str = 'end_of_concession',
                                by_entity: Optional[str] = None,
                                top_n: int = 5,
                                add_total: bool = True,
                                normalize: bool = False) -> plt.Figure:
        """
        Create a time series plot of reserves for a specific metric.
        
        Args:
            metric: Metric to plot (e.g., 'oil_m3', 'gas_mm3')
            scope: Scope to filter by ('end_of_concession' or 'end_of_life')
            by_entity: Optional grouping (e.g., 'cuenca', 'operador')
            top_n: Number of top entities to show
            add_total: If True, add a line for total reserves
            normalize: If True, normalize values (percentage of maximum)
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        combined_df = self.combine_datasets(scope)
        
        if combined_df.empty:
            logger.warning("No data available for plotting")
            return None
        
        # Filter data
        mask = combined_df['standardized_metric'] == metric
        plot_df = combined_df[mask].copy()
        
        if plot_df.empty:
            logger.warning(f"No data for metric {metric}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # If no grouping, just plot total by year
        if not by_entity:
            yearly_data = plot_df.groupby('year')['value'].sum().reset_index()
            yearly_data = yearly_data.sort_values('year')
            
            # Normalize if requested
            if normalize and yearly_data['value'].max() > 0:
                yearly_data['value'] = yearly_data['value'] / yearly_data['value'].max() * 100
            
            # Plot line
            ax.plot(yearly_data['year'], yearly_data['value'], marker='o', 
                   linewidth=2, markersize=8, label="Total")
        else:
            # Check if entity column exists
            if by_entity not in plot_df.columns:
                logger.warning(f"Entity column {by_entity} not found")
                return None
            
            # Group by entity and year
            entity_yearly = plot_df.groupby([by_entity, 'year'])['value'].sum().reset_index()
            
            # Get top N entities
            top_entities = plot_df.groupby(by_entity)['value'].sum().nlargest(top_n).index.tolist()
            
            # Filter to top entities
            entity_yearly = entity_yearly[entity_yearly[by_entity].isin(top_entities)]
            
            # Calculate total by year
            total_yearly = plot_df.groupby('year')['value'].sum().reset_index()
            
            # Normalize if requested
            if normalize:
                max_value = total_yearly['value'].max()
                if max_value > 0:
                    entity_yearly['value'] = entity_yearly['value'] / max_value * 100
                    total_yearly['value'] = total_yearly['value'] / max_value * 100
            
            # Plot each entity
            for i, entity in enumerate(top_entities):
                entity_data = entity_yearly[entity_yearly[by_entity] == entity]
                entity_data = entity_data.sort_values('year')
                
                ax.plot(entity_data['year'], entity_data['value'], marker='o', 
                       linewidth=2, markersize=6, label=entity, 
                       color=self.color_palette[i % len(self.color_palette)])
            
            # Add total line if requested
            if add_total:
                total_yearly = total_yearly.sort_values('year')
                ax.plot(total_yearly['year'], total_yearly['value'], marker='s', 
                       linewidth=3, markersize=8, label="Total", 
                       color='black', linestyle='--')
        
        # Add labels and title
        metric_name = metric.replace('_', ' ').title()
        scope_name = scope.replace('_', ' ').title()
        entity_name = by_entity.replace('_', ' ').title() if by_entity else ""
        
        title = f"{metric_name} Reserves Over Time ({scope_name})"
        if by_entity:
            title += f" by {entity_name}"
        if normalize:
            title += " (Normalized)"
        
        ax.set_title(title)
        ax.set_xlabel("Year")
        
        if normalize:
            ax.set_ylabel("Percentage of Maximum (%)")
        else:
            unit = metric.split('_')[-1] if '_' in metric else ''
            if unit == 'm3':
                ax.set_ylabel("Volume (cubic meters)")
            elif unit == 'mm3':
                ax.set_ylabel("Volume (million cubic meters)")
            elif unit == 'bbl':
                ax.set_ylabel("Volume (barrels)")
            elif unit == 'bcf':
                ax.set_ylabel("Volume (billion cubic feet)")
            elif unit == 'boe':
                ax.set_ylabel("Barrel of Oil Equivalent (BOE)")
            else:
                ax.set_ylabel("Value")
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add legend
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save the figure
        output_filename = f"reserves_timeseries_{metric}_{scope}"
        if by_entity:
            output_filename += f"_by_{by_entity}"
        if normalize:
            output_filename += "_normalized"
        
        output_path = self.output_dir / f"{output_filename}.png"
        fig.tight_layout()
        fig.savefig(output_path)
        
        logger.info(f"Saved time series plot to {output_path}")
        
        return fig
    
    def plot_geographic_distribution(self,
                                   year: int,
                                   metric: str = 'oil_m3',
                                   scope: str = 'end_of_concession',
                                   normalize: bool = True) -> plt.Figure:
        """
        Create a map plot showing geographic distribution of reserves.
        Requires a GeoJSON file with province boundaries.
        
        Args:
            year: Year to analyze
            metric: Metric to plot (e.g., 'oil_m3', 'gas_mm3')
            scope: Scope to filter by ('end_of_concession' or 'end_of_life')
            normalize: If True, normalize values (percentage of maximum)
            
        Returns:
            plt.Figure: Matplotlib figure or None if map_file not available
        """
        if not self.map_file or not self.map_file.exists():
            logger.warning("No map file provided for geographic plotting")
            return None
        
        combined_df = self.combine_datasets(scope)
        
        if combined_df.empty:
            logger.warning("No data available for plotting")
            return None
        
        # Filter data
        mask = (combined_df['year'] == year) & (combined_df['standardized_metric'] == metric)
        plot_df = combined_df[mask].copy()
        
        if plot_df.empty:
            logger.warning(f"No data for year {year} and metric {metric}")
            return None
        
        # Determine province column
        province_cols = ['provincia', 'province']
        province_col = None
        
        for col in province_cols:
            if col in plot_df.columns and plot_df[col].notna().sum() > 0:
                province_col = col
                break
        
        if not province_col:
            logger.warning("No province column found for mapping")
            return None
        
        # Group by province and calculate sum
        province_data = plot_df.groupby(province_col)['value'].sum().reset_index()
        
        # Normalize if requested
        if normalize and province_data['value'].max() > 0:
            province_data['value'] = province_data['value'] / province_data['value'].max() * 100
        
        try:
            # Load geographic data
            gdf = gpd.read_file(self.map_file)
            
            # Ensure province names match between datasets
            # This may require additional cleaning/mapping depending on your data
            province_data[province_col] = province_data[province_col].str.strip().str.upper()
            
            # Merge with geographic data
            # Assuming gdf has a 'NAME' column with province names
            # Adjust this to match your GeoJSON structure
            province_col_geo = 'NAME'  # Column name in GeoJSON
            
            # Create a mapping dictionary if names don't match exactly
            # Adjust this based on your specific data
            province_mapping = {
                'BUENOS AIRES': 'BUENOS AIRES',
                'CAPITAL FEDERAL': 'CIUDAD DE BUENOS AIRES',
                'CATAMARCA': 'CATAMARCA',
                'CHACO': 'CHACO',
                'CHUBUT': 'CHUBUT',
                'CORDOBA': 'CÓRDOBA',
                'CORRIENTES': 'CORRIENTES',
                'ENTRE RIOS': 'ENTRE RÍOS',
                'FORMOSA': 'FORMOSA',
                'JUJUY': 'JUJUY',
                'LA PAMPA': 'LA PAMPA',
                'LA RIOJA': 'LA RIOJA',
                'MENDOZA': 'MENDOZA',
                'MISIONES': 'MISIONES',
                'NEUQUEN': 'NEUQUÉN',
                'RIO NEGRO': 'RÍO NEGRO',
                'SALTA': 'SALTA',
                'SAN JUAN': 'SAN JUAN',
                'SAN LUIS': 'SAN LUIS',
                'SANTA CRUZ': 'SANTA CRUZ',
                'SANTA FE': 'SANTA FE',
                'SANTIAGO DEL ESTERO': 'SANTIAGO DEL ESTERO',
                'TIERRA DEL FUEGO': 'TIERRA DEL FUEGO',
                'TUCUMAN': 'TUCUMÁN'
            }
            
            # Create a new column with mapped province names
            province_data['province_mapped'] = province_data[province_col].map(
                lambda x: next((v for k, v in province_mapping.items() if k in x or x in k), x)
            )
            
            # Merge data
            merged_gdf = gdf.merge(province_data, how='left', 
                                  left_on=province_col_geo, right_on='province_mapped')
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot map
            merged_gdf.plot(column='value', cmap='viridis', linewidth=0.8, 
                          ax=ax, edgecolor='0.8', legend=True)
            
            # Add labels
            metric_name = metric.replace('_', ' ').title()
            scope_name = scope.replace('_', ' ').title()
            
            title = f"Geographic Distribution of {metric_name} in {year} ({scope_name})"
            if normalize:
                title += " (Normalized)"
            
            ax.set_title(title)
            
            # Remove axis
            ax.set_axis_off()
            
            # Add annotations for top provinces
            if not province_data.empty:
                top_provinces = province_data.nlargest(5, 'value')
                
                for _, row in top_provinces.iterrows():
                    province = row['province_mapped']
                    value = row['value']
                    
                    # Find centroid of province
                    province_poly = merged_gdf[merged_gdf[province_col_geo] == province]
                    if not province_poly.empty:
                        centroid = province_poly.geometry.iloc[0].centroid
                        ax.annotate(
                            f"{province}\n{value:.1f}{'%' if normalize else ''}",
                            xy=(centroid.x, centroid.y),
                            horizontalalignment='center',
                            fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                        )
            
            # Save the figure
            output_filename = f"reserves_map_{year}_{metric}_{scope}"
            if normalize:
                output_filename += "_normalized"
            
            output_path = self.output_dir / f"{output_filename}.png"
            fig.tight_layout()
            fig.savefig(output_path)
            
            logger.info(f"Saved map plot to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating geographic plot: {e}")
            return None
    
    def generate_reserve_report(self, year: Optional[int] = None, output_format: str = 'markdown') -> str:
        """
        Generate a comprehensive report on reserve data for a specific year or all years.
        
        Args:
            year: Year to analyze (None for all years)
            output_format: Format of the report ('markdown' or 'html')
            
        Returns:
            str: Report content in specified format
        """
        combined_df = self.combine_datasets()
        
        if combined_df.empty:
            return "No data available for reporting"
        
        # Filter by year if specified
        if year is not None:
            report_df = combined_df[combined_df['year'] == year].copy()
            if report_df.empty:
                return f"No data available for year {year}"
        else:
            report_df = combined_df.copy()
        
        # Get available years
        years = sorted(report_df['year'].unique().tolist())
        
        # Get available metrics
        metrics = report_df['standardized_metric'].dropna().unique().tolist()
        
        # Start building the report
        if output_format == 'markdown':
            report = f"# Reserves Data Report\n\n"
            if year:
                report += f"## Year: {year}\n\n"
            else:
                report += f"## Years: {', '.join(map(str, years))}\n\n"
            
            # Summary statistics
            report += "## Summary Statistics\n\n"
            
            # Table of metrics
            report += "### Available Metrics\n\n"
            report += "| Metric | Total Value | Unit |\n"
            report += "|--------|-------------|------|\n"
            
            for metric in metrics:
                metric_total = report_df[report_df['standardized_metric'] == metric]['value'].sum()
                unit = metric.split('_')[-1] if '_' in metric else ''
                report += f"| {metric.replace('_', ' ').title()} | {metric_total:,.2f} | {unit} |\n"
            
            report += "\n"
            
            # Temporal analysis
            report += "## Temporal Analysis\n\n"
            
            if len(years) > 1:
                report += "### Year-over-Year Comparison\n\n"
                yearly_totals = report_df.groupby(['year', 'standardized_metric'])['value'].sum().reset_index()
                
                for metric in metrics:
                    metric_data = yearly_totals[yearly_totals['standardized_metric'] == metric]
                    report += f"#### {metric.replace('_', ' ').title()}\n\n"
                    report += "| Year | Value | Change (%) |\n"
                    report += "|------|-------|------------|\n"
                    
                    prev_value = None
                    for _, row in metric_data.sort_values('year').iterrows():
                        year_val = row['year']
                        value = row['value']
                        
                        if prev_value is not None and prev_value != 0:
                            change_pct = (value - prev_value) / prev_value * 100
                            report += f"| {year_val} | {value:,.2f} | {change_pct:+.2f}% |\n"
                        else:
                            report += f"| {year_val} | {value:,.2f} | - |\n"
                        
                        prev_value = value
                    
                    report += "\n"
            
            # Geographic analysis
            province_cols = ['provincia', 'province']
            province_col = None
            
            for col in province_cols:
                if col in report_df.columns and report_df[col].notna().sum() > 0:
                    province_col = col
                    break
            
            if province_col:
                report += "## Geographic Analysis\n\n"
                report += f"### Top Provinces by {metrics[0].replace('_', ' ').title() if metrics else 'Reserves'}\n\n"
                
                if metrics:
                    province_data = report_df[report_df['standardized_metric'] == metrics[0]].groupby(province_col)['value'].sum().reset_index()
                    province_data = province_data.sort_values('value', ascending=False).head(10)
                    
                    report += f"| {province_col.title()} | Value | Percentage |\n"
                    report += "|" + "-" * (len(province_col)+2) + "|---------|------------|\n"
                    
                    total = province_data['value'].sum()
                    
                    for _, row in province_data.iterrows():
                        province = row[province_col]
                        value = row['value']
                        percentage = (value / total * 100) if total > 0 else 0
                        
                        report += f"| {province} | {value:,.2f} | {percentage:.2f}% |\n"
                    
                    report += "\n"
            
            # Entity analysis (operator, basin)
            entity_cols = ['operador', 'cuenca']
            
            for entity_col in entity_cols:
                if entity_col in report_df.columns and report_df[entity_col].notna().sum() > 0:
                    entity_name = entity_col.replace('_', ' ').title()
                    report += f"## {entity_name} Analysis\n\n"
                    
                    if metrics:
                        entity_data = report_df[report_df['standardized_metric'] == metrics[0]].groupby(entity_col)['value'].sum().reset_index()
                        entity_data = entity_data.sort_values('value', ascending=False).head(10)
                        
                        report += f"### Top {entity_name}s by {metrics[0].replace('_', ' ').title()}\n\n"
                        report += f"| {entity_name} | Value | Percentage |\n"
                        report += "|" + "-" * (len(entity_name)+2) + "|---------|------------|\n"
                        
                        total = entity_data['value'].sum()
                        
                        for _, row in entity_data.iterrows():
                            entity = row[entity_col]
                            value = row['value']
                            percentage = (value / total * 100) if total > 0 else 0
                            
                            report += f"| {entity} | {value:,.2f} | {percentage:.2f}% |\n"
                        
                        report += "\n"
            
        elif output_format == 'html':
            # Similar content but in HTML format
            report = "<h1>Reserves Data Report</h1>\n"
            if year:
                report += f"<h2>Year: {year}</h2>\n"
            else:
                report += f"<h2>Years: {', '.join(map(str, years))}</h2>\n"
            
            # Summary statistics
            report += "<h2>Summary Statistics</h2>\n"
            
            # Table of metrics
            report += "<h3>Available Metrics</h3>\n"
            report += "<table border='1'>\n"
            report += "<tr><th>Metric</th><th>Total Value</th><th>Unit</th></tr>\n"
            
            for metric in metrics:
                metric_total = report_df[report_df['standardized_metric'] == metric]['value'].sum()
                unit = metric.split('_')[-1] if '_' in metric else ''
                report += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{metric_total:,.2f}</td><td>{unit}</td></tr>\n"
            
            report += "</table>\n"
            
            # Rest of the HTML report with similar sections...
            # (Condensed for brevity)
            
        else:
            report = "Unsupported output format"
        
        # Save the report
        report_name = f"reserves_report_{year}" if year else "reserves_report_all_years"
        report_path = self.output_dir / f"{report_name}.{output_format}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Saved report to {report_path}")
        
        return report
    
    def generate_comprehensive_dashboard(self, 
                                       output_file: Optional[str] = None,
                                       include_maps: bool = True) -> str:
        """
        Generate a comprehensive HTML dashboard with multiple visualizations.
        
        Args:
            output_file: Path to save HTML output (None for automatic naming)
            include_maps: Whether to include geographic maps
            
        Returns:
            str: Path to saved HTML file
        """
        combined_df = self.combine_datasets()
        
        if combined_df.empty:
            logger.warning("No data available for dashboard")
            return None
        
        # Get available years, metrics, and scopes
        years = sorted(combined_df['year'].unique().tolist())
        metrics = combined_df['standardized_metric'].dropna().unique().tolist()
        scopes = combined_df['scope'].unique().tolist()
        
        # Default path if not specified
        if output_file is None:
            output_file = self.output_dir / "reserves_dashboard.html"
        else:
            output_file = Path(output_file)
        
        # Generate multiple plots
        # This is a placeholder for the dashboard generation
        # In a real implementation, this would create multiple visualizations
        # and combine them into an HTML dashboard
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reserves Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4472C4; color: white; padding: 10px; }}
                .section {{ margin-top: 20px; border: 1px solid #ddd; padding: 15px; }}
                .plot-container {{ margin: 15px 0; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .table th {{ background-color: #4472C4; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Oil and Gas Reserves Dashboard</h1>
                <p>Years: {', '.join(map(str, years))}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <table class="table">
                    <tr>
                        <th>Metric</th>
                        <th>Total Value</th>
                        <th>Unit</th>
                    </tr>
        """
        
        # Add metric statistics
        for metric in metrics:
            metric_total = combined_df[combined_df['standardized_metric'] == metric]['value'].sum()
            unit = metric.split('_')[-1] if '_' in metric else ''
            html_content += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{metric_total:,.2f}</td>
                        <td>{unit}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>Key visualizations are available in the output directory:</p>
                <ul>
                    <li>Time series plots showing reserve changes over time</li>
                    <li>Bar charts comparing reserves across basins and provinces</li>
                    <li>Composition charts showing distribution of reserves</li>
                    <li>Geographic maps showing spatial distribution of reserves</li>
                </ul>
                <p>Generated plots are saved in: {self.output_dir}</p>
            </div>
            
            <div class="section">
                <h2>Data Quality</h2>
                <p>The following metrics were used to assess data quality:</p>
                <ul>
                    <li>Missing values: {combined_df.isna().sum().sum()} total missing values</li>
                    <li>Coverage: Data available for {len(years)} years</li>
                    <li>Metrics available: {len(metrics)} standardized metrics</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated dashboard at {output_file}")
        
        return str(output_file)
    
    def run_standard_visualizations(self, 
                                  years: Optional[List[int]] = None,
                                  metrics: Optional[List[str]] = None,
                                  scopes: Optional[List[str]] = None,
                                  include_maps: bool = True,
                                  generate_report: bool = True) -> Dict:
        """
        Run a standard set of visualizations for the given parameters.
        
        Args:
            years: List of years to visualize (None for all)
            metrics: List of metrics to visualize (None for standard set)
            scopes: List of scopes to include (None for all)
            include_maps: Whether to include geographic maps
            generate_report: Whether to generate a text report
            
        Returns:
            Dict: Summary of generated visualizations
        """
        # Load all data
        all_data = self.combine_datasets()
        
        if all_data.empty:
            logger.warning("No data available for visualization")
            return {"status": "error", "message": "No data available"}
        
        # Use all available years if not specified
        if years is None:
            years = sorted(all_data['year'].unique().tolist())
        
        # Use default metrics if not specified
        if metrics is None:
            default_metrics = ['oil_m3', 'gas_mm3', 'condensate_m3', 'total_boe']
            metrics = [m for m in default_metrics if m in all_data['standardized_metric'].unique()]
        
        # Use all available scopes if not specified
        if scopes is None:
            scopes = all_data['scope'].unique().tolist()
        
        results = {
            "years": years,
            "metrics": metrics,
            "scopes": scopes,
            "visualizations": []
        }
        
        logger.info(f"Generating standard visualizations for years: {years}")
        logger.info(f"Metrics: {metrics}")
        logger.info(f"Scopes: {scopes}")
        
        # 1. Time series for each metric and scope
        for metric in metrics:
            for scope in scopes:
                # Basic time series
                try:
                    self.plot_reserves_time_series(metric=metric, scope=scope)
                    results["visualizations"].append(f"time_series_{metric}_{scope}")
                except Exception as e:
                    logger.error(f"Error creating time series for {metric} {scope}: {e}")
                
                # By basin if available
                try:
                    self.plot_reserves_time_series(metric=metric, scope=scope, by_entity='cuenca')
                    results["visualizations"].append(f"time_series_{metric}_{scope}_by_cuenca")
                except Exception as e:
                    logger.error(f"Error creating time series by basin for {metric} {scope}: {e}")
        
        # 2. Bar charts for each year
        for year in years:
            for metric in metrics:
                for scope in scopes:
                    try:
                        self.plot_reserves_by_basin(year=year, metric=metric, scope=scope)
                        results["visualizations"].append(f"basin_chart_{year}_{metric}_{scope}")
                    except Exception as e:
                        logger.error(f"Error creating basin chart for {year} {metric} {scope}: {e}")
        
        # 3. Composition plots for each year
        for year in years:
            for scope in scopes:
                try:
                    self.plot_reserve_composition(year=year, scope=scope)
                    results["visualizations"].append(f"composition_{year}_{scope}")
                except Exception as e:
                    logger.error(f"Error creating composition chart for {year} {scope}: {e}")
        
        # 4. Geographic plots if requested and map file available
        if include_maps and self.map_file and self.map_file.exists():
            for year in years:
                for metric in metrics[:2]:  # Limit to first two metrics to avoid too many maps
                    for scope in scopes:
                        try:
                            self.plot_geographic_distribution(year=year, metric=metric, scope=scope)
                            results["visualizations"].append(f"map_{year}_{metric}_{scope}")
                        except Exception as e:
                            logger.error(f"Error creating map for {year} {metric} {scope}: {e}")
        
        # 5. Generate comprehensive report if requested
        if generate_report:
            try:
                # Generate reports for each year
                for year in years:
                    self.generate_reserve_report(year=year)
                    results["visualizations"].append(f"report_{year}")
                
                # Generate overall report
                self.generate_reserve_report()
                results["visualizations"].append("report_all_years")
                
                # Generate dashboard
                dashboard_path = self.generate_comprehensive_dashboard()
                results["dashboard"] = dashboard_path
            except Exception as e:
                logger.error(f"Error generating reports: {e}")
        
        logger.info(f"Generated {len(results['visualizations'])} visualizations")
        
        return results