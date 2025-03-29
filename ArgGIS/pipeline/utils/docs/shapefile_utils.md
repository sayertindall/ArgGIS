# Shapefile Utilities Module Documentation

## Overview

This module provides comprehensive utilities for working with shapefiles in Python. It offers functionality for loading, validating, transforming, querying, and visualizing spatial data stored in shapefile format.

## Installation

```python
pip install geopandas shapely pyproj rtree contextily folium
```

## Key Features

- Shapefile loading and validation
- Coordinate system transformations
- Spatial queries and analysis
- Data filtering and aggregation
- Visualization utilities
- Export capabilities
- Geometry operations
- Topology validation and cleaning
- Spatial statistics
- Raster-vector operations
- Interactive mapping
- Spatial indexing

## Module Functions

### `find_shapefiles(directory: str) -> List[str]`

Finds all shapefiles in a directory and its subdirectories.

**Parameters:**

- `directory`: Directory path to search for shapefiles

**Returns:**
List of paths to .shp files found

### `get_shapefile_components(shapefile_path: str) -> Dict[str, str]`

Gets all component files of a shapefile.

**Parameters:**

- `shapefile_path`: Path to the .shp file

**Returns:**
Dictionary mapping component types to file paths

### `unzip_file(zip_path: str, extract_dir: str = None) -> Dict[str, Any]`

Unzips a file to a specified directory.

**Parameters:**

- `zip_path`: Path to the zip file
- `extract_dir`: Directory to extract files to (defaults to same directory as zip file)

**Returns:**
Dictionary containing success status, extracted files list, and error message if any

## ShapefileUtils Class

### Initialization

```python
shapefile_utils = ShapefileUtils(shapefile_path)
```

**Parameters:**

- `shapefile_path`: Path to the shapefile

### Properties

- `gdf`: Returns the loaded GeoDataFrame (lazy-loaded)
- `components`: Dictionary of shapefile component files

### Methods

#### `load_shapefile() -> gpd.GeoDataFrame`

Loads and validates the shapefile.

**Returns:**
Loaded shapefile data as GeoDataFrame

#### `get_metadata() -> Dict`

Gets metadata about the shapefile.

**Returns:**
Dictionary with CRS, bounds, columns, etc.

#### `spatial_query(geometry: Union[Point, Polygon, LineString], operation: str = 'intersects', buffer_km: float = 0) -> gpd.GeoDataFrame`

Performs spatial queries on the data.

**Parameters:**

- `geometry`: Query geometry (Point, Polygon, or LineString)
- `operation`: Spatial operation ('intersects', 'within', 'contains', 'crosses', 'touches')
- `buffer_km`: Buffer distance in kilometers

**Returns:**
GeoDataFrame with query results

#### `transform_crs(target_crs: Union[str, int, dict, CRS]) -> None`

Transforms the data to a new coordinate reference system.

**Parameters:**

- `target_crs`: Target CRS (can be EPSG code, WKT string, PROJ string, or dict)

#### `filter_by_attribute(attribute: str, value: Any, operator: str = '==') -> gpd.GeoDataFrame`

Filters features by attribute values.

**Parameters:**

- `attribute`: Attribute/column name
- `value`: Value to compare against
- `operator`: Comparison operator ('==', '!=', '>', '<', '>=', '<=')

**Returns:**
Filtered GeoDataFrame

## Advanced Features

- Spatial indexing for fast queries
- Geometry validation and repair
- Interactive Folium maps
- Contextily basemap integration
- Spatial statistics calculations
- Raster-vector operations

## Examples

```python
# Basic usage
utils = ShapefileUtils('data.shp')
print(utils.gdf.head())

# Spatial query
point = Point(-58.4, -34.6)
results = utils.spatial_query(point, buffer_km=10)

# Coordinate transformation
utils.transform_crs('EPSG:3857')

# Attribute filtering
filtered = utils.filter_by_attribute('population', 10000, '>=')
```

## Dependencies

- geopandas
- shapely
- pyproj
- rtree
- contextily
- folium

## License

MIT License
