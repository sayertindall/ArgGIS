# Argentina GIS Pipeline TODO

## Configuration Centralization

- [ ] Audit and collect all configuration settings from:
  - pipeline/config.py
  - All scattered YAML files (focus on config.yaml)
  - Environment-specific settings
- [ ] Consolidate all settings into root config.yaml:
  - Dataset paths and CRS settings
  - Processing parameters and transforms logic
  - Logging configuration
  - Database connections

## Data Processing & Persistence

- [ ] Scan and normalize dataset headers:
  - Generate consistent headers manifest for data/raw directory
  - Normalize field names across drilling, production, and reserves data
  - Validate CRS consistency across GIS layers
- [ ] Implement metadata tracking:
  - Create metadata JSONs for each dataset in metadata/datasets
  - Track data lineage and transformations in YAML manifests
  - Version control for processed datasets
- [ ] Configure data storage:
  - Setup PostGIS/DuckDB for spatial data persistence
  - Implement GeoParquet for optimized storage
  - Define schema for wells, production, and reserves tables

## Pipeline Integration

- [ ] Consolidate ETL workflows:
  - Standardize data ingestion process
  - Create reusable transform utilities
  - Implement validation checks
- [ ] Establish logging system:
  - Central logging config in pipeline/**init**.py
  - Add context-specific logging
  - Track data operations and transformations
