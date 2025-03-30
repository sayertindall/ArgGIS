# ArgGIS Core Module

The ArgGIS core module is the engine behind our oilfield data processing and visualization system. It unifies disparate geospatial and operational datasets into a coherent model that powers mapping, analytics, and reporting workflows. This README details the module's architecture, process flows, unified data models, key capabilities, and visualization outputs.

---

## Overview

ArgGIS is built to ingest, transform, and visualize oilfield data. It streamlines the pipeline from raw data ingestion to interactive outputs, ensuring that datasets such as well logs, production records, drilling data, and seismic surveys are seamlessly integrated and made actionable for decision support.

---

## Architecture & Workflow

### Process Flow Diagram

```mermaid
flowchart LR
    A["Raw Data Sources"] --> B["Ingestion: ingest.py"]
    B --> C["Preprocessing & Standardization"]
    C --> D["Unified Data Models"]
    D --> E["Transformation & ETL: transform/"]
    E --> F["Data Join"]
    F --> G["Mapping & Visualization: mapping/"]
    G --> H["Interactive Dashboards & Outputs"]

    subgraph Extract
      A
      B
    end

    subgraph Transform
      C
      D
      E
      F
    end

    subgraph Load
      G
      H
    end

    style A fill:#ffffff,stroke:#212121,stroke-width:2px
    style B fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style D fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    style E fill:#ede7f6,stroke:#6a1b9a,stroke-width:2px
    style F fill:#f8bbd0,stroke:#ad1457,stroke-width:2px
    style G fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
    style H fill:#fffde7,stroke:#fbc02d,stroke-width:2px
```

### Key Workflow Steps

1. **Data Ingestion**

   - **Script:** `ingest.py`
   - **Task:** Load raw files from various sources and convert them into a consistent format (GeoDataFrames, CSVs).

2. **Preprocessing & Standardization**

   - **Objective:** Ensure all data adheres to a common spatial reference (e.g., EPSG:4326/3857) and unified schema.
   - **Techniques:** Reprojection, data cleaning, null handling, and duplicate removal.

3. **Unified Data Models**

   - **Concept:** Abstract heterogeneous datasets into unified models for:
     - **Wells:** Location, status, production history.
     - **Drilling:** Activity records, meterage, completions.
     - **Production:** Volume metrics across reservoirs and areas.
     - **Geographical:** Basin boundaries, concessions, and administrative areas.
   - **Benefit:** Simplifies queries and joins across different data types.

4. **Data Transformation & ETL**

   - **Module:** `transform/`
   - **Tasks:**
     - Apply business rules to integrate datasets.
     - Normalize and join spatial and non-spatial data.
     - Generate transformed outputs ready for mapping.

5. **Mapping & Visualization**

   - **Module:** `mapping/`
   - **Capabilities:**
     - Render well status maps, production heat maps, drilling evolution animations, and reserve distribution overlays.
     - Use libraries such as GeoPandas and Folium for quick rendering and interactive tools like Dash for web dashboards.

6. **Output Generation**
   - **Outputs:**
     - Static maps in `outputs/maps`
     - Interactive dashboards in `outputs/interactive`
     - Data tables in `outputs/tables`

---

## Unified Data Models

### Data Relationships

```mermaid
graph TB
    C(("Data Catalog & Validation"))

    subgraph Administrative["Administrative Data"]
        direction LR
        A1[Concessions]:::admin
        A2["Exploitation Lots"]:::admin
    end

    subgraph Drilling["Drilling Data"]
        direction LR
        D1[Active Wells]:::drill
        D2[Completed Wells]:::drill
        D3["Completions by Concept/Province"]:::drill
        D4["Completions by Type/Basin"]:::drill
        D5["Completions by Type/Company"]:::drill
        D6["Hydraulic Fracturing"]:::drill
        D7[Meterage Data]:::drill
    end

    subgraph Forecast["Forecast Data"]
        direction LR
        F1[Production Forecast CSV]:::forecast
        F2[Production Forecast XLSX]:::forecast
    end

    subgraph Geographical["Geographical Data"]
        direction LR
        G1[Basin Metadata]:::geo
        G2[Sedimentary Basins]:::geo
    end

    subgraph Production["Production Data"]
        direction LR
        P1[Production by Area]:::prod
        P2[Production by Reservoir]:::prod
        P3[Gas Production Daily]:::prod
    end

    subgraph Reserves["Reserves Data"]
        direction LR
        R1[Reserves Manifest & Files]:::reserve
    end

    subgraph Reservoir["Reservoir Data"]
        direction LR
        RE1[Reservoir Locations]:::res
        RE2[Reservoir Locations by Depth]:::res
        RE3[Reserves Volumes JSON]:::res
    end

    subgraph Seismic["Seismic Data"]
        direction LR
        S1[2D Seismic Lines]:::seis
        S2[3D Seismic Surveys]:::seis
    end

    subgraph Well["Well Data"]
        direction LR
        W1[Well Locations]:::well
        W2[Well Production History]:::well
    end

    Administrative --> C
    Drilling --> C
    Forecast --> C
    Geographical --> C
    Production --> C
    Reserves --> C
    Reservoir --> C
    Seismic --> C
    Well --> C

    %% Styling
classDef admin fill:#e3f2fd,stroke:#1565c0
classDef drill fill:#e8f5e9,stroke:#2e7d32
classDef forecast fill:#fff8e1,stroke:#f57c00
classDef geo fill:#ede7f6,stroke:#6a1b9a
classDef prod fill:#e8eaf6,stroke:#3949ab
classDef reserve fill:#f8bbd0,stroke:#ad1457
classDef res fill:#f1f8e9,stroke:#33691e
classDef seis fill:#fffde7,stroke:#fbc02d
classDef well fill:#e0f7fa,stroke:#006064

style C fill:#ffebee,stroke:#b71c1c,stroke-width:3px

```

### Data Model Diagram

```mermaid
erDiagram
    WELLS {
      int id PK
      string name
      geometry location
      string status
    }
    PRODUCTION {
      int id PK
      int well_id FK
      date production_date
      float oil_volume
      float gas_volume
    }
    DRILLING {
      int id PK
      int well_id FK
      date spud_date
      date completion_date
      float drilled_meters
    }
    RESERVES {
      int id PK
      int well_id FK
      string reserve_type
      float reserve_volume
    }
    GEOGRAPHY {
      int id PK
      string basin_name
      geometry boundary
    }
    WELLS ||--o{ PRODUCTION : "has"
    WELLS ||--o{ DRILLING : "has"
    WELLS ||--o{ RESERVES : "has"
    GEOGRAPHY ||--o{ WELLS : "contains"
```

### Model Details

- **Wells:** Central entity representing each oilfield well with geospatial coordinates and operational status.
- **Production:** Time-series data associated with wells, tracking oil, gas, and water volumes.
- **Drilling:** Records documenting drilling activities, including timelines and meters drilled.
- **Reserves:** Data on proven and probable reserves tied to each well.
- **Geography:** Spatial datasets for basins, concessions, and administrative boundaries providing operational context.

---

## Key Capabilities

ArgGIS offers a range of analytical and visualization capabilities:

1. **Well Status Mapping**

   - Interactive maps showing well locations color-coded by status.
   - Filtering by operator, status, depth, and other attributes.
   - Temporal views of well status changes.

2. **Production Heat Maps**

   - Visualize production intensity across geographic areas.
   - Aggregation by grid, hexbin, lease, or basin.
   - Time-series heat maps showing production evolution.

3. **Drilling Activity Analysis**

   - Track drilling evolution over time with animated visualizations.
   - Analyze drilling efficiency metrics by company and basin.
   - Compare actual vs. planned drilling activities.

4. **Reserve Visualization**

   - Map proven and probable reserves by basin and field.
   - Track reserve changes over time (2004–2023).
   - Compare reserves across different calculation methodologies (EOC vs. EOL).

5. **Seismic Coverage Analysis**

   - Visualize 2D seismic line density.
   - Map 3D seismic survey coverage.
   - Identify areas for potential new seismic acquisition.

6. **Production Forecasting**
   - Visualize production forecasts against historical production.
   - Analyze production decline curves.
   - Compare forecast scenarios.

---

## Visualization Outputs

ArgGIS generates several types of visualization outputs:

```mermaid
flowchart LR
    A["Processed Data"] --> B["Well Status Map"]
    A --> C["Production Heat Map"]
    A --> D["Drilling Activity Evolution"]
    A --> E["Reserve Distribution Map"]
    A --> F["Seismic Coverage Map"]
    A --> G["Production Forecast Visualization"]

    B --> H["Interactive Dashboards"]
    C --> H
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I["Decision Support"]

    style A fill:#e0f7fa,stroke:#006064,stroke-width:2px
    style B fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style C fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    style D fill:#ede7f6,stroke:#4a148c,stroke-width:2px
    style E fill:#fffde7,stroke:#fbc02d,stroke-width:2px
    style F fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
    style G fill:#f8bbd0,stroke:#ad1457,stroke-width:2px
    style H fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style I fill:#fff8e1,stroke:#f57c00,stroke-width:2px
```

### Example Visualizations

1. **Well Status Map**

   - Interactive map with wells color-coded by status (active, suspended, abandoned).
   - Filter controls for well type, operator, and time period.
   - Pop-up windows with detailed well information.

2. **Production Heat Map**

   - Choropleth visualization of production intensity.
   - Temporal slider to view production changes over time.
   - Aggregation options by different geographic units.

3. **Drilling Activity Evolution**

   - Animated time-series visualization showing drilling progression.
   - Company-specific views of drilling activity.
   - Comparison of drilling activity by basin or province.

4. **Reserve Distribution Map**
   - Choropleth map showing reserve volumes by basin.
   - Time series visualization of reserve changes.
   - Comparison of different reserve categories (proven, probable, possible).

---

## Technical Architecture

ArgGIS is built on a modern stack designed for geospatial data processing:

```mermaid
flowchart TD
A["Data Sources"] --> B["ETL Pipeline"]
B --> C["Data Storage"]
C --> D["Analysis Layer"]
D --> E["Visualization Layer"]
E --> F["User Interface"]

    subgraph "Data Sources"
        A1["Shapefiles"]
        A2["CSVs"]
        A3["Excel Files"]
        A4["JSON"]
    end

    subgraph "ETL Pipeline Components"
        B1["ingest.py"]
        B2["transform/"]
        B3["utils/"]
    end

    subgraph "Data Storage"
        C1["PostGIS/GeoParquet"]
        C2["Processed Datasets"]
    end

    subgraph "Analysis Layer"
        D1["GeoPandas"]
        D2["Shapely"]
        D3["Custom Analysis Modules"]
    end

    subgraph "Visualization Layer"
        E1["Folium"]
        E2["Plotly"]
        E3["Kepler.gl"]
    end

    A --> A1
    A --> A2
    A --> A3
    A --> A4

    B --> B1
    B --> B2
    B --> B3

    C --> C1
    C --> C2

    D --> D1
    D --> D2
    D --> D3

    E --> E1
    E --> E2
    E --> E3

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style A1 fill:#bbdefb,stroke:#1565c0,stroke-width:1px
    style A2 fill:#bbdefb,stroke:#1565c0,stroke-width:1px
    style A3 fill:#bbdefb,stroke:#1565c0,stroke-width:1px
    style A4 fill:#bbdefb,stroke:#1565c0,stroke-width:1px

    %% Processing layer - green tones
    style B fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style B1 fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px
    style B2 fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px
    style B3 fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px

    %% Storage layer - purple tones
    style C fill:#ede7f6,stroke:#6a1b9a,stroke-width:2px
    style C1 fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px
    style C2 fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px

    %% Analysis layer - amber tones
    style D fill:#fffde7,stroke:#fbc02d,stroke-width:2px
    style D1 fill:#ffecb3,stroke:#fbc02d,stroke-width:1px
    style D2 fill:#ffecb3,stroke:#fbc02d,stroke-width:1px
    style D3 fill:#ffecb3,stroke:#fbc02d,stroke-width:1px

    %% Visualization layer - red tones
    style E fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    style E1 fill:#ffcdd2,stroke:#b71c1c,stroke-width:1px
    style E2 fill:#ffcdd2,stroke:#b71c1c,stroke-width:1px
    style E3 fill:#ffcdd2,stroke:#b71c1c,stroke-width:1px

    %% UI layer - teal tones
    style F fill:#e0f2f1,stroke:#00695c,stroke-width:2px

```

### Core Technologies

- **Python 3.11+:** Foundation for data processing and analysis.
- **GeoPandas, Shapely, Fiona:** Core geospatial libraries.
- **Folium, Kepler.gl, Plotly:** Visualization libraries.
- **PostGIS/GeoParquet:** Spatial data storage solutions.
- **Docker, Poetry:** Environment management and reproducibility.

---

## Directory Structure & Key Components

```
ArgGIS/
├── pipeline/
│   ├── ingest.py                  # Data ingestion routines
│   ├── transform/
│   │   ├── __init__.py
│   │   └── reserve_processor.py   # Transformation logic for reserves
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── map_well_status.py     # Mapping for well status
│   │   ├── map_production.py      # Production heat maps
│   │   ├── map_drilling.py        # Drilling activity visualizations
│   │   └── map_reserves.py        # Reserve distribution mapping
│   └── utils/
│       ├── geo_utils.py           # Geospatial helper functions
│       ├── join_utils.py          # Data joining utilities
│       └── file_utils.py          # File management routines
```

### Module Responsibilities

- **Ingest Module:** Centralizes file reading and initial data formatting.
- **Transform Module:** Handles data normalization and business-specific transformations to create unified data models.
- **Mapping Module:** Focuses on geospatial rendering and visualization tasks.
- **Utility Functions:** Provide shared functionality across ingestion, transformation, and mapping tasks.

---

## Improvements & Future Enhancements

- **Enhanced Data Validation:** Integrate comprehensive validation rules during ingestion to ensure data integrity.
- **Scalability:** Leverage distributed processing for large datasets using tools like Dask.
- **Real-Time Updates:** Implement streaming data ingestion for near-real-time monitoring.
- **Advanced Analytics:** Incorporate machine learning modules for predictive maintenance and production forecasting.

---
