## **Data Manifest**

### **Data Inventory by Category**

#### **Geographical Data**

| ID                   | Dataset Name       | Description                                  | Format    | Path                                                                       | Status    |
| -------------------- | ------------------ | -------------------------------------------- | --------- | -------------------------------------------------------------------------- | --------- |
| `sedimentary_basins` | Sedimentary Basins | Sedimentary basin polygons                   | Shapefile | `data/raw/geographical/sedimentary-basins/geo-sedimentary-basins.shp`      | Collected |
| `basin_metadata`     | Basin Metadata     | Attributes describing each sedimentary basin | CSV       | `data/raw/geographical/basin-metadata/geo-sedimentary-basins-metadata.csv` | Collected |

#### **Administrative Data**

| ID                  | Dataset Name      | Description                                        | Format    | Path                                                                      | Status    |
| ------------------- | ----------------- | -------------------------------------------------- | --------- | ------------------------------------------------------------------------- | --------- |
| `exploitation_lots` | Exploitation Lots | Boundaries of exploitation areas                   | CSV       | `data/raw/administrative/exploitation-lots/admin-exploitation-blocks.csv` | Collected |
| `concessions`       | Concessions       | Legal exploitation concessions                     | Shapefile | `data/raw/administrative/concessions/admin-exploitation-concessions.shp`  | Collected |
| `satellite_rasters` | Raster Basemaps   | Contextual vector data (terrain, boundaries, etc.) | Shapefile | `data/raw/basemap/raster-basemaps/` (directory not yet populated)         | Optional  |

#### **Well Data**

| ID                | Dataset Name          | Description                                    | Format    | Path                                                           | Status    |
| ----------------- | --------------------- | ---------------------------------------------- | --------- | -------------------------------------------------------------- | --------- |
| `well_shapefile`  | Well Locations        | All wells incl. location, type, company, depth | Shapefile | `data/raw/well/locations/well-locations.shp`                   | Collected |
| `well_status_log` | Well Status Over Time | Operational status per well over time          | CSV       | `data/raw/well/production-history/well-production-history.csv` | Collected |

#### **Reservoir Data**

| ID                     | Dataset Name             | Description                                       | Format    | Path                                                                     | Status    |
| ---------------------- | ------------------------ | ------------------------------------------------- | --------- | ------------------------------------------------------------------------ | --------- |
| `reservoirs`           | Reservoirs               | Hydrocarbon reservoir polygons with operator info | Shapefile | `data/raw/reservoir/locations/reservoir-locations.shp`                   | Collected |
| `reservoirs_avg_depth` | Reservoirs by Avg. Depth | Reservoirs categorized by average depth           | Shapefile | `data/raw/reservoir/locations-by-depth/reservoir-locations-by-depth.shp` | Collected |
| `reserves_volumes`     | Oil & Gas Reserves       | Proven/probable reserve volumes                   | JSON      | `data/raw/reservoir/reserves/reserves-metadata.json`                     | Collected |

#### **Drilling Data**

| ID                            | Dataset Name                 | Description                                          | Format | Path                                                                                                               | Status    |
| ----------------------------- | ---------------------------- | ---------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------ | --------- |
| `drilled_meters_by_company`   | Drilling Meterage (Company)  | Meters drilled per company/month/concept (2009–2025) | CSV    | `data/raw/drilling/meterage-by-company/drilling-meters-by-company.csv`                                             | Collected |
| `drilled_meters_before_2009`  | Drilling Meterage (Pre-2009) | Historic drilled meters by company and concept       | CSV    | `data/raw/drilling/meterage-pre-2009/drilling-meters-1900-2009.csv`                                                | Collected |
| `wells_in_drilling`           | Active Drilling Wells        | Active drilling wells by company/month               | CSV    | `data/raw/drilling/active-wells/drilling-active-wells.csv`                                                         | Collected |
| `completed_wells`             | Completed Wells              | Completions with company, concept, location          | CSV    | `data/raw/drilling/completed-wells/drilling-completed-wells.csv`                                                   | Collected |
| `wells_completed_by_type`     | Completions by Type/Company  | Well completions by type and company                 | CSV    | `data/raw/drilling/completions-by-type-company/drilling-completed-wells-by-type-and-company.csv`                   | Collected |
| `wells_completed_by_concept`  | Completions by Concept/Prov. | Well completions by concept and province             | CSV    | `data/raw/drilling/completions-by-concept-province/drilling-completed-wells-by-concept-and-province-2009-2025.csv` | Collected |
| `wells_completed_by_welltype` | Completions by Type/Basin    | Well completions by type and basin                   | CSV    | `data/raw/drilling/completions-by-type-basin/drilling-completed-wells-by-type-and-basin.csv`                       | Collected |
| `fracture_data`               | Hydraulic Fracturing         | Hydraulic fracturing registry (Annex IV)             | CSV    | `data/raw/drilling/hydraulic-fracturing/well-fracture-data.csv`                                                    | Collected |

#### **Production Data**

| ID                        | Dataset Name                     | Description                               | Format | Path                                                                               | Status    |
| ------------------------- | -------------------------------- | ----------------------------------------- | ------ | ---------------------------------------------------------------------------------- | --------- |
| `production_by_reservoir` | Production by Reservoir          | Oil production volumes by reservoir       | CSV    | `data/raw/production/by-reservoir/production-oil-by-reservoir.csv`                 | Collected |
| `production_by_area`      | Production by Area & Consortium  | Oil production by area and consortium     | CSV    | `data/raw/production/by-area-consortium/production-oil-by-area-and-consortium.csv` | Collected |
| `gas_production_daily`    | Daily Gas Production by Province | Gas production daily averages by province | CSV    | `data/raw/production/gas-daily-by-province/production-gas-daily-by-province.csv`   | Collected |

#### **Forecast Data**

| ID                         | Dataset Name               | Description                      | Format | Path                                                   | Status    |
| -------------------------- | -------------------------- | -------------------------------- | ------ | ------------------------------------------------------ | --------- |
| `production_forecast_csv`  | Production Forecast (CSV)  | Projected oil and gas production | CSV    | `data/raw/forecast/csv/production-forecast.csv`        | Collected |
| `production_forecast_xlsx` | Production Forecast (XLSX) | Detailed forecast data for 2024  | XLSX   | `data/raw/forecast/xlsx/production-forecast-2024.xlsx` | Collected |

#### **Seismic Data**

| ID                   | Dataset Name       | Description                         | Format    | Path                                                 | Status    |
| -------------------- | ------------------ | ----------------------------------- | --------- | ---------------------------------------------------- | --------- |
| `2d_seismic_lines`   | 2D Seismic Lines   | Seismic exploration line geometries | Shapefile | `data/raw/seismic/2d-lines/seismic-2d-lines.shp`     | Collected |
| `3d_seismic_surveys` | 3D Seismic Surveys | 3D seismic survey polygons          | Shapefile | `data/raw/seismic/3d-surveys/seismic-3d-surveys.shp` | Collected |

---
