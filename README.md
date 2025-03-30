# Oilfield Data Management System

This project is designed to track and analyze oilfield operations by integrating diverse datasets—from well production and drilling records to GIS and seismic data. The goal is to enable informed decision-making for both operational management and strategic planning, ensuring a comprehensive view of oilfield performance and asset status.

## Workflow & Data Pipeline

The data pipeline consists of several stages that convert raw data into actionable insights:

[![](https://mermaid.ink/img/pako:eNp11EtP4zAQAOC_YnmlPRUUO68mh5WgD2gFWgl2OSzlMIknbbSpXTnOQkH893Xzagoileom-WbGnqR-o6kSSGOaFeo53YA25OZuJYk9Lh7v4JlMwQC5V5VOsXwiZ2c_yOXjii7kGkuTK0lulagKXNGnJuiyJhNL6sBJgSBzuSbfyQMUuYBDTI8nNZ5aPFFKi1yCQXK_Lw1uyb0BKcBefD2NmdYxs67AQhpc65rYGkuVH6r1eFbjucVzBFNpJDO5ziWiHqp5ra6s-i3zLEfRLHpivwp1ZFc1u24rJ1DauRqlYY0fyMKSh7ys7IJfu5ndwm43rLmo5ZLVvTSoITX5Pzyw8qPhdcVykyjbjk93XXv3QkKxN3kKBflZmV1lGtW4skpsg3YbMnsxhzLtw20fVzOgFJ_4Lw2yzJTets-qbX_b2LZzX4ffKBBtU5rh-mt70qx2dc2wZO3I29H9nMXsCyQXJMuLIv6W1ceoNFr9xfgbZ4fPEF62EN2MZ6KHzA_81BnCSQfHmY_RMSOGwuVDOD2WHiPrYeaHqXOScdZlFBhmQQ8DYEkEQzjvMo6TRDg9BME8PxzCq36OCIOMbuRFkAzh9XGOtvhxjknqcDGEiy6jk4UZ9NBxAifwhnDJupTMLvvYH9cNIoYnkvfFMUHsZRKylKUn0u2ki37m99ID5o2tpCO61rmgsdEVjugW9RYOp_TtkGVFzQa39s8Y258CM6gKs6Ir-W7DdiD_KLXtIrWq1hsaZ1CU9qza2X0JpznYt3HbX9X2HUM9UZU0NHbHdQ4av9EXe-Z455Hnuk4YOGzsRW44onsacx6dj7ltPndYyLg79t5H9LUu65z7TuDyKGQ-91nEeTCiKHK7fdw222-9C7__ByvQsgM?type=png)](https://mermaid.live/edit#pako:eNp11EtP4zAQAOC_YnmlPRUUO68mh5WgD2gFWgl2OSzlMIknbbSpXTnOQkH893Xzagoileom-WbGnqR-o6kSSGOaFeo53YA25OZuJYk9Lh7v4JlMwQC5V5VOsXwiZ2c_yOXjii7kGkuTK0lulagKXNGnJuiyJhNL6sBJgSBzuSbfyQMUuYBDTI8nNZ5aPFFKi1yCQXK_Lw1uyb0BKcBefD2NmdYxs67AQhpc65rYGkuVH6r1eFbjucVzBFNpJDO5ziWiHqp5ra6s-i3zLEfRLHpivwp1ZFc1u24rJ1DauRqlYY0fyMKSh7ys7IJfu5ndwm43rLmo5ZLVvTSoITX5Pzyw8qPhdcVykyjbjk93XXv3QkKxN3kKBflZmV1lGtW4skpsg3YbMnsxhzLtw20fVzOgFJ_4Lw2yzJTets-qbX_b2LZzX4ffKBBtU5rh-mt70qx2dc2wZO3I29H9nMXsCyQXJMuLIv6W1ceoNFr9xfgbZ4fPEF62EN2MZ6KHzA_81BnCSQfHmY_RMSOGwuVDOD2WHiPrYeaHqXOScdZlFBhmQQ8DYEkEQzjvMo6TRDg9BME8PxzCq36OCIOMbuRFkAzh9XGOtvhxjknqcDGEiy6jk4UZ9NBxAifwhnDJupTMLvvYH9cNIoYnkvfFMUHsZRKylKUn0u2ki37m99ID5o2tpCO61rmgsdEVjugW9RYOp_TtkGVFzQa39s8Y258CM6gKs6Ir-W7DdiD_KLXtIrWq1hsaZ1CU9qza2X0JpznYt3HbX9X2HUM9UZU0NHbHdQ4av9EXe-Z455Hnuk4YOGzsRW44onsacx6dj7ltPndYyLg79t5H9LUu65z7TuDyKGQ-91nEeTCiKHK7fdw222-9C7__ByvQsgM)

1. **Data Ingestion & Preprocessing**

   - **Data Conversion:** CSV files are converted into GeoDataFrames.
   - **Spatial Standardization:** All layers are reprojected to a common coordinate system (EPSG:4326/3857) to ensure consistency.
   - **Data Cleaning:** Null values, duplicates, and outliers are removed or corrected.

1. **Database Design & Storage**

   - **Normalization:** Data is normalized into dedicated tables for wells, production, drilling, and reserves.
   - **Spatial Databases:** Geospatial data is stored in **PostGIS** or using **GeoParquet** for efficient spatial queries.

1. **ETL & Workflow Automation**

   - **ETL Frameworks:** Tools like **Apache Airflow** or **Prefect** schedule and monitor data ingestion, transformation, and syncing with field reports or partner APIs.
   - **Versioning:** Large datasets are versioned with **DVC** or **Git LFS**.

1. **Visualization & Mapping**
   - **Mapping:** Maps are created using **GeoPandas**, **Folium**, or **Leaflet** to visualize well statuses, drilling evolution, production heat maps, and reserve distributions.
   - **Interactive Dashboards:** Interactive visualizations are built with **Plotly**, **Bokeh**, or web frameworks like **Dash** and **Streamlit**.

## **Data Manifest**

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

## Data Relationships

[![](https://mermaid.ink/img/pako:eNqNVl1vm0gU_Stoqlat5KTMYIPhYaXUblJVyW4VV0m10Icxc7FHy4c1QBpvlP--FzB4wFRrPyQzl3PuPefOMMMLCTMBxCMbxXdb4_unIDXwt3j_PiBLXnBjgX_ibGO8Mx54LAUvZJYG5MOHIG2QebluqFcikanMC4WQJ_AD0g8YVbaA_GxY1U9IBWGVzri9P0avqL_I0hDyHJ_kPz3P41UeDcAw9-fnXZzJgjf0rMgxcx8KqTiRuFQyjmW6wQTt8FxZS-pfhbWPR4jjWpeoUmgIhsKTXQwFiN-DLKx9gFX-jPXeqO3uio_fVPYkcdhYGRKnp8Tv-x18_MRztDxKmf2GUsV4uh8n2Uj6sheKl7EMjWvFw6JU2KhxtOPfoV_FN836DiBja3CdYXt5XmCZdnjuGlxTH3skyuZJx16sHqq60WGu4dko_sft6scpYUzrDWT1QIY8Rr369FzNN9SvV8jAPnFxaNEGMg3B_BUImUBacLU3anTeQ41JOxpDYZrLM2V967USN8aVglraDqMajA1g95CDesqkGsFa_g3PjZ4WGe_7wDErTU7I0Ug7PNfGPfU7yh1PZQS4vu-MaxlD3ULVPPy_2uinK47js6t_bstXpNssrI-jtrCOY2O4qp9LfPG3IwTr6Oshi8sE_39d_fVnDznmZwUyT2SIbg6jc72sqM-WLd24lWnTwBwDGoj51hG0KlHffgAbE1Udhqio-neunEfq1_BeU3-BfvY8sgaibbgveNtkat_HaooGN9LFxR_GonnSXQdarDswtFjvCNDimggt2q3hSazaCFqwbakWqr0d5k3k7VtjVewrlc08jHmeLyEy6lvPiNCB9wasiEVigh6zf8B7Q2f2LDQH-PqEbvHzaAZuh2fgCIsN8O1BeaBEUTQH2lGimROawxJ4drUFBDiR3aFtTtcuH6Cr86HTA1yDW-7U5esB_PBWt3Lm67UwOwYXdDpzThktmqL4o1_Lsl0KA3S1m49eUf_R6zo0mRjAq63WqjcjJ-Id3DRt05527wMuHhiLLjOsATro2qEhDQ_Ti19SFFvP2j0HKZngp5kUxCtUCROSgEp4NSUvVdaAFFtI8JPBw6GACG_tIiBB-oo0vOP_zrKkZaqs3GyJF_E4x1m5w8sIlpLjdk66qMKXBdQiK9OCeFOzzkG8F_JMPMucXrpTyzId26TzqWs5E7InHmPu5ZzhMjGTOpRZ8-nrhPxblzUvZ6ZtMdehMzajLmP2hOBFh2_oXfPJWX95vv4H1wVRFw?type=png)](https://mermaid.live/edit#pako:eNqNVl1vm0gU_Stoqlat5KTMYIPhYaXUblJVyW4VV0m10Icxc7FHy4c1QBpvlP--FzB4wFRrPyQzl3PuPefOMMMLCTMBxCMbxXdb4_unIDXwt3j_PiBLXnBjgX_ibGO8Mx54LAUvZJYG5MOHIG2QebluqFcikanMC4WQJ_AD0g8YVbaA_GxY1U9IBWGVzri9P0avqL_I0hDyHJ_kPz3P41UeDcAw9-fnXZzJgjf0rMgxcx8KqTiRuFQyjmW6wQTt8FxZS-pfhbWPR4jjWpeoUmgIhsKTXQwFiN-DLKx9gFX-jPXeqO3uio_fVPYkcdhYGRKnp8Tv-x18_MRztDxKmf2GUsV4uh8n2Uj6sheKl7EMjWvFw6JU2KhxtOPfoV_FN836DiBja3CdYXt5XmCZdnjuGlxTH3skyuZJx16sHqq60WGu4dko_sft6scpYUzrDWT1QIY8Rr369FzNN9SvV8jAPnFxaNEGMg3B_BUImUBacLU3anTeQ41JOxpDYZrLM2V967USN8aVglraDqMajA1g95CDesqkGsFa_g3PjZ4WGe_7wDErTU7I0Ug7PNfGPfU7yh1PZQS4vu-MaxlD3ULVPPy_2uinK47js6t_bstXpNssrI-jtrCOY2O4qp9LfPG3IwTr6Oshi8sE_39d_fVnDznmZwUyT2SIbg6jc72sqM-WLd24lWnTwBwDGoj51hG0KlHffgAbE1Udhqio-neunEfq1_BeU3-BfvY8sgaibbgveNtkat_HaooGN9LFxR_GonnSXQdarDswtFjvCNDimggt2q3hSazaCFqwbakWqr0d5k3k7VtjVewrlc08jHmeLyEy6lvPiNCB9wasiEVigh6zf8B7Q2f2LDQH-PqEbvHzaAZuh2fgCIsN8O1BeaBEUTQH2lGimROawxJ4drUFBDiR3aFtTtcuH6Cr86HTA1yDW-7U5esB_PBWt3Lm67UwOwYXdDpzThktmqL4o1_Lsl0KA3S1m49eUf_R6zo0mRjAq63WqjcjJ-Id3DRt05527wMuHhiLLjOsATro2qEhDQ_Ti19SFFvP2j0HKZngp5kUxCtUCROSgEp4NSUvVdaAFFtI8JPBw6GACG_tIiBB-oo0vOP_zrKkZaqs3GyJF_E4x1m5w8sIlpLjdk66qMKXBdQiK9OCeFOzzkG8F_JMPMucXrpTyzId26TzqWs5E7InHmPu5ZzhMjGTOpRZ8-nrhPxblzUvZ6ZtMdehMzajLmP2hOBFh2_oXfPJWX95vv4H1wVRFw)

### Capabilities

- **Monitoring:**  
  Track well statuses, drilling activity, and production trends to optimize operational efficiency.

- **Spatial Analysis & Mapping:**  
  Leverage GIS data to overlay oilfield infrastructure, geological features, and legal boundaries, aiding in asset management and risk assessment.

- **Forecasting:**  
  Integrate production forecasts to support planning, budgeting, and strategic investment decisions.

- **Visualization:**  
  Enable dynamic mapping and dashboards to support detailed analyses and presentations for internal teams and external stakeholders.

### Data Model Diagram

[![](https://mermaid.ink/img/pako:eNqlVW1vmzAQ_ivIVb-lFZgQAt-6JuuqZW2VdJs2RYocfBBrYCPbtE3T_PcB4S1Nu0wbX9A9d37ufPfY3qBAUEA-AjliJJIkmXMj_yiTEGgmuHH_Yc532OmpMeaaaQZqB3wfTyYzY7Mzio9xbTBq3H1uIaUl45HBSQItGIFIQMu1EYuAFFkO4pUmOqvSbHe_u-nt6Ovl_fXtzZGUBfQIcbzI8Y8dnBINRioFzcqdLQq79YaxINoQLF48iDhLDjwRUXueqqrR9Hoyub65-p-aVJrRV9WUeCCSNIZ3a6WSxTHQRd5KkPu9mo5n4-m38ewfq6qGIEGBfICFXqcHyWvfGy25Gt9eTS_uPv34O2ksiWJ88Y5AliLjlMh1k6CR4hTiUjpqxdI9Pb68nJ2JTVcuvjFHK6Lm6I2wZn5_Cmra-Sqo3WkVuFtVRAWCa8J4GdrUfBkTperTExTGCMJyAsoI82H6JzAMHfB6eWvEL_BPMLjUxpV59sioXvk4fXpF0Gq6ZrFDHNKGxXIGTmAeYynlVAyk4qDghoOGY0CspUeOcVSqqHcTDpdLajYchFp9xz3Gkc8-v4fS1bomCcO8lIYkXAYmpm-Q7HfZuFCKRTwBrrv9riZUtrwLd9TSdrMb0OikblPX2eij3n_X2Yqk2dmcox6KJKPI1zKDHkpAJqQwUXlk5kivID8PqFAShZBksS6EtM2XpYT_FCKpV0qRRSvkhyRWuZWlxUVR3eQNKoFTkJf5UdLIdwYlB_I36An5ttk_9_q2bboD0xr2PdvtoTXyMfbOh9j2-ti0XAvbw_62h57LtOa5Yw5s7LmWgx3LwzjnA8q0kF92L0n5oGx_A9Qt9jk?type=png)](https://mermaid.live/edit#pako:eNqlVW1vmzAQ_ivIVb-lFZgQAt-6JuuqZW2VdJs2RYocfBBrYCPbtE3T_PcB4S1Nu0wbX9A9d37ufPfY3qBAUEA-AjliJJIkmXMj_yiTEGgmuHH_Yc532OmpMeaaaQZqB3wfTyYzY7Mzio9xbTBq3H1uIaUl45HBSQItGIFIQMu1EYuAFFkO4pUmOqvSbHe_u-nt6Ovl_fXtzZGUBfQIcbzI8Y8dnBINRioFzcqdLQq79YaxINoQLF48iDhLDjwRUXueqqrR9Hoyub65-p-aVJrRV9WUeCCSNIZ3a6WSxTHQRd5KkPu9mo5n4-m38ewfq6qGIEGBfICFXqcHyWvfGy25Gt9eTS_uPv34O2ksiWJ88Y5AliLjlMh1k6CR4hTiUjpqxdI9Pb68nJ2JTVcuvjFHK6Lm6I2wZn5_Cmra-Sqo3WkVuFtVRAWCa8J4GdrUfBkTperTExTGCMJyAsoI82H6JzAMHfB6eWvEL_BPMLjUxpV59sioXvk4fXpF0Gq6ZrFDHNKGxXIGTmAeYynlVAyk4qDghoOGY0CspUeOcVSqqHcTDpdLajYchFp9xz3Gkc8-v4fS1bomCcO8lIYkXAYmpm-Q7HfZuFCKRTwBrrv9riZUtrwLd9TSdrMb0OikblPX2eij3n_X2Yqk2dmcox6KJKPI1zKDHkpAJqQwUXlk5kivID8PqFAShZBksS6EtM2XpYT_FCKpV0qRRSvkhyRWuZWlxUVR3eQNKoFTkJf5UdLIdwYlB_I36An5ttk_9_q2bboD0xr2PdvtoTXyMfbOh9j2-ti0XAvbw_62h57LtOa5Yw5s7LmWgx3LwzjnA8q0kF92L0n5oGx_A9Qt9jk)

### Model Details

- **Wells:** Central entity representing each oilfield well with geospatial coordinates and operational status.
- **Production:** Time-series data associated with wells, tracking oil, gas, and water volumes.
- **Drilling:** Records documenting drilling activities, including timelines and meters drilled.
- **Reserves:** Data on proven and probable reserves tied to each well.
- **Geography:** Spatial datasets for basins, concessions, and administrative boundaries providing operational context.

## System Architecture

ArgGIS is built on a modern stack designed for geospatial data processing:

[![](https://mermaid.ink/img/pako:eNqdVl2PmzoQ_SsWVd-yWWzIBzxUSoCt9t62d6W0fWjTBwNDggqYGqNuutr_fg0EJyZRqoSXZDw-c2aOPbZfjIjFYLhGkrHf0ZZygT7762LxfW34VFC0YjWPoFobP9Dd3Tu0lOPB5w_oKS0hSwuQ4-ti2bo8BRGM003n8lqXL12Lgma7Kq3QB7oD3jr91hlI59e0qmmW_qEiZcXRjKCd8SBnfKmAo8dCAE9o1MVeF0h-VR1uOC23aJhw526-BZYBVltaQpJmbS1HPiJ93urrYNRq6nyOIEMPpxBbOv9Z_fdJjUIRn0nnWCfksbxkBRRCy2zZZJYWG6jEuNxpLMsmMcFpUSWM5_e6r0mvFmlW3f8lB31BDhG8hviJVeL94-r-PbAnyn_VIDQWr8ngiTOpZgUxakJVIKq_MA7X-RDPbzhbriKmuqQ-6Vco01Xwm0q9uhIsRyryRxbXx4tyPpGze-oQOWiyeWBZWucaY9BWnTExyCRoMvkXygz4eJOd4160m3WBNYtolqVZdo_sGmiJNYtoltXP7TrKw5pFem_XUj7WLKJZKlLXXAHWLKJZam4ldhnIvGUDZe4bsBKSxKNKcPYT3Dd4Mp1E5t68-53GYuuS8llD4j00DGNIwotQPICS26HW7VD7SmgHfvsW7TtGdjXKmk2H7tCGAxRIyP6vjjmWvZzzZAKOoiAwiy1yUc5lL2c0h2l0GToobEluh1pXQpUm-xNICVLWXDbSqSJer0gMs2SqCKYUhw69qIjXKwI4BJhdhA7K8siVUFWWOpH6umgeyt-Tsvw9QZIk8RFBEkYmiS-W5WMFhSi0LkIHZfnkdqh1JVQpoh-6vSxcXh8nogQHDim74ghnOMLRRVGCgyhRHJOL0EFlAbkdal0JVaJ8eVRKCKDZqRQP_QY05bmKVWTTnDqTc1KsC2NkbHgaG67gNYyMHHhOG9N4aeLKh8MWcnnhu_KvPL5onYnm7nuVsJIW3xjLeyRn9WZruAnNKmnVZUwF-CmVl2iuRrm85IB7rC6E4drTNobhvhjPhmuZ9tixLcucTU08tx1rNjJ2hkuIM54Ty7GJiWeYWHP7dWT8aWnN8cScWsSZ4QmZYIcQGQ_iVB4QH7uHaPseff0fLN5HdQ?type=png)](https://mermaid.live/edit#pako:eNqdVl2PmzoQ_SsWVd-yWWzIBzxUSoCt9t62d6W0fWjTBwNDggqYGqNuutr_fg0EJyZRqoSXZDw-c2aOPbZfjIjFYLhGkrHf0ZZygT7762LxfW34VFC0YjWPoFobP9Dd3Tu0lOPB5w_oKS0hSwuQ4-ti2bo8BRGM003n8lqXL12Lgma7Kq3QB7oD3jr91hlI59e0qmmW_qEiZcXRjKCd8SBnfKmAo8dCAE9o1MVeF0h-VR1uOC23aJhw526-BZYBVltaQpJmbS1HPiJ93urrYNRq6nyOIEMPpxBbOv9Z_fdJjUIRn0nnWCfksbxkBRRCy2zZZJYWG6jEuNxpLMsmMcFpUSWM5_e6r0mvFmlW3f8lB31BDhG8hviJVeL94-r-PbAnyn_VIDQWr8ngiTOpZgUxakJVIKq_MA7X-RDPbzhbriKmuqQ-6Vco01Xwm0q9uhIsRyryRxbXx4tyPpGze-oQOWiyeWBZWucaY9BWnTExyCRoMvkXygz4eJOd4160m3WBNYtolqVZdo_sGmiJNYtoltXP7TrKw5pFem_XUj7WLKJZKlLXXAHWLKJZam4ldhnIvGUDZe4bsBKSxKNKcPYT3Dd4Mp1E5t68-53GYuuS8llD4j00DGNIwotQPICS26HW7VD7SmgHfvsW7TtGdjXKmk2H7tCGAxRIyP6vjjmWvZzzZAKOoiAwiy1yUc5lL2c0h2l0GToobEluh1pXQpUm-xNICVLWXDbSqSJer0gMs2SqCKYUhw69qIjXKwI4BJhdhA7K8siVUFWWOpH6umgeyt-Tsvw9QZIk8RFBEkYmiS-W5WMFhSi0LkIHZfnkdqh1JVQpoh-6vSxcXh8nogQHDim74ghnOMLRRVGCgyhRHJOL0EFlAbkdal0JVaJ8eVRKCKDZqRQP_QY05bmKVWTTnDqTc1KsC2NkbHgaG67gNYyMHHhOG9N4aeLKh8MWcnnhu_KvPL5onYnm7nuVsJIW3xjLeyRn9WZruAnNKmnVZUwF-CmVl2iuRrm85IB7rC6E4drTNobhvhjPhmuZ9tixLcucTU08tx1rNjJ2hkuIM54Ty7GJiWeYWHP7dWT8aWnN8cScWsSZ4QmZYIcQGQ_iVB4QH7uHaPseff0fLN5HdQ)

### Visualization Outputs

The system generates several types of visualization outputs:

1. **Well Status Map**

   - Interactive map with wells color-coded by status (active, suspended, abandoned)
   - Filter controls for well type, operator, and time period
   - Pop-up information windows with detailed well information

2. **Production Heat Map**

   - Choropleth visualization of production intensity
   - Temporal slider to view production changes over time
   - Aggregation options by different geographic units

3. **Drilling Activity Evolution**

   - Animated time-series visualization showing drilling progression
   - Company-specific views of drilling activity
   - Comparison of drilling activity by basin or province

4. **Reserve Distribution Map**
   - Choropleth map showing reserve volumes by basin
   - Time series visualization of reserve changes
   - Comparison of different reserve categories (proven, probable, possible)

#### System Diagram

[![](https://mermaid.ink/img/pako:eNp9lF1r2zAUhv-KUG-TIsvfvhg0cT4KK4wWNtiyC9k6SsQcK0hy2jT0v09xEtcbwb7ya73POe85GB1xqTjgDItKvZYbpi36-ryqkXsefq3wN61KMAY4ypllK_wbjcdf0MSd_ICqQi-W2cagJ7ZzRxeqdUzPLG9KK1WNlsDsDVfuXLmWVSXrNXpw1r20BzTbq6o5Yf-5Z879DAb0HlAujdWyaG03Cs-d9QWk2coSTdUeNFvDDdvi35RzpaFkxqLv0jSsku-sS3GmJi21dNRjbV3RU2KXhZlNoZjmpis_PRvPIu-LWV_M-2JxFWe5bOXjaUNQSnPK99LsdkrbXiBjDxW4YYTbYXYHRMSCjdxm1B_I7giJSBRc5PhVcrvJ6O6tT04upPBEAmlH-n6UejBITq-kgAKgI4vYK71ykMyvaTnEIurIgHlBMkzOup7CsR0pipJQPkjOrz0TYL2efhqkrBgkF9eeSVFw0pGMe0EYD5LLrqcIe7ulEHOfDpKPn3Mm4H3OGcYlITdIPMJrLTnOrG5ghLegt-wk8fFUdYXtBrawwpl75SBYU7lfaFV_OGzH6p9Kba-kVs16gzPBKuNUs-PMQi7ZWrNt91VDzUFPVVNbnIVBWwNnR_yGM58E92ng-ySOiJcEqR-P8AFnlKb3CXW7psSLPeonwccIv7dtyX1IIp-msRfS0EspjUYYuLRKP51vpfZy-vgLfOt0sw?type=png)](https://mermaid.live/edit#pako:eNp9lF1r2zAUhv-KUG-TIsvfvhg0cT4KK4wWNtiyC9k6SsQcK0hy2jT0v09xEtcbwb7ya73POe85GB1xqTjgDItKvZYbpi36-ryqkXsefq3wN61KMAY4ypllK_wbjcdf0MSd_ICqQi-W2cagJ7ZzRxeqdUzPLG9KK1WNlsDsDVfuXLmWVSXrNXpw1r20BzTbq6o5Yf-5Z879DAb0HlAujdWyaG03Cs-d9QWk2coSTdUeNFvDDdvi35RzpaFkxqLv0jSsku-sS3GmJi21dNRjbV3RU2KXhZlNoZjmpis_PRvPIu-LWV_M-2JxFWe5bOXjaUNQSnPK99LsdkrbXiBjDxW4YYTbYXYHRMSCjdxm1B_I7giJSBRc5PhVcrvJ6O6tT04upPBEAmlH-n6UejBITq-kgAKgI4vYK71ykMyvaTnEIurIgHlBMkzOup7CsR0pipJQPkjOrz0TYL2efhqkrBgkF9eeSVFw0pGMe0EYD5LLrqcIe7ulEHOfDpKPn3Mm4H3OGcYlITdIPMJrLTnOrG5ghLegt-wk8fFUdYXtBrawwpl75SBYU7lfaFV_OGzH6p9Kba-kVs16gzPBKuNUs-PMQi7ZWrNt91VDzUFPVVNbnIVBWwNnR_yGM58E92ng-ySOiJcEqR-P8AFnlKb3CXW7psSLPeonwccIv7dtyX1IIp-msRfS0EspjUYYuLRKP51vpfZy-vgLfOt0sw)

## Tech Stack & Environment

**Core Technologies:**

- **Python 3.11+**
- **Pandas**, **GeoPandas**, **Shapely**, **Fiona**, **Pyproj**, **openpyxl**
- **Plotly**, **Bokeh**, **Folium/Leaflet**
- **Docker**, **Poetry** (for reproducible environments)
- **DVC/Git LFS** (for versioning large datasets)

**Enhancements:**

- **PostGIS** / **GeoParquet** for spatial database storage
- **Contextily** for ESRI/Mapbox basemaps

## Project Structure Overview

```
ArgGIS
├── backups                             # Backup storage for critical project data
│   ├── manifest_backup.csv             # Backup of data manifest
│   ├── reserves_backup.json            # Backup of reserves data
├── data                                # Main data directory
│   ├── interim                         # Intermediate data processing outputs
│   ├── processed                       # Final processed datasets
│   │   └── reserves                    # Processed reserve calculations
│   └── raw                             # Original unmodified source data
│       ├── administrative              # Administrative and management data
│       ├── drilling                    # Drilling operations data
│       ├── forecast                    # Production forecasting data
│       ├── geographical                # Geographic and spatial data
│       ├── production                  # Oil/gas production data
│       ├── reserves                    # Hydrocarbon reserves data
│       ├── reservoir                   # Reservoir characteristics data
│       ├── seismic                     # Seismic survey data
│       └── well                        # Well information and logs
├── metadata                            # Project and data documentation
│   ├── datasets                        # Dataset-specific metadata
│   │   ├── documentation.md            # Detailed data documentation
│   │   ├── file_mapping.csv            # File relationship mappings
│   │   ├── manifest.md                 # Dataset manifest documentation
│   │   └── manifest.yaml               # Dataset manifest configuration
│   └── reserves                        # Reserves-specific metadata
│       ├── reserves-metadata.json      # Reserves data documentation
├── notebooks                           # Jupyter notebooks for analysis
│   ├── EDA                             # Exploratory Data Analysis
│   ├── MappingPrototypes               # GIS mapping prototypes
│   ├── Reports                         # Analysis reports and presentations
├── outputs                             # Generated output files
│   ├── datastore                       # Processed data storage
│   ├── interactive                     # Interactive visualizations
│   ├── maps                            # Generated maps and spatial plots
│   └── tables                          # Tabular outputs and reports
└── pipeline                            # Data processing pipeline
    ├── config.py                       # Pipeline configuration settings
    ├── ingest.py                       # Data ingestion and loading
    ├── mapping                         # GIS mapping modules
    │   ├── map_drilling.py             # Drilling activity visualizations
    │   ├── map_production.py           # Production data mapping
    │   ├── map_reserves.py             # Reserves distribution mapping
    │   └── map_well_status.py          # Well status visualization
    ├── run.py                          # Pipeline execution script
    ├── transform                       # Data transformation modules
    │      └── reserve_processor.py     # Transformation logic for reserves
    └── utils                           # Utility functions and helpers
        ├── csv_utils.md                # CSV handling documentation
        ├── csv_utils.py                # CSV processing utilities
        ├── file_utils.py               # File management functions
        ├── geo_utils.py                # Geospatial processing utilities
        ├── join_utils.py               # Data joining and merging tools
        ├── metadata_utils.py           # Metadata handling utilities
        ├── shapefile_utils             # Shapefile processing tools
        └── transform_utils.py          # Data transformation utilities
```

### Module Responsibilities

- **Ingest Module:** Centralizes file reading and initial data formatting.
- **Transform Module:** Handles data normalization and business-specific transformations to create unified data models.
- **Mapping Module:** Focuses on geospatial rendering and visualization tasks.
- **Utility Functions:** Provide shared functionality across ingestion, transformation, and mapping tasks.

The project is organized into the following main components:

- **ArgGIS Directory:**  
  Contains the core code, backups, logs, metadata, notebooks for EDA and mapping prototypes, and the pipeline scripts for ingestion, transformation, and visualization.

- **Pipeline Module:**  
  Scripts and utilities are divided into submodules for ingestion (`ingest.py`), transformation (`transform/`), and mapping (`mapping/`), ensuring modularity and ease of maintenance.

- **Metadata & Documentation:**  
  Comprehensive documentation and data manifests are maintained within the `metadata` directory for clarity on dataset origins, formats, and transformations.

- **Visualization Outputs:**  
  Maps and interactive visualizations are generated and stored in the `outputs` directory, allowing for straightforward sharing and review.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Docker (optional, for containerized deployment)
- Git (for version control)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ArgGIS.git
   cd ArgGIS
   ```

2. Set up the environment:

   ```bash
   # Using Poetry
   poetry install

   # Or using pip
   pip install -r requirements.txt
   ```

3. Run the data pipeline:

   ```bash
   python pipeline/run.py
   ```

4. Launch the visualization interface:
   ```bash
   # Command to launch visualization interface
   ```

## License

This project is licensed under the [LICENSE] - see the LICENSE file for details.
