Here’s a clean and structured ingestion strategy based on your goals.

---

## **1. Rename All Files Systematically**

To make parsing and referencing deterministic, rename files using this pattern:

```
reserves_<year>_<scope>.xlsx
```

Where:

- `<year>` is extracted from the folder or filename
- `<scope>` is either `fin_concesion` or `fin_vida_util`

### **Examples**

| Original File Name                     | New File Name                      |
| -------------------------------------- | ---------------------------------- |
| `reservas al 31-12-2020.xlsx`          | `reserves_2020_fin_concesion.xlsx` |
| `02 - reservas fin vida util 2010.xls` | `reserves_2010_fin_vida_util.xls`  |
| `RESERVAS 2013 FC.xlsx`                | `reserves_2013_fin_concesion.xlsx` |
| `reservas 2012 Fin de vida util.xlsx`  | `reserves_2012_fin_vida_util.xlsx` |

> If a file contains **both sheets** (e.g. 2017–2023), keep the name generic:  
> `reserves_2023.xlsx` → still processed as two separate outputs based on sheet name.

---

## **2. Move to `data/raw/reservoir/reserves/`**

All renamed files should be stored here:

```
data/raw/reservoir/reserves/
```

Your pipeline expects to ingest from this folder directly.

---

## **3. Ingest via Manifest**

Create a manifest CSV or JSON to drive the transformation. This file should live in:

```
metadata/reservoir/reserves/reserve_manifest.csv
```

### **Manifest Columns**

| filename                        | year | sheet_name       | scope         | engine   |
| ------------------------------- | ---- | ---------------- | ------------- | -------- |
| reserves_2020.xlsx              | 2020 | Fin Concesion    | fin_concesion | openpyxl |
| reserves_2020.xlsx              | 2020 | Fin de vida útil | fin_vida_util | openpyxl |
| reserves_2010_fin_concesion.xls | 2010 | 1-FC             | fin_concesion | xlrd     |
| reserves_2010_fin_vida_util.xls | 2010 | 2-FVU            | fin_vida_util | xlrd     |

Use this manifest as the **single source of truth** for:

- File name
- Year
- Sheet to load
- Scope to tag
- Engine to use

---

### ✅ **Benefits of This Approach**

- Files are deterministically named → easy to reference
- Manifest-driven parsing → scalable and controlled
- Keeps raw data clean, normalized, and ready for transformation

---

Ready to:

- Generate a manifest CSV from your current folder layout
- Update the pipeline to use it directly

Let me know if you want the automation for that.
