import pandas as pd
from pathlib import Path

INPUT_DIR = Path("ArgGeoEnergy/data/raw/reserves")  # Set to your actual path if needed
MANIFEST_PATH = Path("ArgGeoEnergy/data/raw/reserves/manifest.csv")

def determine_scope_tag(suffix):
    return "end_of_concession" if suffix == "EOC" else "end_of_life"

def determine_sheet_name(suffix):
    return "End of Concession" if suffix == "EOC" else "End of Life"

def determine_engine(extension):
    return "openpyxl" if extension == ".xlsx" else "xlrd"

rows = []

for file in sorted(INPUT_DIR.glob("reserves_20*_E??.xls*")):
    name_parts = file.stem.split("_")  # e.g. ['reserves', '2020', 'EOC']
    if len(name_parts) < 3:
        continue  # skip unexpected format

    year = name_parts[1]
    suffix = name_parts[2]
    extension = file.suffix.lower()

    row = {
        "filename": file.name,
        "year": int(year),
        "sheet_name": determine_sheet_name(suffix),
        "scope": determine_scope_tag(suffix),
        "engine": determine_engine(extension)
    }
    rows.append(row)

# Save manifest CSV
manifest_df = pd.DataFrame(rows)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
manifest_df.to_csv(MANIFEST_PATH, index=False)

print(f"✓ Manifest saved to {MANIFEST_PATH}")
