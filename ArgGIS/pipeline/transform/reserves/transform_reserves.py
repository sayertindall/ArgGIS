import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from ArgGIS.pipeline.config import ROOT_DIR, DATA_RAW, DATA_PROCESSED, METADATA_DIR, TRANSFORM_CONFIG
from ArgGIS.pipeline.utils.transform_utils import (
    detect_header_row,
    normalize_columns,
    tidy_reserve_table
)
from ArgGIS.pipeline.utils.file_utils import (
    backup_directory,
    clean_directory,
    ensure_directory
)

# Get specific transform config
RESERVES_PATH = DATA_RAW / "reserves"
MANIFEST_PATH = RESERVES_PATH / "manifest.csv"
OUT_DIR = DATA_PROCESSED / "reserves"
META_DIR = METADATA_DIR / "reserves"
BACKUP_DIR = ROOT_DIR / "backups"

# Ensure directories exist
ensure_directory(OUT_DIR)
ensure_directory(META_DIR)
ensure_directory(BACKUP_DIR)


def load_file_map():
    df = pd.read_csv(MANIFEST_PATH)
    
    # Map scopes directly from filename
    for index, row in df.iterrows():
        filename = row['filename'].lower()
        if 'eoc' in filename:
            df.at[index, 'scope'] = 'end_of_concession'
        elif 'eol' in filename:
            df.at[index, 'scope'] = 'end_of_life'
        else:
            # Keep existing scope if it's already set
            pass
    
    entries = df.to_dict(orient="records")
    for e in entries:
        e["path"] = RESERVES_PATH / e["filename"]
    return entries

def process_excel_file(file_path, sheet_name, sheet_tag, engine):
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine=engine)
    except ValueError as e:
        # If sheet name is not found, try to use the first sheet instead
        if "Worksheet named" in str(e) and "not found" in str(e):
            try:
                # Open with ExcelFile to get sheet names
                excel = pd.ExcelFile(file_path, engine=engine)
                available_sheets = excel.sheet_names
                
                if not available_sheets:
                    raise ValueError(f"No sheets found in {file_path}")
                
                # Use the first available sheet
                first_sheet = available_sheets[0]
                print(f"! Sheet '{sheet_name}' not found in {file_path.name}, using '{first_sheet}' instead")
                sheet_name = first_sheet
                df_raw = pd.read_excel(file_path, sheet_name=first_sheet, header=None, engine=engine)
            except Exception as inner_e:
                raise ValueError(f"Failed to use alternative sheet: {inner_e}")
        else:
            raise e
    
    header_row = detect_header_row(df_raw)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, engine=engine)
    
    df = normalize_columns(df)
    df = tidy_reserve_table(df)
    
    df["source_file"] = file_path.name
    df["scope"] = sheet_tag
    df["year"] = extract_year_from_filename(file_path.name)
    
    # Get standardized scope abbreviation directly from filename
    scope_abbr = "eoc" if "eoc" in file_path.name.lower() else "eol"
    
    # Extract year and create standardized base name
    year = extract_year_from_filename(file_path.name)
    
    # Create standardized output names
    dataset = "reserves"
    standard_name = f"{dataset}_{year}_{scope_abbr}"
    
    # Save to CSV with standardized name
    output_name = f"{standard_name}.csv"
    df.to_csv(OUT_DIR / output_name, index=False)

    # Create metadata with standardized name
    meta = {
        "filename": file_path.name,
        "sheet": sheet_name,
        "header_row": header_row,
        "columns": list(df.columns),
        "row_count": len(df),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": sheet_tag,
        "scope_abbr": scope_abbr,
        "year": year
    }
    
    # Use standardized name for metadata
    meta_name = f"{standard_name}.json"
    (META_DIR / meta_name).write_text(pd.Series(meta).to_json(indent=2))

    print(f"✓ Processed {file_path.name} [{sheet_tag}] -> {output_name}")
    print(f"  Metadata saved as {meta_name}")


def extract_year_from_filename(name: str) -> int:
    # First try splitting by underscore
    parts = name.split('_')
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return int(part)
    
    # Then try splitting by space (fallback)
    for token in name.split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    
    return None


def fix_manifest():
    """Fix the manifest file by properly mapping scopes based on filenames"""
    if not MANIFEST_PATH.exists():
        print(f"Manifest file not found at {MANIFEST_PATH}")
        return False
    
    try:
        df = pd.read_csv(MANIFEST_PATH)
        
        # Make a backup of the manifest
        backup_path = BACKUP_DIR / f"manifest_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_path, index=False)
        print(f"✓ Backed up manifest to {backup_path}")
        
        # Map scopes based on filenames
        for index, row in df.iterrows():
            filename = row['filename'].lower()
            if 'eoc' in filename:
                df.at[index, 'scope'] = 'end_of_concession'
            elif 'eol' in filename:
                df.at[index, 'scope'] = 'end_of_life'
        
        # Save updated manifest
        df.to_csv(MANIFEST_PATH, index=False)
        print(f"✓ Updated manifest file with correct scopes")
        return True
    except Exception as e:
        print(f"✗ Failed to update manifest: {e}")
        return False


def update_sheet_names_in_manifest():
    """Update the sheet names in the manifest to match actual sheets in files"""
    if not MANIFEST_PATH.exists():
        print(f"Manifest file not found at {MANIFEST_PATH}")
        return False
    
    try:
        df = pd.read_csv(MANIFEST_PATH)
        updated_count = 0
        
        for index, row in df.iterrows():
            file_path = RESERVES_PATH / row['filename']
            if not file_path.exists():
                continue
                
            # Determine engine
            engine = 'openpyxl' if file_path.suffix.lower() == '.xlsx' else 'xlrd'
            
            try:
                # Get actual sheet names
                excel = pd.ExcelFile(file_path, engine=engine)
                available_sheets = excel.sheet_names
                
                if not available_sheets:
                    continue
                    
                # Update the sheet name to the first available one
                if row['sheet_name'] not in available_sheets:
                    df.at[index, 'sheet_name'] = available_sheets[0]
                    updated_count += 1
            except Exception as e:
                print(f"✗ Failed to read sheets from {file_path}: {e}")
        
        # Save updated manifest
        df.to_csv(MANIFEST_PATH, index=False)
        print(f"✓ Updated {updated_count} sheet names in manifest file")
        return True
    except Exception as e:
        print(f"✗ Failed to update sheet names in manifest: {e}")
        return False


def run_reserve_transform():
    """Main function to run the entire transformation process"""
    print(f"=== Starting Reserve Transformation with Standardized Naming ===")
    print(f"Raw data location: {RESERVES_PATH}")
    print(f"Output location: {OUT_DIR}")
    print(f"Metadata location: {META_DIR}")
    print(f"Backup location: {BACKUP_DIR}")
    print("")
    
    # First fix the manifest file to ensure proper scope mapping
    fix_manifest()
    
    # Update sheet names to match what's actually in the files
    update_sheet_names_in_manifest()
    
    # Backup output directory before processing
    print(f"Backing up and cleaning directories...")
    backup_directory(OUT_DIR, BACKUP_DIR)
    clean_directory(OUT_DIR, "*.csv") 
    
    # Clean metadata directory after backup
    backup_directory(META_DIR, BACKUP_DIR)
    clean_directory(META_DIR, "*.json")
    
    # Load and process the files
    file_map = load_file_map()
    print(f"\nProcessing {len(file_map)} files...")
    
    for entry in file_map:
        try:
            process_excel_file(entry["path"], entry["sheet_name"], entry["scope"], entry["engine"])
        except Exception as e:
            print(f"✗ Failed to process {entry['path']} [{entry['sheet_name']}] — {e}")
    
    print("\n✓ Transformation completed successfully")
    print(f"  Processed files are in: {OUT_DIR}")
    print(f"  Metadata files are in: {META_DIR}")
    print(f"  Backups are in: {BACKUP_DIR}")


if __name__ == "__main__":
    run_reserve_transform()