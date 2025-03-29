import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explore_excel_files(directory: str) -> Dict:
    """
    Directly explore Excel files and list all their sheet names.
    No fancy algorithms, just straight examination of what's in each file.
    
    Args:
        directory: Directory containing the Excel files
    
    Returns:
        Dict: Dictionary with file details and sheet names
    """
    directory = Path(directory)
    files = list(directory.glob("*.xls")) + list(directory.glob("*.xlsx"))
    
    results = {
        "total_files": len(files),
        "file_details": []
    }
    
    for file_path in files:
        try:
            # Try both engines for maximum compatibility
            engines = ['openpyxl', 'xlrd']
            sheets = []
            used_engine = None
            
            for engine in engines:
                try:
                    excel = pd.ExcelFile(file_path, engine=engine)
                    sheets = excel.sheet_names
                    used_engine = engine
                    break
                except Exception as e:
                    logger.warning(f"Failed with engine {engine} for {file_path.name}: {e}")
            
            if not sheets:
                logger.error(f"Could not read {file_path.name} with any engine")
                continue
                
            # Extract year from filename
            year = None
            for part in file_path.stem.split('_'):
                if part.isdigit() and len(part) == 4:
                    year = part
                    break
            
            file_info = {
                "filename": file_path.name,
                "year": year,
                "sheets": sheets,
                "engine_used": used_engine
            }
            
            results["file_details"].append(file_info)
            logger.info(f"Processed {file_path.name}: found {len(sheets)} sheets")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    return results

def print_sheet_names_by_year(exploration_results: Dict) -> None:
    """
    Print sheet names grouped by year for easy pattern recognition.
    
    Args:
        exploration_results: Results from explore_excel_files function
    """
    # Group by year
    years = {}
    for file_info in exploration_results["file_details"]:
        year = file_info.get("year", "unknown")
        if year not in years:
            years[year] = []
        
        # Add sheet names with file info
        for sheet in file_info["sheets"]:
            years[year].append({
                "filename": file_info["filename"],
                "sheet": sheet
            })
    
    # Print by year
    print("\n=== SHEET NAMES BY YEAR ===")
    for year in sorted(years.keys()):
        print(f"\nYear: {year}")
        for item in years[year]:
            print(f"  {item['filename']} -> {item['sheet']}")

def build_sheet_mapping(exploration_results: Dict) -> Dict:
    """
    Build a mapping between sheet names and their likely type (EOC or EOL).
    
    Args:
        exploration_results: Results from explore_excel_files function
    
    Returns:
        Dict: Mapping of sheet names to types
    """
    sheet_mapping = {}
    
    # Keywords to identify sheet types
    eoc_keywords = ["concession", "eoc", "conc", "concesión", "concesion"]
    eol_keywords = ["life", "eol", "vida", "útil", "util"]
    
    # Check each file's sheets
    for file_info in exploration_results["file_details"]:
        for sheet in file_info["sheets"]:
            sheet_lower = sheet.lower()
            
            # Check if sheet matches EOC pattern
            if any(keyword in sheet_lower.replace(" ", "") for keyword in eoc_keywords):
                sheet_mapping[sheet] = "end_of_concession"
            
            # Check if sheet matches EOL pattern
            elif any(keyword in sheet_lower.replace(" ", "") for keyword in eol_keywords):
                sheet_mapping[sheet] = "end_of_life"
            
            # Unknown type
            else:
                sheet_mapping[sheet] = "unknown"
    
    return sheet_mapping

def create_fixed_manifest(directory: str, output_path: str) -> None:
    """
    Create a fixed manifest.csv file based on the actual sheets in the Excel files.
    
    Args:
        directory: Directory containing the Excel files
        output_path: Path to save the fixed manifest.csv
    """
    directory = Path(directory)
    
    # Explore Excel files
    results = explore_excel_files(directory)
    
    # Build sheet type mapping
    sheet_mapping = build_sheet_mapping(results)
    
    # Create manifest data
    manifest_data = []
    
    for file_info in results["file_details"]:
        filename = file_info["filename"]
        year = file_info.get("year")
        engine = file_info.get("engine_used", "openpyxl")
        
        for sheet in file_info["sheets"]:
            scope = sheet_mapping.get(sheet, "unknown")
            
            manifest_data.append({
                "filename": filename,
                "year": year,
                "sheet_name": sheet,
                "scope": scope,
                "engine": engine
            })
    
    # Create DataFrame and save to CSV
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(output_path, index=False)
    
    logger.info(f"Created fixed manifest with {len(manifest_df)} entries at {output_path}")
    print(f"Fixed manifest saved to {output_path}")

def preview_excel_content(file_path: str, num_rows: int = 5) -> Dict:
    """
    Preview the content of all sheets in an Excel file.
    
    Args:
        file_path: Path to the Excel file
        num_rows: Number of rows to preview from each sheet
    
    Returns:
        Dict: Dictionary with preview data for each sheet
    """
    file_path = Path(file_path)
    
    # Try both engines
    for engine in ['openpyxl', 'xlrd']:
        try:
            excel = pd.ExcelFile(file_path, engine=engine)
            break
        except Exception:
            continue
    else:
        return {"error": f"Could not open {file_path.name} with any engine"}
    
    preview_data = {}
    
    for sheet in excel.sheet_names:
        try:
            # Try first with no header
            df_no_header = pd.read_excel(file_path, sheet_name=sheet, header=None, nrows=num_rows, engine=engine)
            preview_data[f"{sheet}_raw"] = df_no_header.to_dict(orient='records')
            
            # Try to identify header row (simple approach - look at row 2, 3, 4)
            for header_row in [0, 1, 2, 3, 4]:
                if header_row < len(df_no_header):
                    df_with_header = pd.read_excel(
                        file_path, sheet_name=sheet, 
                        header=header_row, nrows=num_rows, engine=engine
                    )
                    preview_data[f"{sheet}_header{header_row}"] = df_with_header.to_dict(orient='records')
        except Exception as e:
            preview_data[f"{sheet}_error"] = str(e)
    
    return preview_data

def main(directory: str, output_manifest: str, preview_file: Optional[str] = None):
    """
    Main function to run the direct Excel exploration and fix the manifest.
    
    Args:
        directory: Directory containing the Excel files
        output_manifest: Path to save the fixed manifest.csv
        preview_file: Optional specific file to preview
    """
    print(f"Exploring Excel files in {directory}...")
    results = explore_excel_files(directory)
    
    print(f"\nFound {results['total_files']} Excel files")
    
    # Print sheet names by year
    print_sheet_names_by_year(results)
    
    # Build and print sheet mapping
    sheet_mapping = build_sheet_mapping(results)
    print("\n=== SHEET TYPE MAPPING ===")
    for sheet, sheet_type in sheet_mapping.items():
        print(f"{sheet} -> {sheet_type}")
    
    # Create fixed manifest
    create_fixed_manifest(directory, output_manifest)
    
    # Preview a specific file if requested
    if preview_file:
        print(f"\nPreviewing {preview_file}...")
        preview_data = preview_excel_content(Path(directory) / preview_file)
        
        # Output preview to a JSON file
        preview_output = f"{preview_file.replace('.', '_')}_preview.json"
        with open(preview_output, 'w') as f:
            json.dump(preview_data, f, indent=2)
        print(f"Preview saved to {preview_output}")

if __name__ == "__main__":
    main("ArgGIS/data/raw/reserves", "outputtttttt_manifest.csv")