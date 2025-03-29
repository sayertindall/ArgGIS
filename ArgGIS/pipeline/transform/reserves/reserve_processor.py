import os
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Union, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ReserveProcessor")

class ReserveDataProcessor:
    """
    A comprehensive processor for standardizing, validating, and transforming
    oil and gas reserve data across multiple years and scopes.
    """
    
    def __init__(self, 
                 raw_dir: Union[str, Path], 
                 processed_dir: Union[str, Path],
                 metadata_dir: Union[str, Path],
                 backup_dir: Union[str, Path], 
                 manifest_path: Optional[Union[str, Path]] = None):
        """
        Initialize the processor with path configurations.
        
        Args:
            raw_dir: Directory containing raw reserve Excel files
            processed_dir: Directory for processed CSV outputs
            metadata_dir: Directory for metadata files
            backup_dir: Directory for backups
            manifest_path: Path to manifest file (defaults to manifest.csv in raw_dir)
        """
        # Convert all paths to Path objects
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.metadata_dir = Path(metadata_dir)
        self.backup_dir = Path(backup_dir)
        
        if manifest_path:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.raw_dir / "manifest.csv"
        
        # Ensure all directories exist
        self._ensure_directories()
        
        # Common column mappings for standardization
        self.column_mappings = {
            # Basin/field information - various forms these might appear
            'cuenca': ['cuenca', 'basin', 'bacin', 'cuenca_', 'Basin'],
            'provincia': ['provincia', 'province', 'prov', 'pcia'],
            'concesion': ['concesión', 'concesion', 'concession', 'conc'],
            'yacimiento': ['yacimiento', 'field', 'yac'],
            'operador': ['operador', 'operator', 'company', 'empresa'],
            
            # Reserve types
            'comprobadas': ['comprobadas', 'probadas', 'proved', 'proven', 'p1', '1p'],
            'probables': ['probables', 'probable', 'p2', '2p'],
            'posibles': ['posibles', 'possible', 'p3', '3p'],
            
            # Combined fields (need special handling)
            'cuenca_provincia': ['cuenca_provincia', 'basin_province'],
            'cuenca_provincia_concesion_yacimiento': [
                'cuenca_provincia_concesión_y_yacimiento', 
                'cuenca_provincia_concesion_y_yacimiento',
                'basin_province_concession_field'
            ]
        }
        
        # Reserve value metrics we expect to encounter
        self.reserve_metrics = [
            'petroleo_m3', 'oil_m3', 'oil_bbl',
            'gas_m3', 'gas_mm3', 'gas_bcf',
            'condensado_m3', 'condensate_m3', 'condensate_bbl',
            'gasolina_m3', 'gasoline_m3', 'gasoline_bbl',
            'total_boe', 'total_m3'
        ]
        
        # Unit conversion factors
        self.unit_conversions = {
            'm3_to_bbl': 6.2898,  # cubic meters to barrels
            'mm3_to_bcf': 0.0353147,  # million cubic meters to billion cubic feet
            'bbl_to_m3': 0.158987,  # barrels to cubic meters
            'bcf_to_mm3': 28.3168  # billion cubic feet to million cubic meters
        }
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        for dir_path in [self.raw_dir, self.processed_dir, self.metadata_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _backup_directory(self, source_dir: Path, file_pattern: str = "*") -> Path:
        """
        Create a timestamped backup of files in a directory.
        
        Args:
            source_dir: Directory to backup
            file_pattern: Pattern of files to include
            
        Returns:
            Path: Path to the backup directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_subdir = self.backup_dir / f"{source_dir.name}_{timestamp}"
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to backup
        for file_path in source_dir.glob(file_pattern):
            if file_path.is_file():
                import shutil
                shutil.copy2(file_path, backup_subdir / file_path.name)
        
        logger.info(f"Backed up {source_dir} to {backup_subdir}")
        return backup_subdir
    
    def _clean_directory(self, directory: Path, file_pattern: str = "*") -> None:
        """
        Remove files matching a pattern from a directory.
        
        Args:
            directory: Directory to clean
            file_pattern: Pattern of files to remove
        """
        for file_path in directory.glob(file_pattern):
            if file_path.is_file():
                file_path.unlink()
        
        logger.info(f"Cleaned files matching '{file_pattern}' from {directory}")
    
    def _detect_header_row(self, df: pd.DataFrame, max_check_rows: int = 15) -> int:
        """
        Detect the most likely header row in a DataFrame.
        Uses multiple heuristics to find the best header candidate.
        
        Args:
            df: DataFrame to analyze
            max_check_rows: Maximum number of rows to check
            
        Returns:
            int: Index of the detected header row
        """
        best_row = 0
        best_score = 0
        
        # Limit check to available rows or max_check_rows
        max_row = min(len(df), max_check_rows)
        
        for i in range(max_row):
            row = df.iloc[i]
            
            # Skip completely empty rows
            if row.isna().all():
                continue
            
            # Calculate various quality scores
            non_null_count = row.notna().sum()
            string_count = sum(1 for val in row if isinstance(val, str))
            numeric_count = sum(1 for val in row if isinstance(val, (int, float)) and not isinstance(val, bool))
            unique_count = len(set(val for val in row if pd.notna(val)))
            
            # Headers usually have strings and good coverage
            score = (string_count * 2) + non_null_count + unique_count - (numeric_count * 0.5)
            
            # Check string characteristics of potential headers
            if string_count > 0:
                # Headers often have lowercase words or words with underscores
                lowercase_count = sum(1 for val in row if isinstance(val, str) and val.lower() == val)
                underscore_count = sum(1 for val in row if isinstance(val, str) and '_' in val)
                score += lowercase_count + (underscore_count * 2)
                
                # Check for common header terms
                header_term_count = sum(1 for val in row if isinstance(val, str) and 
                                       any(term in val.lower() for term in ['id', 'name', 'code', 'date', 'value',
                                                                           'basin', 'field', 'cuenca', 'yacimiento']))
                score += header_term_count * 3
            
            if score > best_score:
                best_score = score
                best_row = i
        
        logger.info(f"Detected header row at index {best_row} with score {best_score}")
        return best_row
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to a standard format.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            DataFrame: DataFrame with normalized column names
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Fix column names
        result.columns = (
            pd.Series(result.columns)
            .astype(str)
            .str.strip()
            .str.lower()
            .str.normalize('NFKD')  # Normalize Unicode characters
            .str.encode('ascii', errors='ignore')  # Remove non-ASCII
            .str.decode('ascii')
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
        
        # Replace empty column names with unnamed_{position}
        result.columns = [f"unnamed_{i}" if not col or col.isspace() 
                        else col for i, col in enumerate(result.columns)]
        
        return result
    
    def _map_standard_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Map various column names to standardized names using fuzzy matching.
        Creates a mapping dictionary to record the transformations.
        
        Args:
            df: DataFrame with normalized column names
            
        Returns:
            Tuple[DataFrame, Dict]: DataFrame with mapped columns and mapping dictionary
        """
        from fuzzywuzzy import fuzz
        result = df.copy()
        mapping = {}
        
        # First pass: direct matches and close matches
        for std_col, variants in self.column_mappings.items():
            for col in result.columns:
                # Check for direct matches
                if col in variants:
                    mapping[col] = std_col
                    continue
                
                # Check for fuzzy matches (if column not already mapped)
                if col not in mapping:
                    for variant in variants:
                        # Use token set ratio to handle word rearrangement and partial matches
                        similarity = fuzz.token_set_ratio(col, variant)
                        if similarity >= 85:  # High threshold for confidence
                            mapping[col] = std_col
                            logger.info(f"Fuzzy matched '{col}' to '{std_col}' (similarity: {similarity})")
                            break
        
        # Second pass: handle combined fields
        for col in result.columns:
            if col not in mapping:
                # Check if this might be a combined field
                if any(term in col for term in ['cuenca', 'basin', 'provincia', 'yacimiento', 'field']):
                    parts = []
                    if any(term in col for term in ['cuenca', 'basin']):
                        parts.append('cuenca')
                    if any(term in col for term in ['provincia', 'province']):
                        parts.append('provincia')
                    if any(term in col for term in ['concesion', 'concesión']):
                        parts.append('concesion')
                    if any(term in col for term in ['yacimiento', 'field']):
                        parts.append('yacimiento')
                    
                    if parts:
                        combined = '_'.join(parts)
                        mapping[col] = combined
                        logger.info(f"Combined field: '{col}' mapped to '{combined}'")
        
        # Apply mappings to column names
        new_columns = []
        for col in result.columns:
            new_columns.append(mapping.get(col, col))
        
        result.columns = new_columns
        return result, mapping
    
    def _split_combined_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split combined fields like cuenca_provincia_concesion_yacimiento into separate columns.
        
        Args:
            df: DataFrame with potentially combined fields
            
        Returns:
            DataFrame: DataFrame with split fields
        """
        result = df.copy()
        
        # Check for combined fields
        combined_patterns = [
            ('cuenca_provincia_concesion_yacimiento', ['cuenca', 'provincia', 'concesion', 'yacimiento']),
            ('cuenca_provincia', ['cuenca', 'provincia'])
        ]
        
        for combined_col, split_cols in combined_patterns:
            if combined_col in result.columns:
                logger.info(f"Splitting combined field: {combined_col}")
                
                # Try different splitting strategies
                # First try dash/hyphen splitting
                if result[combined_col].str.contains('-').any():
                    parts = result[combined_col].str.split('-', expand=True)
                    num_parts = len(parts.columns)
                    
                    # Map parts to their respective columns
                    for i, col in enumerate(split_cols[:num_parts]):
                        if col not in result.columns:  # Only add if not already present
                            result[col] = parts[i].str.strip()
                
                # Next try slash splitting
                elif result[combined_col].str.contains('/').any():
                    parts = result[combined_col].str.split('/', expand=True)
                    num_parts = len(parts.columns)
                    
                    for i, col in enumerate(split_cols[:num_parts]):
                        if col not in result.columns:
                            result[col] = parts[i].str.strip()
                
                # Try comma splitting
                elif result[combined_col].str.contains(',').any():
                    parts = result[combined_col].str.split(',', expand=True)
                    num_parts = len(parts.columns)
                    
                    for i, col in enumerate(split_cols[:num_parts]):
                        if col not in result.columns:
                            result[col] = parts[i].str.strip()
        
        return result
    
    def _convert_to_tidy_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the data to a tidy format with consistent id and value columns.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            DataFrame: DataFrame in tidy format
        """
        # Identify likely ID columns and value columns
        all_columns = set(df.columns)
        
        # Standard ID columns we always want to preserve
        known_id_cols = {'cuenca', 'provincia', 'concesion', 'yacimiento', 'operador', 
                         'comprobadas', 'probables', 'posibles'}
        
        # Get actual ID columns that exist in the DataFrame
        id_cols = list(known_id_cols.intersection(all_columns))
        
        # Add unnamed columns that are likely part of the identifiers
        for col in df.columns:
            if col.startswith('unnamed_') and col not in id_cols:
                # Check if this unnamed column has mostly string values
                if df[col].dtype == 'object' and df[col].notna().sum() > 0:
                    string_ratio = sum(isinstance(val, str) for val in df[col].dropna()) / df[col].notna().sum()
                    if string_ratio > 0.7:  # If more than 70% are strings, consider it an ID column
                        id_cols.append(col)
        
        # All other columns are considered value columns
        value_cols = [col for col in df.columns if col not in id_cols]
        
        # Check if we already have a tidy format (with 'metric' and 'value' columns)
        if 'metric' in df.columns and 'value' in df.columns:
            logger.info("Data already in tidy format")
            return df
        
        # If no value columns, cannot melt
        if not value_cols:
            logger.warning("No value columns found, cannot convert to tidy format")
            return df
        
        logger.info(f"Converting to tidy format with ID columns: {id_cols}")
        logger.info(f"Value columns: {value_cols}")
        
        # Melt the DataFrame to get tidy format
        df_melted = df.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name="metric",
            value_name="value"
        )
        
        return df_melted
    
    def _clean_numeric_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert numeric values to proper data types.
        
        Args:
            df: DataFrame with potentially messy numeric data
            
        Returns:
            DataFrame: DataFrame with cleaned numeric values
        """
        result = df.copy()
        
        # Convert value column to numeric if it exists
        if 'value' in result.columns:
            # First handle string representations
            if result['value'].dtype == 'object':
                # Replace various formats of missing values
                result['value'] = result['value'].replace(['', 'N/A', 'n/a', 'na', 'NA', '-', '–'], np.nan)
                
                # Remove thousands separators and replace decimal commas
                result['value'] = result['value'].astype(str).str.replace(',', '')
                
                # Convert to numeric
                result['value'] = pd.to_numeric(result['value'], errors='coerce')
            
            logger.info(f"Converted 'value' column to numeric, {result['value'].isna().sum()} NaN values")
        
        # Also convert any other numeric-looking columns
        for col in result.columns:
            if col != 'value' and result[col].dtype == 'object':
                # Check if this looks like a numeric column
                sample = result[col].dropna().iloc[:20] if len(result[col].dropna()) > 20 else result[col].dropna()
                numeric_candidates = [
                    val for val in sample 
                    if isinstance(val, str) and (val.replace('.', '').replace(',', '').replace('-', '').isdigit() 
                                               or (val.replace('.', '').replace(',', '').replace('-', '').isdigit() 
                                                  and val.count('.') <= 1))
                ]
                
                if len(numeric_candidates) > 0.7 * len(sample):  # If >70% look numeric
                    # Clean and convert
                    result[col] = result[col].replace(['', 'N/A', 'n/a', 'na', 'NA', '-', '–'], np.nan)
                    if isinstance(result[col], pd.Series) and result[col].dtype == 'object':
                        result[col] = result[col].astype(str).str.replace(',', '')
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                        logger.info(f"Converted '{col}' column to numeric")
        
        return result
    
    def _standardize_reserve_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize reserve measurement units to consistent formats.
        
        Args:
            df: DataFrame with reserve metrics
            
        Returns:
            DataFrame: DataFrame with standardized units
        """
        # Only applicable if we have 'metric' and 'value' columns
        if 'metric' not in df.columns or 'value' not in df.columns:
            return df
        
        result = df.copy()
        
        # Create a standardized_metric column
        result['standardized_metric'] = result['metric']
        
        # Standardize different variations of metrics
        metric_mapping = {
            # Oil metrics
            'petroleo': 'oil_m3',
            'petróleo': 'oil_m3',
            'oil': 'oil_m3',
            'crude_oil': 'oil_m3',
            'crude': 'oil_m3',
            'oil_bbl': 'oil_bbl',
            
            # Gas metrics
            'gas': 'gas_mm3',
            'natural_gas': 'gas_mm3',
            'gas_bcf': 'gas_bcf',
            
            # Condensate metrics
            'condensado': 'condensate_m3',
            'condensate': 'condensate_m3',
            'condensate_bbl': 'condensate_bbl',
            
            # Gasoline metrics
            'gasolina': 'gasoline_m3',
            'gasoline': 'gasoline_m3',
            'gasoline_bbl': 'gasoline_bbl',
            
            # Total metrics
            'total': 'total_boe',
            'total_boe': 'total_boe',
            'boe': 'total_boe'
        }
        
        # Apply fuzzy mapping using token_set_ratio
        from fuzzywuzzy import fuzz
        
        def map_metric(metric):
            metric_str = str(metric).lower()
            
            # Direct mapping
            if metric_str in metric_mapping:
                return metric_mapping[metric_str]
            
            # Fuzzy mapping
            best_match = None
            best_score = 0
            
            for key, value in metric_mapping.items():
                score = fuzz.token_set_ratio(metric_str, key)
                if score > best_score and score >= 80:  # Use threshold
                    best_score = score
                    best_match = value
            
            return best_match if best_match else metric_str
        
        # Apply the mapping
        result['standardized_metric'] = result['metric'].apply(map_metric)
        
        # Extract unit information if available
        result['unit'] = result['standardized_metric'].str.split('_').str[-1]
        
        # Add conversion to standard units (for plotting consistency)
        result['value_boe'] = None  # Barrel of oil equivalent
        
        # Convert values to BOE based on unit
        mask_oil_m3 = result['standardized_metric'] == 'oil_m3'
        mask_oil_bbl = result['standardized_metric'] == 'oil_bbl'
        mask_gas_mm3 = result['standardized_metric'] == 'gas_mm3'
        mask_gas_bcf = result['standardized_metric'] == 'gas_bcf'
        mask_condensate_m3 = result['standardized_metric'] == 'condensate_m3'
        mask_condensate_bbl = result['standardized_metric'] == 'condensate_bbl'
        
        # Apply conversions
        if mask_oil_m3.any():
            result.loc[mask_oil_m3, 'value_boe'] = result.loc[mask_oil_m3, 'value'] * self.unit_conversions['m3_to_bbl']
        
        if mask_oil_bbl.any():
            result.loc[mask_oil_bbl, 'value_boe'] = result.loc[mask_oil_bbl, 'value']
        
        if mask_gas_mm3.any():
            # Convert gas mm3 to boe (approximately 1 mm3 = 6000 boe)
            result.loc[mask_gas_mm3, 'value_boe'] = result.loc[mask_gas_mm3, 'value'] * 6000
        
        if mask_gas_bcf.any():
            # Convert bcf to mm3, then to boe
            mm3_value = result.loc[mask_gas_bcf, 'value'] * self.unit_conversions['bcf_to_mm3']
            result.loc[mask_gas_bcf, 'value_boe'] = mm3_value * 6000
        
        if mask_condensate_m3.any():
            result.loc[mask_condensate_m3, 'value_boe'] = result.loc[mask_condensate_m3, 'value'] * self.unit_conversions['m3_to_bbl']
        
        if mask_condensate_bbl.any():
            result.loc[mask_condensate_bbl, 'value_boe'] = result.loc[mask_condensate_bbl, 'value']
        
        # For metrics already in BOE
        mask_boe = result['standardized_metric'] == 'total_boe'
        if mask_boe.any():
            result.loc[mask_boe, 'value_boe'] = result.loc[mask_boe, 'value']
        
        return result
    
    def _extract_year_from_filename(self, filename: str) -> int:
        """
        Extract year from filename using multiple strategies.
        
        Args:
            filename: Filename to extract year from
            
        Returns:
            int: Extracted year or None if not found
        """
        # Try common patterns
        # Pattern 1: Split by underscore
        parts = filename.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
        
        # Pattern 2: Split by space
        for token in filename.split():
            if token.isdigit() and len(token) == 4:
                return int(token)
        
        # Pattern 3: Extract any 4-digit sequence that looks like a year
        import re
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', filename)
        if year_matches:
            return int(year_matches[0])
        
        return None
    
    def _extract_scope_from_filename(self, filename: str) -> str:
        """
        Extract scope (EOC or EOL) from filename.
        
        Args:
            filename: Filename to extract scope from
            
        Returns:
            str: 'end_of_concession', 'end_of_life', or None
        """
        filename_lower = filename.lower()
        
        if 'eoc' in filename_lower or 'concession' in filename_lower or 'concesion' in filename_lower:
            return 'end_of_concession'
        elif 'eol' in filename_lower or 'life' in filename_lower or 'vida' in filename_lower:
            return 'end_of_life'
        
        return None
    
    def load_file_map(self) -> List[Dict]:
        """
        Load the manifest file mapping files to their sheets and scopes.
        Updates scope based on filename if not specified.
        
        Returns:
            List[Dict]: List of file entries with path, sheet, scope, and engine
        """
        if not self.manifest_path.exists():
            logger.error(f"Manifest file not found at {self.manifest_path}")
            return []
        
        df = pd.read_csv(self.manifest_path)
        
        # Ensure required columns exist
        required_cols = ['filename', 'sheet_name', 'scope', 'engine']
        for col in required_cols:
            if col not in df.columns:
                if col == 'engine':
                    df['engine'] = 'openpyxl'  # Default engine
                elif col == 'scope':
                    df['scope'] = None  # Will be updated from filename
                elif col == 'sheet_name':
                    df['sheet_name'] = None  # Will be detected later
                else:
                    logger.error(f"Required column '{col}' not found in manifest")
                    return []
        
        # Update scope from filename if not specified
        for index, row in df.iterrows():
            if pd.isna(row['scope']) or row['scope'] == '' or row['scope'] == 'unknown':
                scope = self._extract_scope_from_filename(row['filename'])
                if scope:
                    df.at[index, 'scope'] = scope
        
        # Convert to list of dictionaries
        entries = df.to_dict(orient="records")
        
        # Add full path to each entry
        for entry in entries:
            entry["path"] = self.raw_dir / entry["filename"]
            
            # Skip entries where file doesn't exist
            if not entry["path"].exists():
                logger.warning(f"File not found: {entry['path']}")
                continue
            
            # Determine appropriate engine based on file extension if not specified
            if pd.isna(entry.get('engine')) or entry.get('engine') == '':
                if entry["path"].suffix.lower() == '.xlsx':
                    entry['engine'] = 'openpyxl'
                else:
                    entry['engine'] = 'xlrd'
        
        return entries
    
    def update_sheet_names_in_manifest(self) -> bool:
        """
        Update the sheet names in the manifest to match actual sheets in files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.manifest_path.exists():
            logger.error(f"Manifest file not found at {self.manifest_path}")
            return False
        
        try:
            df = pd.read_csv(self.manifest_path)
            updated_count = 0
            
            for index, row in df.iterrows():
                file_path = self.raw_dir / row['filename']
                if not file_path.exists():
                    continue
                
                # Skip if sheet name is already specified and valid
                if not pd.isna(row['sheet_name']) and row['sheet_name'] != '':
                    continue
                
                # Determine engine
                engine = row.get('engine', None)
                if pd.isna(engine) or engine == '':
                    engine = 'openpyxl' if file_path.suffix.lower() == '.xlsx' else 'xlrd'
                
                try:
                    # Get actual sheet names
                    excel = pd.ExcelFile(file_path, engine=engine)
                    available_sheets = excel.sheet_names
                    
                    if not available_sheets:
                        continue
                    
                    # Find best sheet based on scope
                    scope = row.get('scope', None)
                    best_sheet = self._find_best_sheet_for_scope(available_sheets, scope)
                    
                    # Update the sheet name
                    df.at[index, 'sheet_name'] = best_sheet
                    updated_count += 1
                except Exception as e:
                    logger.error(f"Failed to read sheets from {file_path}: {e}")
            
            # Save updated manifest
            df.to_csv(self.manifest_path, index=False)
            logger.info(f"Updated {updated_count} sheet names in manifest file")
            return True
        except Exception as e:
            logger.error(f"Failed to update sheet names in manifest: {e}")
            return False
    
    def _find_best_sheet_for_scope(self, available_sheets: List[str], scope: str) -> str:
        """
        Find the best sheet for a given scope from available sheets.
        
        Args:
            available_sheets: List of available sheet names
            scope: Scope ('end_of_concession' or 'end_of_life')
            
        Returns:
            str: Best matching sheet name
        """
        if not available_sheets:
            return None
        
        # If only one sheet, use that
        if len(available_sheets) == 1:
            return available_sheets[0]
        
        # Keywords to look for
        eoc_keywords = ['concession', 'eoc', 'conc', 'concesión', 'concesion']
        eol_keywords = ['life', 'eol', 'vida', 'útil', 'util']
        
        if scope == 'end_of_concession':
            keywords = eoc_keywords
        elif scope == 'end_of_life':
            keywords = eol_keywords
        else:
            # If scope not specified, use first sheet
            return available_sheets[0]
        
        # Look for sheets matching keywords
        for sheet in available_sheets:
            sheet_lower = sheet.lower()
            if any(keyword in sheet_lower for keyword in keywords):
                return sheet
        
        # If no match found, use first sheet
        return available_sheets[0]
    
    def process_excel_file(self, file_path: Path, sheet_name: str, scope: str, engine: str) -> pd.DataFrame:
        """
        Process a single Excel file into standardized format.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name to process
            scope: Scope tag ('end_of_concession' or 'end_of_life')
            engine: Excel engine to use ('openpyxl' or 'xlrd')
            
        Returns:
            DataFrame: Processed DataFrame in tidy format
        """
        try:
            # Step 1: Read raw data without assuming header
            try:
                df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine=engine)
            except ValueError as e:
                # Sheet not found, try to use the first sheet
                if "Worksheet named" in str(e) and "not found" in str(e):
                    try:
                        excel = pd.ExcelFile(file_path, engine=engine)
                        available_sheets = excel.sheet_names
                        
                        if not available_sheets:
                            raise ValueError(f"No sheets found in {file_path}")
                        
                        first_sheet = available_sheets[0]
                        logger.warning(f"Sheet '{sheet_name}' not found in {file_path.name}, using '{first_sheet}' instead")
                        sheet_name = first_sheet
                        df_raw = pd.read_excel(file_path, sheet_name=first_sheet, header=None, engine=engine)
                    except Exception as inner_e:
                        raise ValueError(f"Failed to use alternative sheet: {inner_e}")
                else:
                    raise e
            
            # Step 2: Detect header row
            header_row = self._detect_header_row(df_raw)
            
            # Step 3: Read data with correct header
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, engine=engine)
            
            # Step 4: Normalize column names
            df = self._normalize_columns(df)
            
            # Step 5: Map to standard column names and split combined fields
            df, column_mapping = self._map_standard_columns(df)
            df = self._split_combined_fields(df)
            
            # Step 6: Convert to tidy format
            df = self._convert_to_tidy_format(df)
            
            # Step 7: Clean numeric values
            df = self._clean_numeric_values(df)
            
            # Step 8: Add source file info
            df["source_file"] = file_path.name
            df["scope"] = scope
            df["year"] = self._extract_year_from_filename(file_path.name)
            
            # Step 9: Standardize units
            df = self._standardize_reserve_units(df)
            
            # Get standardized scope abbreviation
            scope_abbr = "eoc" if scope == "end_of_concession" else "eol"
            
            # Create standardized metadata
            meta = {
                "filename": file_path.name,
                "sheet": sheet_name,
                "header_row": header_row,
                "column_mapping": column_mapping,
                "columns": list(df.columns),
                "row_count": len(df),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "scope": scope,
                "scope_abbr": scope_abbr,
                "year": df["year"].iloc[0] if len(df) > 0 else None
            }
            
            return df, meta
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, meta: Dict, 
                          dataset: str = "reserves") -> Tuple[Path, Path]:
        """
        Save processed data and metadata with standardized naming.
        
        Args:
            df: Processed DataFrame
            meta: Metadata dictionary
            dataset: Dataset name prefix
            
        Returns:
            Tuple[Path, Path]: Paths to saved CSV and metadata files
        """
        # Extract year and scope for standardized naming
        year = meta.get('year')
        scope_abbr = meta.get('scope_abbr')
        
        if not year or not scope_abbr:
            logger.error("Cannot save data without year and scope")
            return None, None
        
        # Create standardized base name
        standard_name = f"{dataset}_{year}_{scope_abbr}"
        
        # Save CSV
        csv_path = self.processed_dir / f"{standard_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save metadata
        meta_path = self.metadata_dir / f"{standard_name}.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(df)} rows to {csv_path}")
        logger.info(f"Saved metadata to {meta_path}")
        
        return csv_path, meta_path
    
    def validate_output(self, df: pd.DataFrame) -> Dict:
        """
        Validate processed output data for completeness and consistency.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dict: Validation results
        """
        results = {
            "valid": True,
            "issues": [],
            "stats": {}
        }
        
        # Check required columns
        required_columns = ['value', 'metric', 'standardized_metric', 'year', 'scope']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results["valid"] = False
            results["issues"].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty DataFrames
        if len(df) == 0:
            results["valid"] = False
            results["issues"].append("DataFrame is empty")
        
        # Check for missing values in critical columns
        if 'value' in df.columns:
            null_values = df['value'].isna().sum()
            null_percentage = (null_values / len(df)) * 100 if len(df) > 0 else 0
            
            results["stats"]["null_values"] = null_values
            results["stats"]["null_percentage"] = null_percentage
            
            if null_percentage > 50:
                results["valid"] = False
                results["issues"].append(f"High percentage of null values: {null_percentage:.2f}%")
        
        # Calculate basic statistics
        if 'value' in df.columns and len(df) > 0:
            results["stats"]["min_value"] = df['value'].min()
            results["stats"]["max_value"] = df['value'].max()
            results["stats"]["mean_value"] = df['value'].mean()
            
            # Check for suspicious values (e.g., all zeros)
            zero_percentage = (df['value'] == 0).sum() / len(df) * 100
            results["stats"]["zero_percentage"] = zero_percentage
            
            if zero_percentage > 90:
                results["valid"] = False
                results["issues"].append(f"High percentage of zero values: {zero_percentage:.2f}%")
        
        # Check metric distribution
        if 'standardized_metric' in df.columns:
            metric_counts = df['standardized_metric'].value_counts().to_dict()
            results["stats"]["metric_distribution"] = metric_counts
        
        return results
    
    def create_fixed_manifest(self) -> bool:
        """
        Create or fix the manifest file by scanning all Excel files in raw directory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find all Excel files
            excel_files = list(self.raw_dir.glob("*.xls")) + list(self.raw_dir.glob("*.xlsx"))
            
            if not excel_files:
                logger.warning(f"No Excel files found in {self.raw_dir}")
                return False
            
            # Create manifest data
            manifest_data = []
            
            for file_path in excel_files:
                # Extract information from filename
                year = self._extract_year_from_filename(file_path.name)
                scope = self._extract_scope_from_filename(file_path.name)
                
                # Determine engine
                engine = 'openpyxl' if file_path.suffix.lower() == '.xlsx' else 'xlrd'
                
                # Try to read sheet names
                try:
                    excel = pd.ExcelFile(file_path, engine=engine)
                    sheets = excel.sheet_names
                except Exception as e:
                    logger.error(f"Failed to read sheets from {file_path}: {e}")
                    # Try alternate engine
                    alt_engine = 'xlrd' if engine == 'openpyxl' else 'openpyxl'
                    try:
                        excel = pd.ExcelFile(file_path, engine=alt_engine)
                        sheets = excel.sheet_names
                        engine = alt_engine
                    except Exception as e2:
                        logger.error(f"Failed with both engines: {e2}")
                        sheets = []
                
                # Find best sheet for scope
                best_sheet = self._find_best_sheet_for_scope(sheets, scope)
                
                manifest_data.append({
                    'filename': file_path.name,
                    'year': year,
                    'sheet_name': best_sheet,
                    'scope': scope,
                    'engine': engine
                })
            
            # Create DataFrame and save
            manifest_df = pd.DataFrame(manifest_data)
            manifest_df.to_csv(self.manifest_path, index=False)
            
            logger.info(f"Created manifest with {len(manifest_df)} entries at {self.manifest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create manifest: {e}")
            return False
    
    def run_transform(self) -> Dict:
        """
        Run the entire transformation process for all files in the manifest.
        
        Returns:
            Dict: Summary of transformation results
        """
        logger.info(f"=== Starting Reserve Transformation with Enhanced Processing ===")
        logger.info(f"Raw data location: {self.raw_dir}")
        logger.info(f"Output location: {self.processed_dir}")
        logger.info(f"Metadata location: {self.metadata_dir}")
        logger.info(f"Backup location: {self.backup_dir}")
        
        results = {
            "processed_files": 0,
            "failed_files": 0,
            "output_files": [],
            "errors": []
        }
        
        # Check if manifest exists, otherwise create it
        if not self.manifest_path.exists():
            logger.info("Manifest file not found, creating one...")
            self.create_fixed_manifest()
        else:
            # Update sheet names in manifest if needed
            self.update_sheet_names_in_manifest()
        
        # Backup output and metadata directories
        self._backup_directory(self.processed_dir, "*.csv")
        self._backup_directory(self.metadata_dir, "*.json")
        
        # Clean directories
        self._clean_directory(self.processed_dir, "*.csv")
        self._clean_directory(self.metadata_dir, "*.json")
        
        # Load file map
        file_map = self.load_file_map()
        logger.info(f"Processing {len(file_map)} files...")
        
        # Process each file
        for entry in file_map:
            try:
                file_path = entry.get("path")
                sheet_name = entry.get("sheet_name")
                scope = entry.get("scope")
                engine = entry.get("engine")
                
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                logger.info(f"Processing {file_path.name} [{sheet_name}]...")
                df, meta = self.process_excel_file(file_path, sheet_name, scope, engine)
                
                # Validate output
                validation = self.validate_output(df)
                if not validation["valid"]:
                    logger.warning(f"Validation issues for {file_path.name}: {validation['issues']}")
                    meta["validation"] = validation
                
                # Save processed data
                csv_path, meta_path = self.save_processed_data(df, meta)
                
                if csv_path:
                    results["processed_files"] += 1
                    results["output_files"].append(str(csv_path))
                
            except Exception as e:
                logger.error(f"Failed to process {entry.get('path')}: {e}")
                results["failed_files"] += 1
                results["errors"].append({
                    "file": str(entry.get("path")),
                    "error": str(e)
                })
        
        logger.info(f"Transformation completed: {results['processed_files']} files processed, "
                   f"{results['failed_files']} files failed")
        
        return results