# pipeline/transform/reserves.py
import logging
from pathlib import Path
from ArgGIS.pipeline.config import ROOT_DIR, DATA_RAW, DATA_PROCESSED, METADATA_DIR, TRANSFORM_CONFIG
from ArgGIS.pipeline.utils.file_utils import (
    backup_directory,
    clean_directory,
    ensure_directory
)

# Import the new processor class
from ArgGIS.pipeline.transform.reserves.reserve_processor import ReserveDataProcessor

# Configure logging
logger = logging.getLogger("transform.reserves")

# Get specific transform config
RESERVES_PATH = DATA_RAW / "reserves"
MANIFEST_PATH = RESERVES_PATH / "manifest.csv"
OUT_DIR = DATA_PROCESSED / "reserves"
META_DIR = METADATA_DIR / "reserves"
BACKUP_DIR = ROOT_DIR / "backups"

def run_reserve_transform():
    """
    Run the reserve transformation process using the enhanced ReserveDataProcessor.
    """
    logger.info("Starting reserve transformation with enhanced processor")
    
    # Create processor instance
    processor = ReserveDataProcessor(
        raw_dir=RESERVES_PATH,
        processed_dir=OUT_DIR,
        metadata_dir=META_DIR,
        backup_dir=BACKUP_DIR,
        manifest_path=MANIFEST_PATH
    )
    
    # Check if manifest exists, create it if needed
    if not MANIFEST_PATH.exists():
        logger.info("Manifest file not found, creating one...")
        processor.create_fixed_manifest()
    
    # Run the transformation
    results = processor.run_transform()
    
    # Log results
    logger.info(f"Reserve transformation completed: {results['processed_files']} files processed")
    
    if results['failed_files'] > 0:
        logger.warning(f"{results['failed_files']} files failed processing")
        for error in results['errors']:
            logger.error(f"Error processing {error['file']}: {error['error']}")
    
    logger.info(f"Output files are available in: {OUT_DIR}")
    logger.info(f"Metadata files are available in: {META_DIR}")
    
    return results


if __name__ == "__main__":
    # Ensure directories exist
    ensure_directory(OUT_DIR)
    ensure_directory(META_DIR)
    ensure_directory(BACKUP_DIR)
    
    # Run the transform
    run_reserve_transform()