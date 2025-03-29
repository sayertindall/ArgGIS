#!/usr/bin/env python3
"""
Script to download all datasets from Argentina's Energy Data Portal
Organizes resources by dataset into separate directories
"""

import os
import requests
import json
from tqdm import tqdm
import time
import re
import logging
from urllib.parse import urlparse, unquote

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_log.txt"),
        logging.StreamHandler()
    ]
)

# Base directory for all downloads
BASE_DIR = "argentina_energy_data"
os.makedirs(BASE_DIR, exist_ok=True)

# Base API URL
API_URL = "http://datos.energia.gob.ar/api/3/action/package_show?id="

# List of dataset IDs to download
DATASET_IDS = [
    # # Hydrocarbon Exploration - 3D Seismic Surveys
    # "exploracion-hidrocarburos-sismicas-3d",
    
    # # Hydrocarbon Installations - Pipelines
    # "instalaciones-hidrocarburos-ductos-res--319-93",
    
    # # Hydrocarbon Exploration - 2D Seismic Lines
    # "exploracion-hidrocarburos-lineas-sismicas-2d",
    
    # # Vaca Muerta Development
    # "desarrollo-de-vaca-muerta-impacto-economico-agregado-y-sectorial",
    
    # # Oil Fields by Average Depth
    # "produccion-hidrocarburos-yacimientos-segun-profundidad-promedio",
    
    # # Declared Venting Points
    # "produccion-hidrocarburos-puntos-de-venteo-declarados",
    
    # # Exploitation Lots
    # "produccion-hidrocarburos-lotes-de-explotacion",
    
    # # Exploration Permits
    # "exploracion-hidrocarburos-permisos-de-exploracion",
    
    # # Exploitation Concessions
    # "produccion-hidrocarburos-concesiones-de-explotacion",
    
    # # Oil and Gas Well Drilling
    # "perforacion-de-pozos-de-petroleo-y-gas",
    
    # # Well Fracturing Data
    # "datos-de-fractura-de-pozos-adjunto-iv",
    
    # # Hydrocarbon Fields
    # "produccion-hidrocarburos-yacimientos",
    
    # # Oil and Gas Production
    # "produccion-de-petroleo-y-gas-tablas-dinamicas",
    
    # Oil and Gas Reserves
    "reservas-de-petroleo-y-gas",
    
    # Oil and Gas Production by Well
    "produccion-de-petroleo-y-gas-por-pozo"
]

def clean_filename(filename):
    """Clean filename to be valid on all operating systems"""
    # Replace invalid characters
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Decode URL encoded characters
    filename = unquote(filename)
    # Limit length
    if len(filename) > 150:
        name, ext = os.path.splitext(filename)
        filename = name[:145] + ext
    return filename

def download_file(url, filepath, retries=3, delay=2):
    """Download a file with progress indicator and retries"""
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Get file size for progress bar (if available)
            total_size = int(response.headers.get('content-length', 0))
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Set up progress bar
            progress_bar = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=os.path.basename(filepath)
            )
            
            # Download file in chunks
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            
            # Verify file was downloaded (has size > 0)
            if os.path.getsize(filepath) > 0:
                return True
            else:
                logging.warning(f"Downloaded file is empty: {filepath}")
                
                # Delete empty file
                os.remove(filepath)
                
                if attempt < retries - 1:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                continue
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading {url}: {e}")
            
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            continue
            
        except Exception as e:
            logging.error(f"Unexpected error downloading {url}: {e}")
            return False
            
    return False

def get_dataset_metadata(dataset_id):
    """Get dataset metadata from the API"""
    try:
        url = f"{API_URL}{dataset_id}"
        logging.info(f"Fetching metadata for dataset: {dataset_id}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['success']:
            return data['result']
        else:
            logging.error(f"API Error for dataset {dataset_id}: {data.get('error', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching metadata for dataset {dataset_id}: {e}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding API response for dataset {dataset_id}")
        return None

def extract_filename_from_url(url):
    """Extract filename from URL"""
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)
    return os.path.basename(path)

def download_dataset(dataset_id):
    """Download all resources for a dataset"""
    # Get dataset metadata
    dataset = get_dataset_metadata(dataset_id)
    
    if not dataset:
        logging.error(f"Failed to get metadata for dataset: {dataset_id}")
        return
    
    # Create directory for dataset
    dataset_title = dataset.get('title', dataset_id)
    dataset_dir = os.path.join(BASE_DIR, clean_filename(dataset_id))
    os.makedirs(dataset_dir, exist_ok=True)
    
    logging.info(f"Processing dataset: {dataset_title} ({dataset_id})")
    logging.info(f"Resources count: {dataset.get('num_resources', 0)}")
    
    # Save dataset metadata
    metadata_path = os.path.join(dataset_dir, "_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Process each resource
    resources = dataset.get('resources', [])
    for index, resource in enumerate(resources):
        resource_id = resource.get('id')
        resource_name = resource.get('name', 'Unknown')
        resource_format = resource.get('format', 'Unknown')
        resource_url = resource.get('url')
        
        if not resource_url:
            logging.warning(f"Skipping resource '{resource_name}' (ID: {resource_id}) - No URL found")
            continue
        
        # Determine filename
        if resource_url.endswith('/'):
            resource_url = resource_url[:-1]
            
        filename = extract_filename_from_url(resource_url)
        
        # If filename is not valid or empty, use resource name with format
        if not filename or filename == "download":
            clean_name = clean_filename(resource_name)
            filename = f"{clean_name}.{resource_format.lower()}"
        
        # Full path for saving the file
        file_path = os.path.join(dataset_dir, clean_filename(filename))
        
        # Skip if file exists and is not empty
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logging.info(f"Skipping {filename} (already downloaded)")
            continue
        
        # Download the file
        logging.info(f"Downloading resource {index+1}/{len(resources)}: '{resource_name}' ({resource_format})")
        success = download_file(resource_url, file_path)
        
        if success:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            logging.info(f"Downloaded {filename} ({file_size:.2f} MB)")
        else:
            logging.error(f"Failed to download {filename}")
        
        # Add a small delay between downloads to avoid overloading the server
        time.sleep(1)
    
    logging.info(f"Completed dataset: {dataset_title}")

def main():
    """Main function"""
    start_time = time.time()
    logging.info("Starting download of ALL datasets")
    
    for dataset_id in DATASET_IDS:
        try:
            download_dataset(dataset_id)
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_id}: {e}")
        
        # Add a delay between datasets
        time.sleep(2)
    
    elapsed_time = time.time() - start_time
    logging.info(f"All downloads completed in {elapsed_time:.2f} seconds!")
    logging.info(f"Files saved to: {os.path.abspath(BASE_DIR)}")

if __name__ == "__main__":
    main()