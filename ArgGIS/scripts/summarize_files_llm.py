import os
import re
import json
from pathlib import Path
from textwrap import dedent
from typing import Iterator

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.workflow import Workflow
from agno.storage.sqlite import SqliteStorage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it or create a .env file.")

# Get model ID from environment or use default
MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "gpt-4o")

# Check if outputs directory exists
outputs_dir = Path("outputs")
if not outputs_dir.exists():
    raise FileNotFoundError("outputs directory not found. Run files.py first.")

# Check if DATA_OVERVIEW.md exists
data_overview_path = outputs_dir / "DATA_OVERVIEW.md"
if not data_overview_path.exists():
    raise FileNotFoundError("DATA_OVERVIEW.md not found. Run files.py first.")

# Create reports directory
reports_dir = outputs_dir / "dataset_reports"
if not reports_dir.exists():
    reports_dir.mkdir(parents=True, exist_ok=True)

# Path for consolidated report
enhanced_overview_md = str(reports_dir / "ENHANCED_DATA_OVERVIEW.md")
enhanced_overview_json = str(reports_dir / "enhanced_data_summaries.json")

def extract_datasets_from_markdown(md_file_path):
    """
    Extract each dataset description from the markdown file.
    Each dataset starts with a ### header and ends before the next ### or ---
    """
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by dataset sections (starts with ### and ends with --- or another ###)
    dataset_blocks = re.split(r'(?=###\s+)', content)
    
    # Process each block
    datasets = []
    for block in dataset_blocks:
        # Skip if it's not a dataset block or just metadata
        if not block.startswith('###') or 'Failed Files' in block:
            continue
        
        # Clean up the block
        block = block.strip()
        
        # Extract dataset details
        match = re.match(r'###\s+(.*?)\s+\(in\s+(.*?)\)', block)
        if match:
            file_name = match.group(1)
            folder_name = match.group(2)
            
            # Create dataset object
            dataset = {
                'file_name': file_name,
                'folder': folder_name,
                'content': block,
                'type': 'Unknown'
            }
            
            # Try to identify the dataset type
            if "**Type**: Shapefile" in block:
                dataset['type'] = 'Shapefile'
            elif "**Type**: CSV" in block:
                dataset['type'] = 'CSV'
            elif "**Type**: Excel" in block:
                dataset['type'] = 'Excel'
            elif "**Type**: JSON" in block:
                dataset['type'] = 'JSON'
            
            datasets.append(dataset)
    
    return datasets

# Common context about Argentina's oil and gas industry to add to all prompts
argentina_context = """
Context about Argentina's oil/gas industry:
- Argentina is one of South America's largest producers of oil and natural gas
- The Vaca Muerta formation in Neuquén Basin is one of the world's largest shale oil/gas reserves
- Key producing provinces include Neuquén, Mendoza, Chubut, Santa Cruz, and Río Negro
- YPF (state-owned) is the largest oil company, but many international and private companies operate there
- The country has both conventional and unconventional (shale/tight) resources
- Data organization typically follows administrative divisions (provinces, concessions, exploitation lots)
"""

def get_prompt_for_dataset(dataset):
    """Create a prompt based on the dataset type"""
    if dataset['type'] == 'Shapefile':
        return f"""I have a GIS shapefile with the following details:

{dataset['content']}

{argentina_context}

Please provide a comprehensive summary that explains:
1. What this data represents in the context of Argentina's oil/gas industry
2. The meaning of key fields/columns (including any Spanish-language terms if present)
3. The geographical scope and significance of this data
4. How this data might be used in analysis (e.g., production forecasting, reserve estimation, investment planning)
5. Any limitations or special considerations when using this data

Keep your response concise (2-3 paragraphs maximum) but informative for someone who needs to understand this dataset quickly.
"""
    elif dataset['type'] == 'CSV':
        return f"""I have a CSV file with the following details:

{dataset['content']}

{argentina_context}

Please provide a comprehensive summary that explains:
1. What this data represents in the context of Argentina's oil/gas industry
2. The meaning of key fields/columns (including any Spanish-language terms if present)
3. The temporal scope and significance of this data
4. How this data might be used in analysis (e.g., production trends, drilling activity, investment analysis)
5. Any limitations or special considerations when using this data

Keep your response concise (2-3 paragraphs maximum) but informative for someone who needs to understand this dataset quickly.
"""
    elif dataset['type'] == 'Excel':
        return f"""I have an Excel file with the following details:

{dataset['content']}

{argentina_context}

Please provide a comprehensive summary that explains:
1. What this data represents in the context of Argentina's oil/gas industry
2. The meaning of key sheets and fields/columns (including any Spanish-language terms if present)
3. The temporal and geographical scope of this data
4. How this data might be used in analysis of Argentina's oil/gas resources
5. Any limitations or special considerations when using this data

Keep your response concise (2-3 paragraphs maximum) but informative for someone who needs to understand this dataset quickly.
"""
    else:
        return f"""I have a data file with the following details:

{dataset['content']}

{argentina_context}

Please provide a comprehensive summary that explains:
1. What this data represents in the context of Argentina's oil/gas industry
2. The meaning of key fields/columns (including any Spanish-language terms if present)
3. The significance of this data
4. How this data might be used in analysis of Argentina's oil/gas resources
5. Any limitations or special considerations when using this data

Keep your response concise (2-3 paragraphs maximum) but informative for someone who needs to understand this dataset quickly.
"""

class DatasetSummarizer(Workflow):
    """Workflow for summarizing a dataset and explaining its significance."""
    
    description = "Summarize a dataset and provide context about its significance in Argentina's oil/gas industry."
    
    dataset_agent: Agent = Agent(
        model=OpenAIChat(id=MODEL_ID),
        name="Dataset Summarizer",
        description=dedent("""\
        You are a data scientist specializing in the oil and gas industry, with particular expertise
        in Argentina's energy sector. Your job is to examine GIS and tabular datasets and provide
        clear, insightful summaries that explain what the data represents, its significance,
        and how it might be used in analysis.
        """),
        markdown=True,
    )
    
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.report_file = str(reports_dir / f"{self.dataset['folder']}_{self.dataset['file_name']}.md")
        self.dataset_agent.save_response_to_file = self.report_file
        
    def run(self) -> Iterator[RunResponse]:
        """Run the workflow, yielding RunResponse objects as required by Agno."""
        prompt = get_prompt_for_dataset(self.dataset)
        
        # Call the agent's run method which returns a RunResponse or Iterator[RunResponse]
        response = self.dataset_agent.run(prompt, stream=False)
        
        # If response is an iterator, yield from it
        if hasattr(response, '__iter__') and not isinstance(response, (str, dict)):
            yield from response
        # If it's a single RunResponse, yield it
        elif isinstance(response, RunResponse):
            yield response
        # Otherwise create a new RunResponse
        else:
            yield RunResponse(run_id=self.run_id, content=str(response))

def generate_enhanced_overview(datasets, output_file):
    """
    Generate a new markdown file with enhanced summaries
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Argentina GIS Data Overview\n\n")
        f.write("This document provides enhanced summaries of all GIS and tabular datasets in the project.\n\n")
        f.write(f"**Total Datasets**: {len(datasets)}\n\n")
        
        # Group datasets by type
        by_type = {}
        for ds in datasets:
            ds_type = ds['type']
            if ds_type not in by_type:
                by_type[ds_type] = []
            by_type[ds_type].append(ds)
        
        # Create a table of contents
        f.write("## Table of Contents\n\n")
        
        for ds_type, ds_list in by_type.items():
            f.write(f"### {ds_type} Files ({len(ds_list)})\n\n")
            for ds in ds_list:
                # Create a safe anchor link by slugifying the filename and folder
                safe_filename = re.sub(r'[^\w\s-]', '', ds['file_name'].lower()).strip().replace(' ', '-')
                safe_folder = re.sub(r'[^\w\s-]', '', ds['folder'].lower()).strip().replace(' ', '-')
                f.write(f"- [{ds['file_name']} (in {ds['folder']})](#user-content-{safe_filename}-in-{safe_folder})\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # Write the detailed summaries
        for ds_type, ds_list in by_type.items():
            f.write(f"## {ds_type} Files\n\n")
            
            for ds in ds_list:
                # Create safe anchor targets
                safe_filename = re.sub(r'[^\w\s-]', '', ds['file_name'].lower()).strip().replace(' ', '-')
                safe_folder = re.sub(r'[^\w\s-]', '', ds['folder'].lower()).strip().replace(' ', '-')
                f.write(f"### {ds['file_name']} (in {ds['folder']})\n\n")
                f.write("#### Enhanced Summary\n\n")
                f.write(f"{ds['enhanced_summary']}\n\n")
                f.write("#### Original Metadata\n\n")
                
                # Extract and format original metadata
                original_content = ds['original_content']
                header_line = original_content.split('\n')[0]
                metadata = original_content.replace(header_line, "").strip()
                f.write(f"{metadata}\n\n")
                f.write("---\n\n")

def process_datasets(datasets):
    """Process datasets using the workflow approach"""
    # Path for progress file
    progress_file = outputs_dir / "enhanced_data_progress.json"
    
    # Check if we have existing progress
    enhanced_datasets = []
    processed_file_names = set()
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                enhanced_datasets = json.load(f)
                
            # Track processed datasets
            for ds in enhanced_datasets:
                processed_file_names.add(f"{ds['folder']}_{ds['file_name']}")
                
            print(f"Loaded {len(enhanced_datasets)} previously processed datasets")
        except Exception as e:
            print(f"Error loading previous progress: {e}")
            print("Starting from scratch...")
            enhanced_datasets = []
    
    # Create storage for each dataset individually to avoid mode conflicts
    
    # Process each dataset sequentially
    for i, dataset in enumerate(datasets):
        # Skip if already processed
        ds_key = f"{dataset['folder']}_{dataset['file_name']}"
        if ds_key in processed_file_names:
            print(f"Skipping already processed dataset: {dataset['file_name']} ({dataset['type']})")
            continue
        
        print(f"Processing dataset {i+1}/{len(datasets)}: {dataset['file_name']} ({dataset['type']})")
        
        try:
            # Set up unique session ID
            session_id = f"dataset-summary-{ds_key}"
            
            # Create fresh storage for each workflow to avoid mode conflicts
            storage = SqliteStorage(
                table_name="dataset_summary_workflows",
                db_file=str(outputs_dir / "agno_workflows.db")
            )
            
            # Create the workflow
            workflow = DatasetSummarizer(
                dataset=dataset,
                session_id=session_id,
                storage=storage
            )
            
            # Run the workflow and collect results
            responses = list(workflow.run())
            
            # Get content from the file that was saved
            report_path = Path(reports_dir / f"{dataset['folder']}_{dataset['file_name']}.md")
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create result dictionary
                result = {
                    'file_name': dataset['file_name'],
                    'folder': dataset['folder'],
                    'type': dataset['type'],
                    'original_content': dataset['content'],
                    'enhanced_summary': content
                }
                
                enhanced_datasets.append(result)
                processed_file_names.add(ds_key)
                
                # Save progress
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_datasets, f, indent=2)
                
                print(f"✓ Successfully processed: {dataset['file_name']} ({dataset['type']})")
            else:
                print(f"✗ Failed to process {dataset['file_name']}: Output file not found")
                
        except Exception as e:
            print(f"✗ Failed to process {dataset['file_name']}: {e}")
    
    return enhanced_datasets

def main():
    print(f"Starting dataset summary enhancement using model: {MODEL_ID}...")
    
    # Extract datasets from the markdown file
    datasets = extract_datasets_from_markdown(data_overview_path)
    print(f"Found {len(datasets)} datasets in DATA_OVERVIEW.md")
    
    # Process datasets using the workflow approach
    enhanced_datasets = process_datasets(datasets)
    
    # Generate enhanced overview
    generate_enhanced_overview(enhanced_datasets, enhanced_overview_md)
    print(f"Enhanced overview generated at {enhanced_overview_md}")
    
    # Save all enhanced summaries as JSON
    with open(enhanced_overview_json, 'w', encoding='utf-8') as f:
        json.dump(enhanced_datasets, f, indent=2)
    print(f"Enhanced summaries saved as JSON at {enhanced_overview_json}")
    
    print(f"Successfully processed {len(enhanced_datasets)} out of {len(datasets)} datasets.")

if __name__ == "__main__":
    main()
