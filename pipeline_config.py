from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from typing import Any, Dict
import os

def get_folder_hash(folder_path: Path) -> str:
    """
    Generate a hash of folder contents based only on filenames and sizes.
    """
    content_list = []
    for entry in sorted(folder_path.iterdir()):
        if entry.is_file():
            # Only include filename and size, not modification time
            content_list.append((entry.name, entry.stat().st_size))
    
    # Convert to stable string representation and hash
    content_str = json.dumps(content_list, sort_keys=True)
    hash_obj = hashlib.sha256(content_str.encode())
    hash_value = hash_obj.hexdigest()[:8]
    
    print(f"Hashing folder {folder_path.name}:")
    print(f"Content list: {content_list[:3]}... ({len(content_list)} files)")
    print(f"Hash: {hash_value}")
    
    return hash_value

@dataclass
class PipelineConfig:
    """Base configuration for image processing pipeline"""
    target_size: int = 512
    file_pattern: str = "*.jpg"
    prefix_separator: str = "_"
    jpeg_quality: int = 95
    zoom_factor: float = 2.5
    face_tolerance: float = 0.6
    min_cluster_size: int = 3

    def get_hash(self, stage_name: str, input_path: Path) -> str:
        """Generate a short hash based on config values, stage name, and input folder"""
        # Convert config to dictionary, excluding None values
        config_dict = {k: v for k, v in asdict(self).items() if v is not None}
        
        # Add stage name and input folder hash
        config_dict['stage'] = stage_name
        config_dict['input_hash'] = get_folder_hash(input_path)
        
        # Convert to stable string representation
        config_str = json.dumps(config_dict, sort_keys=True)
        
        # Generate hash
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()[:8]

def make_stage_folder(base_path: Path, stage_name: str, config: PipelineConfig, input_path: Path | None = None) -> Path:
    """
    Create and return path for pipeline stage with config hash.
    If input_path is None, uses base_path as input path for hashing.
    """
    # Create a "processed" directory next to the input directory
    processed_path = base_path.parent / "processed"
    processed_path.mkdir(exist_ok=True)
    
    # Get input folder for hashing
    input_folder = input_path or base_path
    
    # Get hash based on config, stage name, and input folder contents
    input_hash = get_folder_hash(input_folder)
    config_dict = {k: v for k, v in asdict(config).items() if v is not None}
    config_dict['stage'] = stage_name
    config_dict['input_hash'] = input_hash
    
    # Generate folder name hash
    config_str = json.dumps(config_dict, sort_keys=True)
    folder_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    folder_name = f"{stage_name}_{folder_hash}"
    folder_path = processed_path / folder_name
    
    # Check if folder exists and contains files
    if folder_path.exists() and any(folder_path.iterdir()):
        # Verify input folder hash matches what we expect
        current_hash = get_folder_hash(input_folder)
        if current_hash == input_hash:  # Compare actual hashes directly
            print(f"\nUsing cached results from {folder_name}")
            return folder_path
        print(f"\nInput changed, invalidating cache for {folder_name}")
    else:
        print(f"\nCreating new output folder {folder_name}")
    
    folder_path.mkdir(exist_ok=True)
    return folder_path