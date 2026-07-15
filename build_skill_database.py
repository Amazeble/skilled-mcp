#!/usr/bin/env python3
"""Script to create skill database by indexing all SKILL.md files with turbovec.

This script searches for all SKILL.md files in ./agent directory,
embeds them using turbovec, and stores them in skill_database.tq
with 3 columns: name, metadata, file_dir.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from turbovec import TurboQuantIndex

# Add backend src to path
backend_src = Path(__file__).parent / "packages" / "backend" / "src"
sys.path.insert(0, str(backend_src))

from claude_skills_mcp_backend.skill_loader import load_from_local, parse_skill_md

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_skill_files(base_dir: Path) -> list[Path]:
    """Find all SKILL.md files in the given directory.
    
    Parameters
    ----------
    base_dir : Path
        Base directory to search in.
    
    Returns
    -------
    list[Path]
        List of paths to SKILL.md files.
    """
    # Look for agent directory first
    agent_dir = base_dir / "agent"
    if not agent_dir.exists():
        logger.warning(f"Agent directory not found: {agent_dir}")
        # Try searching in the base directory itself
        agent_dir = base_dir
    
    skill_files = list(agent_dir.rglob("SKILL.md"))
    logger.info(f"Found {len(skill_files)} SKILL.md files in {agent_dir}")
    return skill_files


def create_skill_database(
    output_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    bit_width: int = 4
) -> None:
    """Create skill database with turbovec embeddings.
    
    Parameters
    ----------
    output_path : Path
        Path to output .tq file.
    model_name : str, optional
        Sentence transformer model name, by default "all-MiniLM-L6-v2".
    bit_width : int, optional
        Turbovec quantization bit width, by default 4.
    """
    base_dir = Path(__file__).parent
    
    # Find all SKILL.md files
    skill_files = find_skill_files(base_dir)
    
    if not skill_files:
        logger.error("No SKILL.md files found. Creating empty database.")
        # Create empty database
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write empty file with header info
        with open(output_path, "w") as f:
            f.write("# Empty skill database\n")
        return
    
    # Load all skills
    skills = []
    for skill_file in skill_files:
        try:
            content = skill_file.read_text(encoding="utf-8")
            skill = parse_skill_md(content, str(skill_file))
            if skill:
                # Set base_dir to the skill's directory
                skill.base_dir = str(skill_file.parent)
                skills.append(skill)
                logger.info(f"Loaded skill: {skill.name} from {skill_file}")
        except Exception as e:
            logger.error(f"Error loading {skill_file}: {e}")
            continue
    
    if not skills:
        logger.error("No skills could be loaded.")
        return
    
    logger.info(f"Loaded {len(skills)} skills total")
    
    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    descriptions = [skill.description for skill in skills]
    embeddings = model.encode(descriptions, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dim}")
    
    # Create turbovec index
    logger.info(f"Creating turbovec index (bit_width={bit_width})...")
    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(embeddings)
    
    # Prepare data for storage
    # Each row: name, metadata (JSON), file_dir
    rows = []
    for skill in skills:
        name = skill.name
        file_dir = skill.base_dir
        metadata = {
            "description": skill.description,
            "source": skill.source,
            "documents": list(skill.documents.keys()) if skill.documents else [],
        }
        metadata_json = json.dumps(metadata)
        rows.append((name, metadata_json, file_dir))
    
    # Save to .tq file
    # Format: First line is header, then tab-separated values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(rows)} skills to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("name\tmetadata\tfile_dir\n")
        
        # Write data rows
        for name, metadata_json, file_dir in rows:
            # Escape tabs and newlines in metadata
            metadata_json = metadata_json.replace("\t", " ").replace("\n", " ")
            file_dir = file_dir.replace("\t", " ").replace("\n", " ")
            f.write(f"{name}\t{metadata_json}\t{file_dir}\n")
    
    # Also save the turbovec index separately
    index_path = output_path.with_suffix(".index.tq")
    logger.info(f"Saving turbovec index to {index_path}...")
    
    # Save index and skill metadata for later retrieval
    index_data = {
        "skills": [skill.to_dict() for skill in skills],
        "model_name": model_name,
        "bit_width": bit_width,
        "dim": dim,
    }
    
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Successfully created skill database with {len(skills)} skills")
    logger.info(f"Database file: {output_path}")
    logger.info(f"Index file: {index_path}")


if __name__ == "__main__":
    output_file = Path(__file__).parent / "skill_database.tq"
    
    # Allow custom output path via command line
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])
    
    create_skill_database(output_file)
    print(f"\nSkill database created successfully at: {output_file}")
