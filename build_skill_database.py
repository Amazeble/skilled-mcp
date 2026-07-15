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
from typing import List

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


def view_database(database_path: Path) -> None:
    """View the contents of a skill database file.
    
    Parameters
    ----------
    database_path : Path
        Path to the .tq database file.
    """
    if not database_path.exists():
        print(f"Error: Database file not found: {database_path}")
        return
    
    logger.info(f"Reading database from: {database_path}")
    
    skills = []
    with open(database_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        print("Database is empty.")
        return
    
    # Skip header line if present
    start_idx = 0
    if lines[0].strip().startswith("name\t"):
        start_idx = 1
        logger.info("Header found, skipping...")
    
    for i, line in enumerate(lines[start_idx:], start=start_idx):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split("\t")
        if len(parts) >= 3:
            name = parts[0]
            metadata_json = parts[1]
            file_dir = parts[2]
            
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata = {"raw": metadata_json}
            
            skills.append({
                "name": name,
                "metadata": metadata,
                "file_dir": file_dir,
            })
    
    if not skills:
        print("No skills found in database.")
        return
    
    print(f"\n{'=' * 80}")
    print(f"Skill Database: {database_path}")
    print(f"Total Skills: {len(skills)}")
    print(f"{'=' * 80}\n")
    
    for i, skill in enumerate(skills, 1):
        print(f"{i}. {skill['name']}")
        print(f"   Description: {skill['metadata'].get('description', 'N/A')}")
        print(f"   Source: {skill['metadata'].get('source', 'N/A')}")
        print(f"   Documents: {len(skill['metadata'].get('documents', []))} file(s)")
        print(f"   Directory: {skill['file_dir']}")
        print()


def find_skill_files(base_dir: Path) -> List[Path]:
    """Find all SKILL.md files in the given directory.
    
    Parameters
    ----------
    base_dir : Path
        Base directory to search in.
    
    Returns
    -------
    List[Path]
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
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--view" or sys.argv[1] == "-v":
            # View database mode
            if len(sys.argv) > 2:
                db_path = Path(sys.argv[2])
            else:
                db_path = Path(__file__).parent / "skill_database.tq"
            view_database(db_path)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python build_skill_database.py              - Build the skill database")
            print("  python build_skill_database.py --view       - View the skill database")
            print("  python build_skill_database.py --view <path>- View a specific database file")
            print("  python build_skill_database.py <output_path>- Build database to custom path")
        else:
            # Custom output path for building
            output_file = Path(sys.argv[1])
            create_skill_database(output_file)
            print(f"\nSkill database created successfully at: {output_file}")
    else:
        # Default: build the database
        output_file = Path(__file__).parent / "skill_database.tq"
        create_skill_database(output_file)
        print(f"\nSkill database created successfully at: {output_file}")
