from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime
from typing import Union

# Regex to match the desired folder structure and capture the sequential ID (XXX)
# Example: 20251010_1609_ID005
ID_PATTERN = re.compile(r"^\d{8}_\d{4}_ID(\d{3})$")

def create_run_directory(
    output_dir: Path | str,
) -> Path:
    """
    Creates a new, uniquely named subdirectory within the given output directory.

    The folder name convention is: YYYYMMDD_HHMM_IDXXX, where XXX is a
    zero-padded sequential integer (e.g., ID001, ID002, ...).

    The ID is determined by scanning existing folders matching the pattern and
    finding the next available integer.

    Parameters
    ----------
    output_dir : Path | str
        The main results folder where the new subdirectory will be created.

    Returns
    -------
    Path
        The full path to the newly created subdirectory.
    """
    # 1. Ensure the output_dir is a Path object and exists (creating it if necessary)
    parent_dir = Path(output_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 2. Find the next sequential ID (XXX)
    existing_ids = []

    # Iterate over existing items in the directory
    for item in parent_dir.iterdir():
        if item.is_dir():
            # Check if the folder name matches the ID pattern
            match = ID_PATTERN.match(item.name)
            if match:
                # Extract the XXX part (which is a string like "005") and convert to int
                existing_ids.append(int(match.group(1)))

    # Determine the next ID: max existing ID + 1, starting from 1 if none exist
    next_id = max(existing_ids, default=0) + 1
    next_id_str = f"ID{next_id:03d}" # zero-pad to 3 digits (e.g., 5 -> "ID005")

    # 3. Generate the timestamp string (YYYYMMDD_HHMM)
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M")

    # 4. Construct the final folder name
    folder_name = f"{timestamp_str}_{next_id_str}"

    # 5. Create the final directory path
    run_dir = parent_dir / folder_name

    # 6. Create the directory. exist_ok=False is used to ensure we only create
    # a new folder for the current run, although the sequential ID largely
    # guarantees uniqueness.
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created new run directory: {run_dir}")
        return run_dir
    except FileExistsError:
        # This fallback is highly unlikely to be hit due to the unique sequential ID.
        print(f"Error: Directory {run_dir} already exists. Appending microsecond suffix.")
        folder_name = f"{folder_name}_{now.microsecond}"
        run_dir = parent_dir / folder_name
        run_dir.mkdir(parents=True)
        print(f"Created new run directory: {run_dir}")
        return run_dir
